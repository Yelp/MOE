/*!
  \file gpp_linear_algebra.cpp
  \rst
  Implementations of functions for linear algebra operations; the functionality here largely is a subset of that
  supported by BLAS and LAPACK.

  Linear algebra functions currently do not call libraries like the BLAS/LAPACK because for [currently] small problem
  sizes, overhead kills their performance advantage.  Additionally, for the custom implementations on our specific use
  cases we also gain some performance through more restrictive assumptions on data ordering and no need (due to small
  problem size) for advanced and complex optimizations like blocking.

  However, if/when BLAS is needed, current linear algebra functions are designed to easily map into BLAS calls so they
  can serve as wrappers later.  This also makes it easy to handle BLAS from different vendors and on different computing
  environments (e.g., GPUs, Xeon Phi).

  See gpp_linear_algebra.hpp file docs and (primarily) gpp_common.hpp for a few important implementation notes
  (e.g., restrict, memory allocation, matrix storage style, etc).  Note the matrix looping idiom (gpp_common.hpp,
  item 8) in particular; in summary, we use::

    for (int i = 0; i < m; ++i) {
      y[i] = 0;
      for (int j = 0; j < n; ++j) {
        y[i] += A[j]*x[j];
      }
      A += n;
    }
\endrst*/

#include "gpp_linear_algebra.hpp"

#include <cmath>

#include <algorithm>
#include <limits>
#include <vector>

#include "gpp_common.hpp"
#include "gpp_logging.hpp"

namespace optimal_learning {

/*!\rst
  Slow (compared to computing ``\sqrt(x_i*x_i)``) but stable computation of ``\||vector\|_2``

  Computing ``norm += Square(vector[i])`` can be unsafe due to overflow & precision loss.

  Note that for very large vectors, this method is still potentially inaccurate.  BUT
  we are not using anything nearly that large right now.
  The best solution for accuracy would be to use Kahan summation.
  Around 10^16 elements, this function will fail most of the time.  Around 10^8
  elements, the loss of precision may already be substantial.
\endrst*/
double VectorNorm(double const * restrict vector, int size) noexcept {
  if (unlikely(size == 1)) {
    return std::fabs(vector[0]);
  }
  double scale = 0.0, scaled_norm = 1.0;
  for (int i = 0; i < size; ++i) {
    if (likely(vector[i] != 0.0)) {
      double abs_xi = std::fabs(vector[i]);
      if (scale < abs_xi) {
        double temp = scale/abs_xi;
        scaled_norm = 1.0 + scaled_norm * (temp*temp);
        scale = abs_xi;
      } else {
        double temp = abs_xi/scale;
        scaled_norm += temp*temp;
      }
    }
  }
  return scale * std::sqrt(scaled_norm);
}

void MatrixTranspose(double const * restrict matrix, int num_rows, int num_cols, double * restrict transpose) noexcept {
  for (int i = 0; i < num_rows; ++i) {
    for (int j = 0; j < num_cols; ++j) {
      transpose[j] = matrix[j*num_rows + i];
    }
    transpose += num_cols;
  }
}

void ZeroUpperTriangle(int size, double * restrict matrix) noexcept {
  for (int i = 0; i < size; ++i) {
    for (int j = 0; j < i; ++j) {
      matrix[j] = 0.0;
    }
    matrix += size;
  }
}

/*!\rst
  Cholesky factorization, ``A = L * L^T`` (see Smith 1995 or Golub, Van Loan 1983, etc.)
  This implementation uses the outer-product formulation.  The outer-product version is
  2x slower than gaxpy or dot product style implementations; this is to remain consistent with
  Smith 1995's formulation of the gradient of cholesky.

  This implemention is not optimized nor does it pivot when symmetric, indefinite matrices
  or poorly conditioned SPD matrices are detected.

  Instead, non-SPD matrices trigger an error printed to stdout.

  Should be the same as BLAS call:
  ``dpotrf('L', size_m, A, size_m, &info);``
  Implementation is similar to ``dpotf2``, the unblocked version (same arg list as ``dpotrf``).
\endrst*/
// TODO(GH-172): change this to be gaxpy or (block) dot-prod style
// to improve performance & numerical characteristics.
int ComputeCholeskyFactorL(int size_m, double * restrict chol) noexcept {
  double * restrict chol_temp = chol;
  // Apply outer-product-based Cholesky algorithm: 1/3*N^3 + O(N^2)
  // Here, L_{ij} = chol[j*size_m + i] is the input matrix (on input) and the cholesky factor of that matrix (on exit).
  // Define a macro specifying the data layout assumption on L_{ij}. The macro simplifies complex indexing
  // so that OL_CHOL(i, j) reads just like L_{ij}.
#define OL_CHOL(i, j) chol[((j)*size_m + (i))]
  double A_kk;
  for (int k = 0; k < size_m; ++k) {
    if (likely(chol_temp[k] > 1.0e-16)) {
      // L_{kk} = \sqrt(A_{kk})
      A_kk = std::sqrt(chol_temp[k]);
      chol_temp[k] = A_kk;

      // adjust lead column
      // L_{jk} = L_{jk}/L_{kk}, j = k+1..N
      for (int j = k+1; j < size_m; ++j) {
        chol_temp[j] /= A_kk;
      }

      // row updates
      // L_{ij} = L_{ij} - L_{ik}*L_{jk}, j=k+1..N and i=j..N
      for (int j = k+1; j < size_m; ++j) {  // over columns
        for (int i = j; i < size_m; ++i) {  // over rows
          OL_CHOL(i, j) = OL_CHOL(i, j) - OL_CHOL(i, k) * OL_CHOL(j, k);
        }
      }
#undef OL_CHOL
    } else {
      // We fail if the matrix is singular. In the outer-product formulation here,
      // you can ignore the "0" diagonal entry and continue, which produces a
      // semi-positive definite factorization (see Golub, Van Loan 1983).
      OL_ERROR_PRINTF("cholesky matrix singular %.18E ", chol_temp[k]);
      return k + 1;
    }
    chol_temp += size_m;
  }

  return 0;
}

/*!\rst
  Solve ``A*x = b`` or ``A^T*x = b`` when ``A`` is lower triangular IN-PLACE.
  Uses the standard "backsolve" technique, instead of forming ``A^-1`` which is
  VERY poorly conditioned.  Backsolve, however, is backward-stable.

  See .h file docs for information on "lda".

  Should be equiv to BLAS call:
  ``dtrsv('L', 'N', 'N', size_m, A, lda, x, 1);``
\endrst*/
void TriangularMatrixVectorSolve(double const * restrict A, char trans, int size_m, int lda, double * restrict x) noexcept {
  double temp;
  if (trans == 'N') {  // solve A*x = b, A lower tri
    // work forward thru matrix since the first unknown has the form A_{00}*x_0 = b_0
    for (int j = 0; j < size_m; ++j) {
      // if b_j == 0, then x_j = 0, so no work needs to be done
      if (x[j] != 0.0) {
        // solve j-th value of x
        x[j] /= A[j];
        temp = x[j];

        // remove solved value from the rest of RHS
        for (int i = j+1; i < size_m; ++i) {
          x[i] = x[i] - temp*A[i];
        }
      }
      A += lda;
    }
  } else {  // trans == T; solve A^T * x = b, A is lower tri
    // 'T' version is basically the 'N' version running backwards
    // need to work backwards since now its the LAST unknown that's easily computed
    A += lda*(size_m-1);
    for (int j = size_m-1; j >= 0; --j) {  // i.e., j >= 0
      temp = x[j];
      // could run this loop forward but backward is more similar to the "by-hand" procedure
      for (int i = size_m-1; i >= j+1; --i) {
        temp -= A[i]*x[i];
      }
      temp /= A[j];
      x[j] = temp;
      A -= lda;
    }
  }  // end if over 'T'
}


/*!\rst
  Calls dtrsv on each column of ``X``, solving ``A * X_i = B_i`` (``X_i`` being ``i``-th column of ``X``).
  Does not use blocking or any other optimization techniques.

  Should be equiv to BLAS call:
  ``dtrsm('L', 'L', 'N', 'N', size_m, size_n, 1.0, A, lda, B, size_m);``
\endrst*/
void TriangularMatrixMatrixSolve(double const * restrict A, char trans, int size_m, int size_n, int lda, double * restrict X) noexcept {
  for (int k = 0; k < size_n; ++k) {
    TriangularMatrixVectorSolve(A, trans, size_m, lda, X);
    X += size_m;
  }
}

/*!\rst
  Computes ``A^-1``, the inverse of ``A`` when ``A`` has been previously cholesky-factored.
  Only the lower triangle of ``A`` is read.

  Computes inverse by successive ``x_i = A \ e_i`` operations, where ``e_i`` is the unit vector
  with a 1 in the ``i``-th entry.
  Implementation saves computation (factor fo 2) by ignoring all leading zeros in ``x_i``.  So ``x_0 = A \ e_0``
  is ``\approx m^2`` operations, ``x_1 = A \ e_1`` is ``\approx (m-1)^2`` operations, ..., and
  ``x_{m-1} = A \ e_{m-1}`` is 1 operation.

  This is NOT backward-stable and should NOT be used!  Substantial superfluous
  numerical error can occur for poorly conditioned matrices.
  Caveat: may have utility if you are very certain of what you are doing in the face of [severe] loss of precision
\endrst*/
void TriangularMatrixInverse(double const * restrict matrix, int size_m, double * restrict inv_matrix) noexcept {
  double * restrict inv_matrix_ptr = inv_matrix;
  int cur_size = size_m;

  for (int i = 0; i < size_m; ++i) {
    // zero i-th column
    std::fill(inv_matrix, inv_matrix+size_m, 0.0);

    // set inv_matrix[i][i] to 1.0: creates unit vector
    inv_matrix_ptr = inv_matrix + i;
    inv_matrix_ptr[0] = 1.0;

    // backsolve matrix against previous unit vector
    // cur_size is equivalently size_m - i

    TriangularMatrixVectorSolve(matrix, 'N', cur_size, size_m, inv_matrix_ptr);

    matrix += size_m + 1;  // get next main submatrix, i.e., matrix(1:n,1:n), then matrix(2:n,2:n),etc
    inv_matrix += size_m;
    --cur_size;
  }
}

/*!\rst
  Special case of GeneralMatrixVectorMultiply.  As long as A has zeros in the strict upper-triangle,
  GeneralMatrixVectorMultiply will work too (but take ``>= 2x`` as long).

  Computes results IN-PLACE.
  Avoids accessing the strict upper triangle of A.

  Should be equivalent to BLAS call:
  ``dtrmv('L', trans, 'N', size_m, A, size_m, x, 1);``
\endrst*/
void TriangularMatrixVectorMultiply(double const * restrict A, char trans, int size_m, double * restrict x) noexcept {
  double temp;

  if ('N' == trans) {  // compute x = A * x
    // have to work backwards to permit computing results in-place
    // this is analogous to (but reversed from) TriangularMatrixVectorSolve
    A += size_m * (size_m-1);
    for (int j = size_m-1; j >= 0; --j) {  // i.e., j >= 0
      if (x[j] != 0.0) {
        temp = x[j];
        for (int i = size_m-1; i >= j+1; --i) {  // this loop could be run forward; maybe faster?
          // handles sub-diagonal contributions from j-th column
          x[i] += temp*A[i];
        }
        x[j] *= A[j];  // handles j-th on-diagonal component
        A -= size_m;
      }
    }
  } else {  // assume trans == 'T', compute x = A^T * x
    // now we treat the i-th column of A as its i-th row to account for transpose
    // transpose also allows us to work forward
    for (int j = 0; j < size_m; ++j) {
      temp = x[j] * A[j];  // first iteration unrolled
      for (int i = j+1; i < size_m; ++i) {  // same as i=j with temp=0 to start
        temp += A[i]*x[i];  // treat j-th column of A as j-th row
      }
      x[j] = temp;
      A += size_m;
    }
  }  // end if over 'N' and 'T'
}

/*!\rst
  Special case of GeneralMatrixVectorMultiply for symmetric A (need not be SPD).
  As long as A is stored fully (i.e., upper triangle is valid),
  GeneralMatrixVectorMultiply will work too (but take ``>= 2x`` as long).

  Avoids accessing the strict upper triangle of A.

  Should be equivalent to BLAS call:
  ``dsymv('L', size_m, 1.0, A, size_m, x, 1, 0.0, y, 1);``
\endrst*/
void SymmetricMatrixVectorMultiply(double const * restrict A, double const * restrict x, int size_m, double * restrict y) noexcept {
  std::fill(y, y+size_m, 0.0);
  double temp1 = x[0], temp2 = 0.0;

  // only look at triangle
  // analogous to simultaneously computing A*x and A^T*x for non-symmetric matrices
  // since the j-th loop handles row (dot-product) and column (scaling) updates at once
  for (int j = 0; j < size_m; ++j) {
    temp1 = x[j];
    temp2 = 0.0;
    y[j] += temp1*A[j];  // handling diagonal term as part of jth row
    for (int i = j+1; i < size_m; ++i) {
      y[i] += temp1*A[i];  // contribution from jth column of A
      temp2 += A[i]*x[i];  // contribution from jth row of A
    }
    y[j] += temp2;
    A += size_m;
  }
}

/*!\rst
  Computes matrix-vector product ``y = alpha * A * x + beta * y`` or ``y = alpha * A^T * x + beta * y``.
  Since ``A`` is stored column-major, we treat the matrix-vector product as a weighted sum
  of the columns of ``A``, where ``x`` provides the weights.

  That is, a matrix-vector product can be thought of as: (``trans = 'T'``)
  ``[  a_row1  ][   ]``
  ``[  a_row2  ][ x ]``
  ``[    ...   ][   ]``
  ``[  a_rowm  ][   ]``
  That is, ``y_i`` is the dot product of the ``i``-th row of ``A`` with ``x``.

  OR the "dual" view: (``trans = 'N'``)
  ``[        |        |     |        ][ x_1 ]``
  ``[ a_col1 | a_col2 | ... | a_coln ][ ... ] = x_1*a_col1 + ... + x_n*a_coln``
  ``[        |        |     |        ][ x_n ]``
  That is, ``y`` is the weighted sum of columns of ``A``.

  Should be equivalent to BLAS call:
  ``dgemv(trans, size_m, size_n, alpha, A, size_m, x, 1, beta, y, 1);``
\endrst*/
void GeneralMatrixVectorMultiply(double const * restrict A, char trans, double const * restrict x, double alpha, double beta, int size_m, int size_n, int lda, double * restrict y) noexcept {
  double temp;

  // y = beta*y
  if (beta != 1.0) {
    // length of y changes depending on transposed-ness
    int leny = (trans == 'N')*size_m + (trans == 'T')*size_n;
    if (likely(beta == 0.0)) {
      std::fill(y, y+leny, 0.0);
    } else {
      VectorScale(leny, beta, y);
    }
  }

  if (likely(trans == 'N')) {
    for (int i = 0; i < size_n; ++i) {
      temp = alpha*x[i];
      for (int j = 0; j < size_m; ++j) {
        y[j] += A[j]*temp;
      }
      A += lda;
    }
  } else {
    for (int i = 0; i < size_n; ++i) {
      temp = 0.0;
      for (int j = 0; j < size_m; ++j) {
        temp += A[j]*x[j];
      }
      y[i] += alpha*temp;
      A += lda;
    }
  }
}

/*!\rst
  Matrix-matrix product ``C = alpha * op(A) * B + beta * C``, where ``op(A)`` is ``A`` or ``A^T``.
  Does so by computing matrix-vector products of ``A`` with each column of ``B``
  (to generate corresponding column of ``C``).

  Does not use blocking or other advanced optimization techniques.

  Should be equivalent to BLAS call:
  ``dgemm('N', 'N', size_m, size_n, size_k, alpha, A, size_m, B, size_k, beta, C, size_m);``
\endrst*/
void GeneralMatrixMatrixMultiply(double const * restrict Amat, char transA, double const * restrict Bmat, double alpha, double beta, int size_m, int size_k, int size_n, double * restrict Cmat) noexcept {
  if (transA == 'N') {
    for (int j = 0; j < size_n; ++j) {
      GeneralMatrixVectorMultiply(Amat, 'N', Bmat, alpha, beta, size_m, size_k, size_m, Cmat);
      Bmat += size_k;
      Cmat += size_m;
    }
  } else {
    for (int j = 0; j < size_n; ++j) {
      GeneralMatrixVectorMultiply(Amat, 'T', Bmat, alpha, beta, size_k, size_m, size_k, Cmat);
      Bmat += size_k;
      Cmat += size_m;
    }
  }
}

/*!\rst
  Computes ``A^-1`` by cholesky-factoring ``A = L * L^T``, computing ``L^-1``, and then
  computing ``A^-1 = L^-T * L^-1``.

  This is NOT backward-stable and should NOT be used!  Substantial superfluous
  numerical error can occur for poorly conditioned matrices.
  Caveat: may have utility if you are very certain of what you are doing in the face of [severe] loss of precision
\endrst*/
void SPDMatrixInverse(double const * restrict chol_matrix, int size_m, double * restrict inv_matrix) noexcept {
  std::vector<double> L_inv(size_m*size_m);
  TriangularMatrixInverse(chol_matrix, size_m, L_inv.data());
  GeneralMatrixMatrixMultiply(L_inv.data(), 'T', L_inv.data(), 1.0, 0.0, size_m, size_m, size_m, inv_matrix);
}

int ComputePLUFactorization(int r, int * restrict pivot, double * restrict A) noexcept {
  // TODO(GH-50): after linking to BLAS, this code should only run for r < 64 or so
  // Equivalent LAPACK call:
  // dgetrf_(&r, &r, A, &r, pivot, &info);
  if (unlikely(r == 1)) {
    if (unlikely(std::fabs(A[0]) < std::numeric_limits<double>::epsilon())) {
      return 1;
    }
    pivot[0] = 1;
    // 1x1 matrix, nothing else to do
    return 0;
  }

  double temp;
  double temp1;
  double temp2;

  double * restrict Apos;
  double * restrict Apos1;
  double * restrict Apos2;

  int mod = r % 2u;
  int mod2 = mod;

  for (int k = 0; k < r; ++k) {  // loop over columns
    // prefetch(Apos+k+r*8);
    Apos = &A[k*r];

    // find maximum (absolute value) entry below pivot
    double Acolmax = std::fabs(Apos[k]);  // the diagonal element of col k
    int kmax = k;  // start on the diagonal element
    for (int i = (k+1); i < r; ++i) {
      if (std::fabs(Apos[i]) > Acolmax) {
        Acolmax = std::fabs(Apos[i]);
        kmax = i;
      }
    }

    pivot[k] = kmax + 1;  // shift 1 to match LAPACK (Fortran) index-from-1
    // switch rows k and kmax if necessary
    if (kmax != k) {
      int ir = 0;
      int i = 0;
      if (mod != 0) {
        temp1 = A[ir + k];
        A[ir + k] = A[ir + kmax];
        A[ir + kmax] = temp1;
        ir += r;
        ++i;
      }
      int jr = ir + r;
      for ( ; i < r; i += 2) {
        temp1 = A[ir + k];
        A[ir + k] = A[ir + kmax];
        A[ir + kmax] = temp1;

        temp2 = A[jr + k];
        A[jr + k] = A[jr + kmax];
        A[jr + kmax] = temp2;

        ir += 2*r;
        jr += 2*r;
      }
    }
    // equivalent BLAS call
    // if( kmax != k) dswap_(&r, &A[k], &r, &A[kmax], &r);

    // check if near-singular
    double Akk = A[k*r + k];  // CANNOT USE Apos HERE!!
    // Using Apos here allows the compiler to execute this line as soon
    // as the loop starts, which is invalid.

    if (unlikely(std::fabs(Akk) < std::numeric_limits<double>::min())) {
      OL_VERBOSE_PRINTF("ERROR: PLU: Matrix Singular, A[%d,%d] = %.18E\n", k, k, Akk);
      return k+1;
    }

    // Scale L by the pivot element
    Akk = 1/Akk;  // Set Akk to its reciprocal to call BLAS
    VectorScale(r - (k + 1), Akk, Apos + k+1);

    // compute outer product update:
    // x = kth col, rows k+1:r of A (column vec)
    // y = kth row, cols k+1:r of A (row vec)
    // A(k+1:r,k+1:r) -= x*y
    int j = k+1;
    mod2 ^= 1;  // same as: mod2 = (unsigned) (r - j) % 2u;
    // because (r-j)%2 flips every iteration.  ^=1 flips the last bit
    Apos1 = &A[j*r];

    if (mod2 != 0) {
      temp1 = Apos1[k];
      for (int i = (k+1); i < r; ++i) {
        Apos1[i] -= temp1*Apos[i];
      }
      Apos1 += r;
      ++j;
    }

    Apos2 = Apos1 + r;
    for ( ; j < r; j += 2) {
      // prefetch(Apos1+32*8);
      // prefetch(Apos2+32*8);
      // prefetch(Apos1+k+r*2*8);
      // prefetch(Apos2+k+r*2*8);
      temp1 = Apos1[k];
      temp2 = Apos2[k];

      for (int i = (k+1); i < r; ++i) {  // loop over rows
        temp = Apos[i];
        Apos1[i] -= temp1*temp;
        Apos2[i] -= temp2*temp;
      }
      Apos1 += 2*r;
      Apos2 += 2*r;
    }
  }
  return 0;
}

/*!\rst
  Solves the system ``P * L * U * x = b``.

  1. Apply the permutation ``P`` (pivot) to ``b``.
  2. Forward-substitute to solve ``Ly = Pb``.  This loop is unrolled 4 times for speed.
  3. Backward-substitute to solve ``Ux = y``.  This loop is also unrolled 4 times.
\endrst*/
void PLUMatrixVectorSolve(int r, double const * restrict LU, int const * restrict pivot, double * restrict b) noexcept {
  // First, swap rows of the vector b according to pivot array P;
  // i.e. solve Px=b, storing the result in b
  for (int k = 0; k < r; ++k) {
    int kswap = pivot[k] - 1;  // shift 1 to match LAPACK (Fortran) index-from-1
    // switch rows k and kswap if necessary
    if (kswap != k) {
      double temp = b[k];
      b[k] = b[kswap];
      b[kswap] = temp;
    }
  }
  // equivalent BLAS call
  // dlaswp_(&int_one, b, &r, &int_one, &r, pivot, &inc_one);

  // TODO(GH-50): after linking to BLAS, this should only run for roughly r < 250
  // Equivalent LAPACK call:
  // dgetrs_('N', r, 1, LU, r, pivot, b, r, &info)
  // which is just (BLAS calls):
  // dtrsv_(&lower, &no_trans, &unit, &r, LU, &r, b, &inc_one);
  // dtrsv_(&upper, &no_trans, &no_unit, &r, LU, &r, b, &inc_one);
  double b_k;
  double b_k2;
  double b_k3;
  double b_k4;

  double const * restrict LUpos1;
  double const * restrict LUpos2;
  double const * restrict LUpos3;
  double const * restrict LUpos4;

  // Loops are unrolled 4 times
  // Consider factor 2 if problems are small (r<40ish)
  int rmod4 = r % 4u;
  int rx4 = r*4;
  int k = 0;
  LUpos1 = &LU[0];
  // Solve Lx = b via Forward Subs, storing the result in b
  // if the total size is not a multiple of 4, handle the first few so we can unroll
  for ( ; k < rmod4; ++k) {
    b_k = b[k];  // should be = b(k)/A(k,k), but L is unit diagonal!
    for (int i = k+1; i < r; ++i) {
      b[i] -= b_k*LUpos1[i];
    }
    LUpos1 += r;
  }
  LUpos2 = LUpos1 + r;
  LUpos3 = LUpos1 + 2*r;
  LUpos4 = LUpos1 + 3*r;
  for ( ; k < r; k += 4) {
    // prefetch(LUpos1+k+1+r*4*8);
    // prefetch(LUpos2+k+2+r*4*8);
    // prefetch(LUpos3+k+3+r*4*8);
    // prefetch(LUpos4+k+4+r*4*8);
    // lead-in for loop-unrolling: each outer loop (over k) iteration examines a trapezoidal region of the matrix:
    // |\  <--triangular
    // | \     tip (4x4 here)
    // |  |
    // |  |
    // |__|
    // where the triangluar 'tip' is 4x4 (due to 4x unrolling).  The rectangular region is r - (k+4) rows and 4 columns.

    // The lead-in computes the components of the solution vector (k..k+3) corresponding to the equations in the triangular
    // tip.  This segment is inherently sequential (parallel techniques exist but are extremely complex) and
    // difficult to unroll succintly.

    // nominally this would be LUpos1[k]*b[k], BUT L is known to be unit-diagonal and that information is NOT stored in LU.
    b_k = b[k];

    b[k+1] -= b[k]*LUpos1[k+1];
    b_k2 = b[k+1];
    b[k+2] = b[k+2] - b[k]*LUpos1[k+2] - b[k+1]*LUpos2[k+2];
    b_k3 = b[k+2];
    b[k+3] = b[k+3] - b[k]*LUpos1[k+3] - b[k+1]*LUpos2[k+3] - b[k+2]*LUpos3[k+3];
    b_k4 = b[k+3];
    // unroll substituting the newly found solutions, b[k]..b[k+3], into the remaining equations
    for (int i = k+4; i < r; ++i) {
      b[i] = b[i] - b_k*LUpos1[i] - b_k2*LUpos2[i] - b_k3*LUpos3[i] - b_k4*LUpos4[i];
    }

    LUpos1 += rx4;
    LUpos2 += rx4;
    LUpos3 += rx4;
    LUpos4 += rx4;
  }
  // b(end) = b(end)/L(end,end) not necessary b/c unit diagonal again

  // equivalent BLAS call
  // dtrsv_(&lower, &no_trans, &unit, &r, LU, &r, b, &inc_one);

  // Solve Ux = b via Backward Subs, storing the result in b
  // the structure of this loop is nearly identical to the previous block.
  // Notable differences are that U is not unit diagonal, and now we are back-substituting so
  // we walk in reverse through U.
  // Now, we are working in a region:
  //  __
  // |  |
  // |  |
  // |  |
  //  \ |  <--triangular
  //   \|     base (4x4 here, again due to unrolling)
  k = r-1;
  LUpos1 = &LU[k*r];
  // handle columns of U that would not be caught by unrolling
  for ( ; k >= r-rmod4; --k) {
    b[k] /= LUpos1[k];  // = b(k)/U(k,k)
    b_k = b[k];
    for (int i = 0; i < k; ++i) {
      b[i] -= b_k*LUpos1[i];
    }
    LUpos1 -= r;
  }
  LUpos2 = LUpos1 - r;
  LUpos3 = LUpos1 - 2*r;
  LUpos4 = LUpos1 - 3*r;
  for ( ; k > 0; k -= 4) {
    // prefetch(LUpos1+r*4*8);
    // prefetch(LUpos2+r*4*8);
    // prefetch(LUpos3+r*4*8);
    // prefetch(LUpos4+r*4*8);

    b[k] /= LUpos1[k];
    b_k = b[k];

    b[k-1] -= b[k]*LUpos1[k-1];

    b[k-1] /= LUpos2[k-1];
    b_k2 = b[k-1];

    b[k-2] = b[k-2] - b[k]*LUpos1[k-2] - b[k-1]*LUpos2[k-2];
    b[k-2] /= LUpos3[k-2];
    b_k3 = b[k-2];

    b[k-3] = b[k-3] - b[k]*LUpos1[k-3] - b[k-1]*LUpos2[k-3] - b[k-2]*LUpos3[k-3];
    b[k-3] /= LUpos4[k-3];
    b_k4 = b[k-3];
    for (int i = 0; i < k-3; ++i) {
      b[i] = b[i] - b_k*LUpos1[i] - b_k2*LUpos2[i] - b_k3*LUpos3[i] - b_k4*LUpos4[i];
    }

    LUpos1 -= rx4;
    LUpos2 -= rx4;
    LUpos3 -= rx4;
    LUpos4 -= rx4;
  }
  // equivalent BLAS call:
  // dtrsv_(&upper, &no_trans, &no_unit, &r, LU, &r, b, &inc_one);
}

}  // end namespace optimal_learning
