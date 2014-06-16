/*!
  \file gpp_linear_algebra.hpp
  \rst
  This file provides low level linear algebra functionality in support of the other gpp_* components.  The functions here
  generally wrap BLAS (levels 1 through 3) and LAPACK functionality as well as a few utilities for convenience/debugging
  (e.g., matrix print outs).

  First, look over gpp_common.hpp for general comments on loop layouts, storage formats, and shorthand as well
  as definitions of standard notation.

  The functions here wrap BLAS/LAPACK functionality to make it convenient/easy to switch between different library
  implementations (e.g., different vendors or even on a GPU).  They also provide an opportunity to avoid associated
  overhead for "small" problem sizes.  These functions wrap BLAS:

  * Level 1: ``O(n)`` operations; vector scale, dot product, etc.
  * Level 2: ``O(n^2)`` operations; Matrix-vector multiplies (+ special cases) and triangular solves
  * Level 3: ``O(n^3)`` operations: matrix-matrix multiplies and triangular solves

  and LAPACK:

  * O(n^3), triangular factorization routines (PLU, Cholesky)
  * O(n^3), matrix inverse

  Matrix storage formats are always column-major as prescribed in gpp_common.hpp.
  We also only deal with *lower* triangular matrices here; these are stored in the lower triangle.  The contents
  of the upper triangle are ignored and can technically have any value, even NaN.
  See PLU for further description of its special output format.
\endrst*/

#ifndef MOE_OPTIMAL_LEARNING_CPP_GPP_LINEAR_ALGEBRA_HPP_
#define MOE_OPTIMAL_LEARNING_CPP_GPP_LINEAR_ALGEBRA_HPP_

#include "gpp_common.hpp"

namespace optimal_learning {

/*!\rst
  Computes ``\|x\|_2`` in a reasonably (see implementation notes) accurate and stable way.

  Slower than the naive implementation due to scaling done to prevent overflow & reduce precision loss.

  \param
    :vector[size]: the vector x
    :size: number of elements in x
  \return
    The vector 2-norm (aka Euclidean norm) of x.
\endrst*/
double VectorNorm(double const * restrict vector, int size) noexcept OL_PURE_FUNCTION OL_NONNULL_POINTERS OL_WARN_UNUSED_RESULT;

/*!\rst
  Transposes a 2D matrix. ``O(num_rows*num_cols) = O(n^2)`` for square matrices
  Would have no effect on symmetric matrices.

  For example, ``A[3][4] = [4 53 81 32 12 2 5 8 93 2 1 0]``
  becomes      ``A[4][3] = [4 32 5 2 53 12 8 1 81 2 93 0]``

  \param
    :matrix[num_rows][num_cols]: matrix to be transposed
    :num_rows: number of rows in matrix
    :num_cols: number of columns in matrix
  \output
    :transpose[num_cols][num_rows]: transpose of matrix
\endrst*/
void MatrixTranspose(double const * restrict matrix, int num_rows, int num_cols, double * restrict transpose) noexcept OL_NONNULL_POINTERS;

/*!\rst
  Zeroes the strict upper triangle of a matrix (assuming column-major storage)

  \param
    :size: dimension of matrix
    :matrix[size][size]: matrix whose upper tri is to be zeroed (on input)
  \output
    :matrix[size][size]: lower triangular part of input matrix
\endrst*/
void ZeroUpperTriangle(int size, double * restrict matrix) noexcept OL_NONNULL_POINTERS;

/*!\rst
  Multiplies first ``size`` elements of ``vector`` by ``alpha``, ``vector := vector*alpha``.

  Should be equivalent to BLAS call:
  ``dscal(size, alpha, vector, 1);``

  \param
    :size: number of elements in vector
    :alpha: number to scale by
    :vector[size]: vector to scale
  \output
    :vector[size]: vector with elements scaled
\endrst*/
inline OL_NONNULL_POINTERS void VectorScale(int size, double alpha, double * restrict vector) noexcept {
  for (int i = 0; i < size; ++i) {
    vector[i] *= alpha;
  }
}

/*!\rst
  Computes ``y_i = alpha * x_i + y_i``; ``y`` is modified in-place.

  \param
    :size: number of elements in ``x, y``
    :alpha: quantity to scale by
    :vec1[size]: ``x``, vector to scale and add to ``y``
    :vec2[size]: ``y``, vector to add to
  \output
    :vec2[size]: input ``y`` plus ``alpha*x``
\endrst*/
inline OL_NONNULL_POINTERS void VectorAXPY(int size, double alpha, double const * restrict vec1, double * restrict vec2) noexcept {
  for (int i = 0; i < size; ++i) {
    vec2[i] += alpha*vec1[i];
  }
}

/*!\rst
  Computes dot product between two vectors.

  Equivalent BLAS call:
  ``ddot(size, vector1, 1, vector2, 1);``

  \param
    :vector1[size]: first vector
    :vector2[size]: second vector
    :size: length of vectors
  \return
    dot/inner product, ``<vector1, vector2>``
\endrst*/
inline OL_PURE_FUNCTION OL_NONNULL_POINTERS OL_WARN_UNUSED_RESULT double DotProduct(double const * restrict vector1, double const * restrict vector2, int size) noexcept {
  double sum = 0.0;
  for (int i = 0; i < size; ++i) {
    sum += vector1[i]*vector2[i];
  }
  return sum;
}

/*!\rst
  Computes the cholesky factorization of a symmetric, positive-definite (SPD) matrix,
  ``A = L * L^T``; ``A`` is the input matrix, ``L`` is the (lower triangular) cholesky factor.
  ``O(n^3)`` operations.
  No inputs may be nullptr.

  A must be SPD.  Calling this function on an indefinite matrix will produce a
  nonsensical ``L.``
  Calling this function on a semi-definite matrix may result in a severe loss of precision
  as well as inaccurate ``L``.

  The strict upper triangle of chol is NOT accessed.

  \param
    :size_m: dimension of matrix
    :chol[size_m][size_m]: SPD (square) matrix (``A``) (on entry)
  \output
    :chol[size_m][size_m]: cholesky factor of ``A`` (``L``).  ``L`` is stored in the lower triangle
                           of ``A``.  Do not acccess the upper triangle of ``L``. (on exit)
  \return
    0 if successful. Otherwise the matrix is NOT positive definite and this returns ``i``, the
    index of the ``i``-th leading minor that is not positive definite.
\endrst*/
int ComputeCholeskyFactorL(int size_m, double * restrict chol) noexcept OL_NONNULL_POINTERS OL_WARN_UNUSED_RESULT;

/*!\rst
  Solves the system ``A*x = b`` or ``A^T * x = b`` when ``A`` is lower triangular. ``A`` must be nonsingular.
  Before calling, ``x`` holds the RHS, ``b``.  After return, ``x`` will be OVERWRITTEN with
  the solution.
  No inputs may be nullptr.

  DOES NOT form ``A^-1`` explicitly.

  Nonsensical output for singular ``A``.

  \param
    :A[size_m][size_m]: input to be solved; must be lower triangular and non-singular
    :trans: 'N' to solve ``A * x = b``, 'T' to solve ``A^T * x = b``
    :size_m: dimension of ``A``
    :lda: the first dimension of ``A`` as declared by the caller; ``lda >= size_m``
    :x[size_m]: the RHS vector, ``b``
  \output
    :x[size_m]: the solution, ``A\b``.
\endrst*/
void TriangularMatrixVectorSolve(double const * restrict A, char trans, int size_m, int lda, double * restrict x) noexcept OL_NONNULL_POINTERS;


/*!\rst
  Solve ``A * X = B`` or ``A^T * X = B`` (``A, X, B`` matrices) when ``A`` is lower triangular.
  Solves IN-PLACE.

  \param
    :A[size_m][size_m]: input to be solved; must be lower triangular and non-singular
    :trans: 'N' to solve ``A * X = B``, 'T' to solve ``A^T * X = B``
    :size_m: dimension of ``A``
    :lda: the first dimension of ``A`` as declared by the caller; ``lda >= size_m``
    :X[size_m]: the RHS matrix, ``B``
  \output
    :X[size_m]: the solution, ``A\B``.
\endrst*/
void TriangularMatrixMatrixSolve(double const * restrict A, char trans, int size_m, int size_n, int lda, double * restrict X) noexcept OL_NONNULL_POINTERS;

/*!\rst
  Solves ``A * x = b`` IN-PLACE, where ``A`` has been previously cholesky-factored (``A = L * L^T``) such that
  the lower triangle of ``A`` contains ``L``.
  Consists of two calls to TriangularMatrixVectorSolve. As in that function, before calling, ``x`` holds the
  RHS, ``b``.  After return, ``x`` will be OVERWRITTEN with the solution.
  No inputs may be nullptr.

  Math:
  ``A * x = b``
  ``L * L ^T * x = b``
  ``L^T * x = L \ b  (dtrsv, no transpose)``
  ``x = L^T \ (L \ b)  (dtrsv, transpose)``

  Should be equivalent to BLAS call:
  ``dpotrs('L', size_m, 1, A, size_m, x, size_m, &info);``

  \param
    :A[size_m][size_m]: cholesky-factored system of equations such that its lower triangle
                        contains ``L``.  i.e., result 'chol' from ComputeCholeskyFactorL(A_full, size_m, chol)
    :size_m: dimension of ``A``
    :x[size_m]: the RHS vector, ``b``
  \output
    :x[size_m]: the solution, ``A\b``.
\endrst*/
inline OL_NONNULL_POINTERS void CholeskyFactorLMatrixVectorSolve(double const * restrict A, int size_m, double * restrict x) noexcept {
  TriangularMatrixVectorSolve(A, 'N', size_m, size_m, x);
  TriangularMatrixVectorSolve(A, 'T', size_m, size_m, x);
}

/*!\rst
  Same as CholeskyFactorLMatrixVectorSolve except this accepts matrix RHSs; it solves multi-RHS linear systems of the form:
  ``A * X = B``.
  ``A`` must have been Cholesky-factored (``L * L^T = A``) beforehand, and it must hold the factor ``L`` in its lower trinagle.
  Usually this is obtained via ComputeCholeskyFactorL().

  Note that this operation is done IN-PLACE.  ``X`` initially holds ``B`` and is overwritten with the solution.

  MATH:
  This is analogous to CholeskyFactorLMatrixVectorSolve; see that function for further details.

  Should be equivalent to BLAS call:
  ``dpotrs('L', size_m, size_n, A, size_m, X, size_m, &info);``

  \param
    :A[size_m][size_m]: non-singular matrix holding ``L``, the cholesky factor of ``A``, in its lower triangle
    :size_m: number of rows of ``A, X, B``; number of columns of ``A``
    :size_n: number of columns of ``X, B``
    :X[size_m][size_n]: matrix of RHS vectors, ``B```
  \output
    :X[size_m][size_n]: matrix of solutions, ``A\B``
\endrst*/
inline OL_NONNULL_POINTERS void CholeskyFactorLMatrixMatrixSolve(double const * restrict A, int size_m, int size_n, double * restrict X) noexcept {
  TriangularMatrixMatrixSolve(A, 'N', size_m, size_n, size_m, X);
  TriangularMatrixMatrixSolve(A, 'T', size_m, size_n, size_m, X);
}

/*!\rst
  Computes ``A * x`` or ``A^T * x`` in-place.
  ``A`` must be lower-triangular.  The vector ``x`` is OVERWRITTEN with the result before return.

  \param
    :A[size_m][size_m]: lower triangular matrix to be multiplied
    :trans: 'N' for ``A * x``, 'T' for ``A^T * x``
    :size_m: dimension of ``A, x``
    :x[size_m]: vector to multiply by ``A``
  \output
    :x[size_m]: the product ``A * x`` or ``A^T * x``
\endrst*/
void TriangularMatrixVectorMultiply(double const * restrict A, char trans, int size_m, double * restrict x) noexcept OL_NONNULL_POINTERS;

/*!\rst
  Computes ``y = A * x`` (or equivalently ``y = A^T * x``).  This is NOT done in-place.
  A must be symmetric.  Only the lower triangular part of A is read, so there is no need
  to store the duplicate values when using this function.
  No inputs may be nullptr.

  \param
    :A[size_m][size_m]: symmetric matrix to be multiplied
    :x[size_m]: vector to multiply by ``A``
    :size_m: dimension of ``A, x``
  \output
    :y[size_m]: the product ``A * x``
\endrst*/
void SymmetricMatrixVectorMultiply(double const * restrict A, double const * restrict x, int size_m, double * restrict y) noexcept OL_NONNULL_POINTERS;

/*!\rst
  Computes ``y = alpha * A * x + beta * y`` or ``y = alpha * A^T * x + beta * y``.  This is NOT done in-place
  A can be any 2 dimensional matrix (no requirements on triangularity, symmetry, etc)
  No inputs may be nullptr.

  \param
    :A[size_m][size_n]: input matrix to be multiplied
    :trans: whether to multiply by ``A`` ('N') or ``A^T`` ('T')
    :x[size_n OR size_m]: vector to multiply by ``A``, ``size_n`` if ``trans=='N'``, ``size_m`` if ``trans=='T'``
    :alpha: scale factor on ``A*x``
    :beta: scale factor on ``y``
    :size_m: number of rows of ``A``; size of ``y`` if ``trans == 'N'``, size of ``x`` if ``trans == 'T'``
    :size_n: number of columns of ``A``, size of ``x`` if ``trans == 'N'``, size of ``y`` if ``trans == 'T'``
    :lda: the first dimension of ``A`` as declared by the caller; ``lda >= size_m``
  \output
    :y[size_m OR size_n]: the product ``A * x`` (``size_m`` if ``trans=='N'``, ``size_n`` if ``trans == 'T'``)
\endrst*/
void GeneralMatrixVectorMultiply(double const * restrict A, char trans, double const * restrict x, double alpha, double beta, int size_m, int size_n, int lda, double * restrict y) noexcept OL_NONNULL_POINTERS;

/*!\rst
  Computes the matrix-matrix product ``C = alpha * op(A) * B + beta * C``, where ``op(A) = A`` or ``A^T``, depending on transA
  ``A, B, C`` can be general matrices (no requirements on symmetry, etc.)
  Equivalent to calling GeneralMatrixVectorMultiply with ``A`` against each column of ``B``.
  No inputs may be nullptr.

  \param
    :A[size_m][size_k]: left matrix multiplicand
    :transA: whether to multiply by ``A`` ('N') or ``A^T`` ('T')
    :B[size_k][size_n]: right matrix multiplicand
    :alpha: scale factor on ``A*B``
    :beta: scale factor on ``C``
    :size_m: rows of ``op(A), C``
    :size_k: cols of ``op(A)``, rows of ``B``
    :size_n: cols of ``B, C``
  \output
    :C[size_m][size_n]: the result ``A * B``
\endrst*/
void GeneralMatrixMatrixMultiply(double const * restrict Amat, char transA, double const * restrict Bmat, double alpha, double beta, int size_m, int size_k, int size_n, double * restrict Cmat) noexcept OL_NONNULL_POINTERS;

/*!\rst
  Computes ``L^-1`` when ``L`` is lower triangular.  ``L`` must be nonsingular otherwise
  the output will be nonsense.

  This is NOT backward-stable and should NOT be used!  Substantial superfluous
  numerical error can occur for poorly conditioned matrices.
  Caveat: may have utility if you are very certain of what you are doing in the face of [severe] loss of precision

  \param
    :matrix[size_m][size_m]: triangular matrix, ``L``, to be inverted
    :size_m: dimension of matrix
  OUPUTS:
    :inv_matrix[size_m][size_m]: inverse of ``L``
\endrst*/
void TriangularMatrixInverse(double const * restrict matrix, int size_m, double * restrict inv_matrix) noexcept OL_NONNULL_POINTERS;

/*!\rst
  Computes ``A^-1`` when ``A`` is SPD.  Output is nonsense for non-SPD A matrices.
  A must be previously cholesky-factored (e.g., via ComputeCholeskyFactorL)

  This is NOT backward-stable and should NOT be used!  Substantial superfluous
  numerical error can occur for poorly conditioned matrices.
  Caveat: may have utility if you are very certain of what you are doing in the face of [severe] loss of precision

  \param
    :matrix[size_m][size_m]: matrix, ``A``, to be inverted (only the lower-triangle is read)
        ``A`` must be previously cholesky-factored.
    :size_m: dimension of ``A``
  \output
    :inv_matrix: ``A^-1`` (stored as a full matrix even though it is symmetric)
\endrst*/
void SPDMatrixInverse(double const * restrict matrix, int size_m, double * restrict inv_matrix) noexcept OL_NONNULL_POINTERS;

/*!\rst
  Computes the PLU factorization of a matrix A using LU-decomposition with partial pivoting: ``A = P * L * U``.
  ``P`` is a permutation matrix--an identity matrix with some rows (potentially) swapped.
  ``L`` is lower triangular with 1s on the diagonal.
  ``U`` is upper triangular.
  Since ``L``'s diagonal is unit, ``L`` and ``U`` can be stored together in ``A`` (omitting the 1s).
  Since ``P`` is just a permuted identity, we can instead track the row-swaps done in a vector.

  Fails with error code if the matrix is definitely singular--a pivot element is 0 or subnormal.

  For further details:
  1. http://en.wikipedia.org/wiki/LU_decomposition (not the best info really)
  2. L. Trefethen and D. Bau, Numerical Linear Algebra, Chp 20-23
  3. G. Golub and C. Van Loan, Matrix Computations, Chp 3.
  4. keywords: PLU, LU factorization/decomposition [with partial pivoting]

  \param
    :r: dimension of the matrix
    :A[size][size]: the matrix to be factored (e.g., describing a system of equations)
  \output
    :pivot[size]: array of pivot positions (i.e., row-swaps) used during factorization
    :A[size][size]: a matrix containing the ``L, U`` factors
  \return
    0 if successful.  Otherwise the index (fortran-style, from 1) of the failed pivot element.
\endrst*/
int ComputePLUFactorization(int r, int * restrict pivot, double * restrict A) noexcept OL_NONNULL_POINTERS OL_WARN_UNUSED_RESULT;

/*!\rst
  Solves the system of equations ``A*x = b``, where ``A`` has been previously PLU-factored (by ComputePLUFactorization()).
  So in essence, solves: ``P * L * U * x = b``.
  The input ``b`` is overwritten with the solution ``x``.
  ``A`` is provided in the matrix LU and the array pivot.  ``pivot`` represents the permutation matrix ``P``.
  The triangular matrices ``L`` and ``U`` are stored in ``A``: the upper triangle of ``A`` holds ``U``.
  The strict lower triangle of ``A`` holds ``L`` (since ``L``'s diagonal is unit).

  \param
    :r: size of vector, dimension of matrix
    :LU[size][size]: factored matrix containing ``L, U`` factors the original system, ``A``
    :pivot[size]: pivot position array generated during LU decomposition with partial pivoting
    :b[size]: the right hand side
  \output
    :b[size]: the solution vector
\endrst*/
void PLUMatrixVectorSolve(int r, double const * restrict LU, int const * restrict pivot, double * restrict b) noexcept OL_NONNULL_POINTERS;

}  // end namespace optimal_learning

#endif  // MOE_OPTIMAL_LEARNING_CPP_GPP_LINEAR_ALGEBRA_HPP_
