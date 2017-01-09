/*!
  \file gpp_linear_algebra_test.cpp
  \rst
  Routines to test the functions in gpp_linear_algebra.cpp and gpp_linear_algebra-inl.hpp.

  This includes a battery of tests that verify that all of our linear algebra subroutines are working correctly.  These tests
  fall into a few categories:

  1. manually verifying a general function (e.g., GeneralMatrixVectorMultiply) and then using that to
     verify special cases (e.g., Triangular and Symmetric multiply).
  2. asserting properties of the underlying matrices or operators: e.g., Q*Q^T = I for Q known to be orthogonal, or X*X^-1 = I, etc.
  3. checking correctness against simple, hand-verified cases
  4. checking results against analytically known (usually norm-wise) error bounds, taking conditioning into account over
     well- and ill-conditioned inputs

  These tests live in the functions called through RunLinearAlgebraTests().

  This file also has implementations various Build.*() functions, which provide interesting inputs for the linear algebra
  testing.  These include various random matrices (random, symmetric, SPD), interesting "standard" matrix examples
  (prolate, moler, orthogonal symmetric) as well as some matrices with interesting properties resulting from things like
  Householder Reflections.  We also have some utilites for data manipulation (e.g., extracting the lower triangle) as well as
  routines to manipulate condition number (e.g., adding diagonal dominance).
\endrst*/

#include "gpp_linear_algebra_test.hpp"

#include <cmath>
#include <cstdlib>

#include <algorithm>
#include <limits>
#include <vector>

#include <boost/random/uniform_real.hpp>  // NOLINT(build/include_order)

#include "gpp_common.hpp"
#include "gpp_covariance.hpp"
#include "gpp_exception.hpp"
#include "gpp_linear_algebra.hpp"
#include "gpp_linear_algebra-inl.hpp"
#include "gpp_logging.hpp"
#include "gpp_random.hpp"
#include "gpp_test_utils.hpp"

namespace optimal_learning {

void BuildIdentityMatrix(int size_m, double * restrict matrix) noexcept {
  std::fill(matrix, matrix + size_m*size_m, 0.0);
  for (int i = 0; i < size_m; ++i) {
    matrix[0] = 1.0;
    matrix += size_m+1;  // puts us exactly on the next diagonal entry
  }
}

/*!\rst
  ``A_{i,j} = { 2 * \alpha                                 if i == j``
  ``          { \sin(2 * \pi * \alpha * k)/ (\pi * k)      otherwise``
  where ``k = |i - j|``
\endrst*/
void BuildProlateMatrix(double alpha, int size, double * restrict prolate_matrix) noexcept {
  for (int j = 0; j < size; ++j) {
    for (int i = 0; i < size; ++i) {
      if (i == j) {
        prolate_matrix[j*size + i] = 2.0 * alpha;
      } else {
        int k = std::abs(i-j);
        double angle = 2.0 * kPi * alpha * static_cast<double>(k);
        prolate_matrix[j*size + i] = std::sin(angle)/(kPi * static_cast<double>(k));
      }
    }
  }
}

/*!\rst
  ``A_{ij} = { i * alpha^2 + 1.0               if i == j``
  ``         { min(i, j) * alpha^2 + alpha     otherwise``
\endrst*/
void BuildMolerMatrix(double alpha, int size, double * restrict moler_matrix) noexcept {
  for (int j = 0; j < size; ++j) {
    for (int i = 0; i < size; ++i) {
      if (i == j) {
        moler_matrix[j*size + i] = i * alpha * alpha + 1.0;
      } else {
        moler_matrix[j*size + i] = std::min(i, j) * alpha * alpha + alpha;
      }
    }
  }
}

/*!\rst
  Builds a matrix ``Q`` s.t. ``Q * Q^T = I``, ``Q = Q^T``, and ``\|Q*x\| = \|x\|``.  In particular, ``Q` is (real) orthogonal
  AND symmetric.  This is not the only ``Q`` with the given properties.  ``Q`` is not SPD.

  This is the eigenvector matrix for a n-point second-difference matrix (e.g., discrete hessian).
\endrst*/
void BuildOrthogonalSymmetricMatrix(int size, double * restrict orthog_symm_matrix) noexcept {
  for (int j = 0; j < size; ++j) {
    for (int i = 0; i < size; ++i) {
      // angle = 2.0 * (i+1) * (j+1) * kPi / static_cast<double>(2*size +1);
      // orthog_symm_matrix[j*size + i] = 2.0 * std::sin(angle) / std::sqrt(static_cast<double>(2*size + 1));
      double angle = (i+1) * (j+1) * kPi / static_cast<double>(size+1);
      orthog_symm_matrix[j*size + i] = std::sqrt(2.0/static_cast<double>(size+1)) * std::sin(angle);
    }
  }
}

/*!\rst
  Randomly generates half (diagonal and one triangle) of a matrix and copies those values into the other half.
  The result is not guaranteed to have any special properties (e.g., SPD) beyond symmetry.
\endrst*/
void BuildRandomSymmetricMatrix(int size, double left_bound, double right_bound, UniformRandomGenerator * uniform_generator, double * restrict symmetric_matrix) noexcept {
  boost::uniform_real<double> uniform_double(left_bound, right_bound);
  for (int j = 0; j < size; ++j) {
    for (int i = 0; i < size; ++i) {
      double temp = uniform_double(uniform_generator->engine);
      symmetric_matrix[j*size + i] = temp;
      symmetric_matrix[i*size + j] = temp;
    }
  }
}

/*!\rst
  Randomly generates a lower triangular matrix, zeroing the upper triangle.
\endrst*/
void BuildRandomLowerTriangularMatrix(int size, UniformRandomGenerator * uniform_generator, double * restrict lower_triangular_matrix) noexcept {
  boost::uniform_real<double> uniform_double_unit_interval(0.0, 1.0);

  for (int i = 0; i < size; ++i) {
    for (int j = 0; j < i; ++j) {
      lower_triangular_matrix[i*size + j] = 0.0;
    }
    for (int j = i; j < size; ++j) {
      double temp = uniform_double_unit_interval(uniform_generator->engine);
      lower_triangular_matrix[i*size + j] = temp;
    }
  }
}

/*!\rst
  A matrix ``A`` is SPD if and only if it can be cholesky-factored: ``L * L^T = A``.
  Generate ``L`` randomly and form ``A``.
\endrst*/
void BuildRandomSPDMatrix(int size, UniformRandomGenerator * uniform_generator, double * restrict spd_matrix) noexcept {
  std::vector<double> lower_triangular_matrix(size*size);
  std::vector<double> upper_triangular_matrix(size*size);
  BuildRandomLowerTriangularMatrix(size, uniform_generator, lower_triangular_matrix.data());
  MatrixTranspose(lower_triangular_matrix.data(), size, size, upper_triangular_matrix.data());

  GeneralMatrixMatrixMultiply(lower_triangular_matrix.data(), 'N', upper_triangular_matrix.data(), 1.0, 0.0, size, size, size, spd_matrix);
}

/*!\rst
  The matrix ``F`` (householder) as a function of ``x`` (vector) where:
  ``F = I - 2 * v*v^T``,
  where ``v = w / ||w||_2``,
  and   ``w = sign(x[0]) * \|x\|_2 * e_0 + x``  (``e_0`` is the cartesian unit vector, ``[1; zeros(n-1,1)]``)
\endrst*/
void BuildHouseholderReflectorMatrix(double const * restrict vector, int size, double * restrict householder) noexcept {
  double norm_of_vector = VectorNorm(vector, size);

  std::vector<double> v(vector, vector+size);
  v[0] += std::copysign(norm_of_vector, vector[0]);

  double norm_of_v = VectorNorm(v.data(), size);
  VectorScale(size, 1.0/norm_of_v, v.data());

  BuildIdentityMatrix(size, householder);
  for (int i = 0; i < size; ++i) {
    for (int j = 0; j < size; ++j) {
      householder[i*size + j] -= 2.0*v[i]*v[j];
    }
  }
}

void BuildRandomVector(int size, double left_bound, double right_bound, UniformRandomGenerator * uniform_generator, double * restrict vector) noexcept {
  boost::uniform_real<double> uniform_double(left_bound, right_bound);
  for (int i = 0; i < size; ++i) {
    vector[i] = uniform_double(uniform_generator->engine);
  }
}

bool CheckMatrixIsSymmetric(double const * restrict matrix, int size, double tolerance) noexcept {
  bool symmetric_flag = true;
  for (int j = 0; j < size; ++j) {
    for (int i = 0; i < size; ++i) {
      if (CheckDoubleWithinRelative(matrix[j*size +i], matrix[i*size + j], tolerance) == false) {
        symmetric_flag = false;
        return symmetric_flag;
      }
    }
  }

  return symmetric_flag;
}

namespace {

/*!\rst
  Adds ``scale*eye(size)`` to the result of BuildRandomSPDMatrix.

  Adding positive numbers to the diagonal will significantly improve conditioning (thus turning
  any ill-conditioned example matrix in this file into a well-conditioned one).
\endrst*/
OL_NONNULL_POINTERS void ModifyMatrixDiagonal(int size, double scale, double * restrict spd_matrix) noexcept {
  for (int i = 0; i < size; ++i) {
    spd_matrix[0] += scale;
    spd_matrix += size + 1;  // puts us exactly on the next diagonal entry
  }
}

OL_NONNULL_POINTERS void ExtractLowerTriangularPart(double const * restrict matrix, int size, double * restrict lower_triangular_matrix) noexcept {
  std::fill(lower_triangular_matrix, lower_triangular_matrix + size*size, 0.0);
  for (int i = 0; i < size; ++i) {
    for (int j = i; j < size; ++j) {
      lower_triangular_matrix[i*size + j] = matrix[i*size + j];
    }
  }
}

/*!\rst
  Check Cholesky factorization.
  Uses:

  1. Some simple test cases with whole-number results.
  2. Generate random SPD matrices. Factor them. Check that the factorization is close to the original matrix.

  \return
    number of invalid entries in the factorizations
\endrst*/
OL_WARN_UNUSED_RESULT int TestCholesky() {
  int total_errors = 0;

  // simple hand-spun tests with small integer inputs/outputs so that
  // floating point error is non-existent
  {  // hide scope
    static const int kSize1 = 4;
    static const int kSize2 = 3;
    double matrix_A[kSize1*kSize1] =
        {81.0, 27.0, 0.0, 90.0,
         27.0, 13.0, 8.0, 44.0,
         0.0, 8.0, 52.0, 40.0,
         90.0, 44.0, 40.0, 217.0
        };
    double cholesky_A_exact[kSize1*kSize1] =
        {9.0, 3.0, 0.0, 10.0,
         0.0, 2.0, 4.0, 7.0,
         0.0, 0.0, 6.0, 2.0,
         0.0, 0.0, 0.0, 8.0
        };
    double cholesky_A_computed[kSize1*kSize1];

    double matrix_B[kSize2*kSize2] =
        {25.0, 15.0, -5.0,
         15.0, 18.0, 0.0,
         -5.0, 0.0, 11.0
        };
    double cholesky_B_exact[kSize2*kSize2] =
        {5.0, 3.0, -1.0,
         0.0, 3.0, 1.0,
         0.0, 0.0, 3.0
        };
    double cholesky_B_computed[kSize2*kSize2];

    std::copy(matrix_A, matrix_A + kSize1*kSize1, cholesky_A_computed);
    if (ComputeCholeskyFactorL(kSize1, cholesky_A_computed) != 0) {
      ++total_errors;
    }
    ZeroUpperTriangle(kSize1, cholesky_A_computed);

    std::copy(matrix_B, matrix_B + kSize2*kSize2, cholesky_B_computed);
    if (ComputeCholeskyFactorL(kSize2, cholesky_B_computed) != 0) {
      ++total_errors;
    }
    ZeroUpperTriangle(kSize2, cholesky_B_computed);

    for (int i = 0; i < kSize1*kSize1; ++i) {
      if (!CheckDoubleWithinRelative(cholesky_A_computed[i], cholesky_A_exact[i], 0.0)) {
        ++total_errors;
      }
    }

    for (int i = 0; i < kSize2*kSize2; ++i) {
      if (!CheckDoubleWithinRelative(cholesky_B_computed[i], cholesky_B_exact[i], 0.0)) {
        ++total_errors;
      }
    }
  }

  const int num_tests = 3;
  const int sizes[num_tests] = {5, 11, 20};

  UniformRandomGenerator uniform_generator(34187);
  // in each iteration, we form a random SPD matrix, A.
  // we then Cholesky factor it: L * L^T = A.
  // then we compare the product L * L^T to A and ensure that the deviation is small.
  for (int i = 0; i < num_tests; ++i) {
    std::vector<double> spd_matrix(sizes[i]*sizes[i]);
    std::vector<double> cholesky_factor(sizes[i]*sizes[i]);
    std::vector<double> cholesky_factor_T(sizes[i]*sizes[i]);
    std::vector<double> product_matrix(sizes[i]*sizes[i]);

    // this can be badly conditioned but we don't care
    BuildRandomSPDMatrix(sizes[i], &uniform_generator, spd_matrix.data());

    std::copy(spd_matrix.begin(), spd_matrix.end(), cholesky_factor.begin());
    if (ComputeCholeskyFactorL(sizes[i], cholesky_factor.data()) != 0) {
      ++total_errors;
    }
    ZeroUpperTriangle(sizes[i], cholesky_factor.data());
    MatrixTranspose(cholesky_factor.data(), sizes[i], sizes[i], cholesky_factor_T.data());

    // check L * L^T
    GeneralMatrixMatrixMultiply(cholesky_factor.data(), 'N', cholesky_factor_T.data(), 1.0, 0.0, sizes[i], sizes[i], sizes[i], product_matrix.data());

    // backward stability of cholesky guarantees us the following:
    // L * L^T = A + \delta A, with ||\delta A||/||A|| = O(\epsilon_{machine})
    for (int j = 0; j < sizes[i]*sizes[i]; ++j) {
      if (!CheckDoubleWithinRelative(product_matrix[j], spd_matrix[j], 3*std::numeric_limits<double>::epsilon())) {
        ++total_errors;
      }
    }

    // and again this time doing U^T * U
    GeneralMatrixMatrixMultiply(cholesky_factor_T.data(), 'T', cholesky_factor_T.data(), 1.0, 0.0, sizes[i], sizes[i], sizes[i], product_matrix.data());

    // backward stability of cholesky guarantees us the following:
    // L * L^T = A + \delta A, with ||\delta A||/||A|| = O(\epsilon_{machine})
    for (int j = 0; j < sizes[i]*sizes[i]; ++j) {
      if (!CheckDoubleWithinRelative(product_matrix[j], spd_matrix[j], 3*std::numeric_limits<double>::epsilon())) {
        ++total_errors;
      }
    }
  }

  return total_errors;
}

/*!\rst
  Test that SPDMatrixInverse and CholeskyFactorLMatrixVectorSolve are  working correctly
  against some especially bad named matrices and some random inputs.
  Outline:

  1. Construct matrices, ``A``
  2. Select a solution, ``x``, randomly.
  3. Construct RHS by doing ``A*x``.
  4. Solve ``Ax = b`` using backsolve and direct-inverse; check the size of ``\|b - Ax\|``.

  \return
    number of test cases where the solver error is too large
\endrst*/
OL_WARN_UNUSED_RESULT int TestSPDLinearSolvers() {
  int total_errors = 0;

  // simple/small case where numerical factors are not present.
  // taken from: http://en.wikipedia.org/wiki/Cholesky_decomposition#Example
  {
    constexpr int size = 3;
    std::vector<double> matrix =
        {  4.0,   12.0, -16.0,
          12.0,   37.0, -43.0,
         -16.0,  -43.0,  98.0};

    const std::vector<double> cholesky_factor_L_truth =
        { 2.0, 6.0, -8.0,
          0.0, 1.0, 5.0,
          0.0, 0.0, 3.0};
    std::vector<double> rhs = {-20.0, -43.0, 192.0};
    std::vector<double> solution_truth = {1.0, 2.0, 3.0};

    int local_errors = 0;
    // check factorization is correct; only check lower-triangle
    if (ComputeCholeskyFactorL(size, matrix.data()) != 0) {
      ++total_errors;
    }
    for (int i = 0; i < size; ++i) {
      for (int j = 0; j < size; ++j) {
        if (j >= i) {
          if (!CheckDoubleWithinRelative(matrix[i*size + j], cholesky_factor_L_truth[i*size + j], 0.0)) {
            ++local_errors;
          }
        }
      }
    }

    // check the solve is correct
    CholeskyFactorLMatrixVectorSolve(matrix.data(), size, rhs.data());
    for (int i = 0; i < size; ++i) {
      if (!CheckDoubleWithinRelative(rhs[i], solution_truth[i], 0.0)) {
        ++local_errors;
      }
    }

    total_errors += local_errors;
  }

  const int num_tests = 10;
  const int num_test_sizes = 3;
  const int sizes[num_test_sizes] = {5, 11, 20};

  // following tolerances are based on numerical experiments and/or computations
  // of the matrix condition numbers (in MATLAB)
  const double tolerance_backsolve_max_list[3][num_test_sizes] =
      { {10*std::numeric_limits<double>::epsilon(), 100*std::numeric_limits<double>::epsilon(), 100*std::numeric_limits<double>::epsilon()},       // prolate
        {100*std::numeric_limits<double>::epsilon(), 1.0e4*std::numeric_limits<double>::epsilon(), 1.0e6*std::numeric_limits<double>::epsilon()},  // moler
        {50*std::numeric_limits<double>::epsilon(), 1.0e3*std::numeric_limits<double>::epsilon(), 5.0e4*std::numeric_limits<double>::epsilon()}    // random
      };
  const double tolerance_inverse_max_list[3][num_test_sizes] =
      { {1.0e-13, 1.0e-9, 1.0e-2},  // prolate
        {5.0e-13, 2.0e-8, 1.0e1},   // moler
        {7.0e-10, 7.0e-6, 1.0e2}    // random
      };

  UniformRandomGenerator uniform_generator(34187);
  // In each iteration, we form some an ill-conditioned SPD matrix.  We are using
  // prolate, moler, and random matrices.
  // Then we compute L * L^T = A.
  // We want to test solutions of A * x = b using two methods:
  //   1) Inverse: using L, we form A^-1 explicitly (ILL-CONDITIONED!)
  //   2) Backsolve: we never form A^-1, but instead backsolve L and L^T against b.
  // The resulting residual norms, ||b - A*x||_2 are computed; we check that
  // backsolve is accurate and inverse is affected strongly by conditioning.
  for (int j = 0; j < num_tests; ++j) {
    for (int i = 0; i < num_test_sizes; ++i) {
      std::vector<double> matrix(sizes[i]*sizes[i]);
      std::vector<double> inverse_matrix(sizes[i]*sizes[i]);
      std::vector<double> cholesky_factor(sizes[i]*sizes[i]);
      std::vector<double> rhs(sizes[i]);
      std::vector<double> solution(2*sizes[i]);

      double tolerance_backsolve_max;
      double tolerance_inverse_max;
      switch (j) {
        case 0: {
          BuildProlateMatrix(kProlateDefaultParameter, sizes[i], matrix.data());
          tolerance_backsolve_max = tolerance_backsolve_max_list[0][i];
          tolerance_inverse_max = tolerance_inverse_max_list[0][i];
          break;
        }
        case 1: {
          BuildMolerMatrix(-1.66666666666, sizes[i], matrix.data());
          tolerance_backsolve_max = tolerance_backsolve_max_list[1][i];
          tolerance_inverse_max = tolerance_inverse_max_list[1][i];
          break;
        }
        default: {
          if (j <= 1 || j > num_tests) {
            OL_THROW_EXCEPTION(BoundsException<int>, "Invalid switch option.", j, 2, num_tests);
          } else {
            BuildRandomSPDMatrix(sizes[i], &uniform_generator, matrix.data());
            tolerance_backsolve_max = tolerance_backsolve_max_list[2][i];
            tolerance_inverse_max = tolerance_inverse_max_list[2][i];
            break;
          }
        }
      }
      // cholesky-factor A, form A^-1
      std::copy(matrix.begin(), matrix.end(), cholesky_factor.begin());
      if (ComputeCholeskyFactorL(sizes[i], cholesky_factor.data()) != 0) {
        ++total_errors;
      }
      SPDMatrixInverse(cholesky_factor.data(), sizes[i], inverse_matrix.data());

      // set b = A*random_vector.  This way we know the solution explicitly.
      // this also allows us to ignore ||A|| in our computations when we
      // normalize the RHS.
      BuildRandomVector(sizes[i], 0.0, 1.0, &uniform_generator, solution.data());
      SymmetricMatrixVectorMultiply(matrix.data(), solution.data(), sizes[i], rhs.data());

      // re-scale the RHS so that its norm can be ignored in later computations
      double rhs_norm = VectorNorm(rhs.data(), sizes[i]);
      VectorScale(sizes[i], 1.0/rhs_norm, rhs.data());
      std::copy(rhs.begin(), rhs.end(), solution.begin());

      // Solve L * L^T * x1 = b (backsolve) and compute x2 = A^-1 * b
      CholeskyFactorLMatrixVectorSolve(cholesky_factor.data(), sizes[i], solution.data());
      SymmetricMatrixVectorMultiply(inverse_matrix.data(), rhs.data(), sizes[i], solution.data() + sizes[i]);

      double norm_residual_via_backsolve = ResidualNorm(matrix.data(), solution.data(), rhs.data(), sizes[i]);
      double norm_residual_via_inverse = ResidualNorm(matrix.data(), solution.data() + sizes[i], rhs.data(), sizes[i]);

      if (norm_residual_via_backsolve > tolerance_backsolve_max) {
        ++total_errors;
        OL_ERROR_PRINTF("experiment %d, size[%d] = %d, norm_backsolve = %.18E > %.18E = tol\n", j, i, sizes[i],
               norm_residual_via_backsolve, tolerance_backsolve_max);
      }

      if (norm_residual_via_inverse > tolerance_inverse_max) {
        ++total_errors;
        OL_ERROR_PRINTF("experiment %d, size[%d] = %d, norm_inverse = %.18E > %.18E = tol\n", j, i, sizes[i],
               norm_residual_via_inverse, tolerance_inverse_max);
      }
    }
  }

  return total_errors;
}

/*!\rst
  Test that ``A * x`` and ``A^T * x`` work, where ``A`` is a matrix and ``x`` is a vector.
  Outline:

  1. Check different input size combinations and no/transpose setups on small hand-checked problems.
  2. Exploit special property of Householder matrices: perform ``Ax`` and verify that the output has
     the property (see implementation).

  \return
    number of cases where matrix-vector multiply failed
\endrst*/
OL_WARN_UNUSED_RESULT int TestGeneralMatrixVectorMultiply() noexcept {
  int total_errors = 0;

  // simple, hand-checked problems that are be minimally affected by floating point errors
  {  // hide scope
    static const int kSize_m = 3;  // rows
    static const int kSize_n = 5;  // cols

    double matrix_A[kSize_m*kSize_n] =
        {-7.4, 0.1, 9.1,  // first COLUMN of A (col-major storage)
         1.5, -8.8, -0.3,
         -2.9, 6.4, -9.7,
         -9.1, -6.6, 3.1,
         4.6, 3.0, -1.0
        };
    double matrix_A_T[kSize_n*kSize_m];
    const double test_vector1[kSize_n] = {1.3, -3.8, 4.2, 2.1, 0.2};
    const double test_vector2[kSize_n] = {0.5, -2.0, -0.2, -3.1, 1.9};
    const double result_vector1[kSize_m] = {-45.689999999999998, 47.190000000000005, -21.460000000000001};
    const double result_vector2[kSize_m] = {30.829999999999998, 42.530000000000001, -4.420000000000002};
    double product_vector1[kSize_m];

    const double test_vector3[kSize_m] = {-3.2, 0.4, 1.3};
    const double test_vector4[kSize_m] = {2.8, -4.2, 3.5};
    const double result_vector3[kSize_n] = {35.550000000000004, -8.710000000000001, -0.770000000000000, 30.510000000000002, -14.820000000000000};
    const double result_vector4[kSize_n] = {10.709999999999999, 40.110000000000014, -68.949999999999989, 13.090000000000002, -3.220000000000002};
    double product_vector2[kSize_n];

    MatrixTranspose(matrix_A, kSize_m, kSize_n, matrix_A_T);

    // using A as base
    GeneralMatrixVectorMultiply(matrix_A, 'N', test_vector1, 1.0, 0.0, kSize_m, kSize_n, kSize_m, product_vector1);
    for (int i = 0; i < kSize_m; ++i) {
      if (!CheckDoubleWithinRelative(product_vector1[i], result_vector1[i], std::numeric_limits<double>::epsilon())) {
        ++total_errors;
      }
    }

    GeneralMatrixVectorMultiply(matrix_A, 'N', test_vector2, 1.0, 0.0, kSize_m, kSize_n, kSize_m, product_vector1);
    for (int i = 0; i < kSize_m; ++i) {
      if (!CheckDoubleWithinRelative(product_vector1[i], result_vector2[i], std::numeric_limits<double>::epsilon())) {
        ++total_errors;
      }
    }

    GeneralMatrixVectorMultiply(matrix_A_T, 'N', test_vector3, 1.0, 0.0, kSize_n, kSize_m, kSize_n, product_vector2);
    for (int i = 0; i < kSize_n; ++i) {
      if (!CheckDoubleWithinRelative(product_vector2[i], result_vector3[i], 5*std::numeric_limits<double>::epsilon())) {
        ++total_errors;
      }
    }

    GeneralMatrixVectorMultiply(matrix_A_T, 'N', test_vector4, 1.0, 0.0, kSize_n, kSize_m, kSize_n, product_vector2);
    for (int i = 0; i < kSize_n; ++i) {
      if (!CheckDoubleWithinRelative(product_vector2[i], result_vector4[i], 5*std::numeric_limits<double>::epsilon())) {
        ++total_errors;
      }
    }

    // now with A^T as base
    GeneralMatrixVectorMultiply(matrix_A_T, 'T', test_vector1, 1.0, 0.0, kSize_n, kSize_m, kSize_n, product_vector1);
    for (int i = 0; i < kSize_m; ++i) {
      if (!CheckDoubleWithinRelative(product_vector1[i], result_vector1[i], std::numeric_limits<double>::epsilon())) {
        ++total_errors;
      }
    }

    GeneralMatrixVectorMultiply(matrix_A_T, 'T', test_vector2, 1.0, 0.0, kSize_n, kSize_m, kSize_n, product_vector1);
    for (int i = 0; i < kSize_m; ++i) {
      if (!CheckDoubleWithinRelative(product_vector1[i], result_vector2[i], std::numeric_limits<double>::epsilon())) {
        ++total_errors;
      }
    }

    GeneralMatrixVectorMultiply(matrix_A, 'T', test_vector3, 1.0, 0.0, kSize_m, kSize_n, kSize_m, product_vector2);
    for (int i = 0; i < kSize_n; ++i) {
      if (!CheckDoubleWithinRelative(product_vector2[i], result_vector3[i], 5*std::numeric_limits<double>::epsilon())) {
        ++total_errors;
      }
    }

    GeneralMatrixVectorMultiply(matrix_A, 'T', test_vector4, 1.0, 0.0, kSize_m, kSize_n, kSize_m, product_vector2);
    for (int i = 0; i < kSize_n; ++i) {
      if (!CheckDoubleWithinRelative(product_vector2[i], result_vector4[i], 5*std::numeric_limits<double>::epsilon())) {
        ++total_errors;
      }
    }
  }

  const int num_tests = 3;
  const int sizes[num_tests] = {5, 11, 20};

  UniformRandomGenerator uniform_generator(34187);
  // Here we check the performance of matrix-vector multiply using a very well-conditioned matrix
  // with a very unique property.  F(x), the householder reflector for a vector x, has the following property:
  // F * x = [||x||_2, zeros(n-1,1)]  (for an arbitrary vector, ||F*y|| = ||y|| always)
  // Additionally, F is orthogonal so cond(F) = 1 and our results should be computable very accurately.
  for (int i = 0; i < num_tests; ++i) {
    std::vector<double> house_matrix(sizes[i]*sizes[i]);
    std::vector<double> vector(sizes[i]);
    std::vector<double> reflected_vector(sizes[i]);

    BuildRandomVector(sizes[i], -1.0, 1.0, &uniform_generator, vector.data());
    BuildHouseholderReflectorMatrix(vector.data(), sizes[i], house_matrix.data());
    GeneralMatrixVectorMultiply(house_matrix.data(), 'N', vector.data(), 1.0, 0.0, sizes[i], sizes[i], sizes[i], reflected_vector.data());

    double vector_norm = VectorNorm(vector.data(), sizes[i]);

    if (!CheckDoubleWithinRelative(std::fabs(reflected_vector[0]), vector_norm, 5*std::numeric_limits<double>::epsilon())) {
      ++total_errors;
    }
    for (int j = 1; j < sizes[i]; ++j) {
      if (!CheckDoubleWithin(reflected_vector[j], 0.0, 5*std::numeric_limits<double>::epsilon())) {
        ++total_errors;
      }
    }

    // reflector is symmetric, so test T version too
    GeneralMatrixVectorMultiply(house_matrix.data(), 'T', vector.data(), 1.0, 0.0, sizes[i], sizes[i], sizes[i], reflected_vector.data());

    vector_norm = VectorNorm(vector.data(), sizes[i]);

    if (!CheckDoubleWithinRelative(std::fabs(reflected_vector[0]), vector_norm, 5*std::numeric_limits<double>::epsilon())) {
      ++total_errors;
    }
    for (int j = 1; j < sizes[i]; ++j) {
      if (!CheckDoubleWithin(reflected_vector[j], 0.0, 5*std::numeric_limits<double>::epsilon())) {
        ++total_errors;
      }
    }
  }

  return total_errors;
}

/*!\rst
  Check that ``A * B`` works where ``A, B`` are matrices.
  Outline:

  1. Simple hand-checked test case.
  2. Generate a random orthogonal matrix and verify that ``Q * Q^T = I``.
  3. Generate a random SPD matrix and guarantee good conditioning: verify ``A * A^-1 = I``.

  \return
    number of cases where matrix-matrix multiply failed
\endrst*/
OL_WARN_UNUSED_RESULT int TestGeneralMatrixMatrixMultiply() noexcept {
  int total_errors = 0;

  // simple, hand-checked problems that are be minimally affected by floating point errors
  {  // hide scope
    static const int kSize_m = 3;  // rows of A, C
    static const int kSize_k = 5;  // cols of A, rows of B
    static const int kSize_n = 2;  // cols of B, C

    double matrix_A[kSize_m*kSize_k] =
        {-7.4, 0.1, 9.1,  // first COLUMN of A (col-major storage)
         1.5, -8.8, -0.3,
         -2.9, 6.4, -9.7,
         -9.1, -6.6, 3.1,
         4.6, 3.0, -1.0
        };

    double matrix_B[kSize_k*kSize_n] =
        {-1.3, -8.1, -7.2, -0.4, -5.5,
         7.4, 5.3, -3.1, -2.3, 1.9
        };

    double matrix_AB_exact[kSize_m*kSize_n] =
        {-3.309999999999995, 11.210000000000001, 64.700000000000003,
         -8.150000000000006, -44.860000000000014, 86.789999999999992
        };
    double matrix_AB_computed[kSize_m*kSize_n];

    GeneralMatrixMatrixMultiply(matrix_A, 'N', matrix_B, 1.0, 0.0, kSize_m, kSize_k, kSize_n, matrix_AB_computed);

    for (int i = 0; i < kSize_m*kSize_n; ++i) {
      if (!CheckDoubleWithinRelative(matrix_AB_computed[i], matrix_AB_exact[i], 3.0 * std::numeric_limits<double>::epsilon())) {
        ++total_errors;
      }
    }
  }

  const int num_tests = 3;
  const int sizes[num_tests] = {3, 11, 20};

  UniformRandomGenerator uniform_generator(34187);
  // in each iteration, we perform two tests on matrix-matrix multiply.
  //   1) form a matrix Q such that Q * Q^T = I (orthogonal matrix); nontrivial in that Q != I
  //      Compute Q * Q^T and check the result.
  //   2) build a random, SPD matrix, A. (ill-conditioned).
  //      Improve A's conditioning: A = A + size*I (condition number near 1 now)
  //      Form A^-1 (this is OK because A is well-conditioned)
  //      Check A * A^-1 is near I.
  for (int i = 0; i < num_tests; ++i) {
    std::vector<double> orthog_symm_matrix(sizes[i]*sizes[i]);
    std::vector<double> orthog_symm_matrix_T(sizes[i]*sizes[i]);
    std::vector<double> product_matrix(sizes[i]*sizes[i]);
    std::vector<double> spd_matrix(sizes[i]*sizes[i]);
    std::vector<double> cholesky_factor(sizes[i]*sizes[i]);
    std::vector<double> inverse_spd_matrix(sizes[i]*sizes[i]);
    std::vector<double> identity_matrix(sizes[i]*sizes[i]);

    BuildIdentityMatrix(sizes[i], identity_matrix.data());

    BuildOrthogonalSymmetricMatrix(sizes[i], orthog_symm_matrix.data());
    // not technically necessary since this orthog matrix is also symmetric
    MatrixTranspose(orthog_symm_matrix.data(), sizes[i], sizes[i], orthog_symm_matrix_T.data());

    // Q * Q^T = I if Q is orthogonal
    GeneralMatrixMatrixMultiply(orthog_symm_matrix.data(), 'N', orthog_symm_matrix_T.data(), 1.0, 0.0, sizes[i], sizes[i], sizes[i], product_matrix.data());
    VectorAXPY(sizes[i]*sizes[i], -1.0, identity_matrix.data(), product_matrix.data());
    for (int j = 0; j < sizes[i]*sizes[i]; ++j) {
      // do not use relative comparison b/c we're testing against 0
      if (!CheckDoubleWithin(product_matrix[j], 0.0, 20*std::numeric_limits<double>::epsilon())) {
        ++total_errors;
      }
    }

    // and again testing the T version
    GeneralMatrixMatrixMultiply(orthog_symm_matrix.data(), 'T', orthog_symm_matrix.data(), 1.0, 0.0, sizes[i], sizes[i], sizes[i], product_matrix.data());
    VectorAXPY(sizes[i]*sizes[i], -1.0, identity_matrix.data(), product_matrix.data());
    for (int j = 0; j < sizes[i]*sizes[i]; ++j) {
      // do not use relative comparison b/c we're testing against 0
      if (!CheckDoubleWithin(product_matrix[j], 0.0, 20*std::numeric_limits<double>::epsilon())) {
        ++total_errors;
      }
    }

    BuildRandomSPDMatrix(sizes[i], &uniform_generator, spd_matrix.data());
    // ensure spd matrix is well-conditioned (or we can't form A^-1 stably)
    ModifyMatrixDiagonal(sizes[i], static_cast<double>(sizes[i]), spd_matrix.data());

    std::copy(spd_matrix.begin(), spd_matrix.end(), cholesky_factor.begin());
    if (ComputeCholeskyFactorL(sizes[i], cholesky_factor.data()) != 0) {
      ++total_errors;
    }
    SPDMatrixInverse(cholesky_factor.data(), sizes[i], inverse_spd_matrix.data());
    GeneralMatrixMatrixMultiply(spd_matrix.data(), 'N', inverse_spd_matrix.data(), 1.0, 0.0, sizes[i], sizes[i], sizes[i], product_matrix.data());
    VectorAXPY(sizes[i]*sizes[i], -1.0, identity_matrix.data(), product_matrix.data());
    for (int j = 0; j < sizes[i]*sizes[i]; ++j) {
      // do not use relative comparison b/c we're testing against 0
      if (!CheckDoubleWithin(product_matrix[j], 0.0, 10*std::numeric_limits<double>::epsilon())) {
        ++total_errors;
      }
    }
  }

  return total_errors;
}

/*!\rst
  Check that ``Ax`` works for the special cases of ``A`` being:

  1. triangular
  2. symmetric

  Assuming that dgemv() is correct, we then check (using random matrices):

  1. Assert that dtrmv (triangular) matches dgemv for the appropriate matrices.
  2. Assert that dsymv (symmetric) match dgemv for the appropriate matrices.

  \return
    number of cases where matrix-vector multiply does not match triangular or symmetric specialized multiplies.
\endrst*/
OL_WARN_UNUSED_RESULT int TestSpecialMatrixVectorMultiply() {
  int total_errors_dsymv = 0;
  int total_errors_dtrmv = 0;
  int total_errors_dtrmv_T = 0;
  const int num_tests = 3;
  const int sizes[num_tests] = {3, 11, 20};

  UniformRandomGenerator uniform_generator(34187);
  for (int i = 0; i < num_tests; ++i) {
    std::vector<double> matrix(sizes[i]*sizes[i]);
    std::vector<double> lower_triangular_part_matrix(sizes[i]*sizes[i]);
    std::vector<double> upper_triangular_part_matrix(sizes[i]*sizes[i]);
    std::vector<double> multiplicand_vector(sizes[i]);
    std::vector<double> product_by_dgemv(sizes[i]);
    std::vector<double> product_by_dsymv(sizes[i]);
    std::vector<double> product_by_dtrmv(sizes[i]);

    BuildRandomSymmetricMatrix(sizes[i], -1.0, 1.0, &uniform_generator, matrix.data());
    ExtractLowerTriangularPart(matrix.data(), sizes[i], lower_triangular_part_matrix.data());
    BuildRandomVector(sizes[i], -2.0, 2.0, &uniform_generator, multiplicand_vector.data());

    // testing dsymv
    // generate truth value
    GeneralMatrixVectorMultiply(matrix.data(), 'N', multiplicand_vector.data(), 1.0, 0.0, sizes[i], sizes[i], sizes[i], product_by_dgemv.data());

    // generate test value
    // note: dsymv does not access the upper triangular part, so we will check this
    // by giving it a matrix where that region is all 0s
    SymmetricMatrixVectorMultiply(lower_triangular_part_matrix.data(), multiplicand_vector.data(), sizes[i], product_by_dsymv.data());

    for (int j = 0; j < sizes[i]; ++j) {
      if (!CheckDoubleWithinRelative(product_by_dsymv[j], product_by_dgemv[j], 1000*std::numeric_limits<double>::epsilon())) {
        ++total_errors_dsymv;
      }
    }

    // testing dtrmv, no transpose
    // generate truth value
    GeneralMatrixVectorMultiply(lower_triangular_part_matrix.data(), 'N', multiplicand_vector.data(), 1.0, 0.0, sizes[i], sizes[i], sizes[i], product_by_dgemv.data());

    // generate test value
    // note: dsymv does not access the upper triangular part, so we will check this
    // by giving it a matrix where that region is populated
    std::copy(multiplicand_vector.begin(), multiplicand_vector.end(), product_by_dtrmv.begin());
    TriangularMatrixVectorMultiply(matrix.data(), 'N', sizes[i], product_by_dtrmv.data());

    for (int j = 0; j < sizes[i]; ++j) {
      if (!CheckDoubleWithinRelative(product_by_dtrmv[j], product_by_dgemv[j], 1000*std::numeric_limits<double>::epsilon())) {
        ++total_errors_dtrmv;
      }
    }

    // testing dtrmv, transpose
    // generate truth value
    GeneralMatrixVectorMultiply(lower_triangular_part_matrix.data(), 'T', multiplicand_vector.data(), 1.0, 0.0, sizes[i], sizes[i], sizes[i], product_by_dgemv.data());

    // generate test value
    // note: dsymv does not access the upper triangular part, so we will check this
    // by giving it a matrix where that region is populated
    std::copy(multiplicand_vector.begin(), multiplicand_vector.end(), product_by_dtrmv.begin());
    TriangularMatrixVectorMultiply(matrix.data(), 'T', sizes[i], product_by_dtrmv.data());

    for (int j = 0; j < sizes[i]; ++j) {
      if (!CheckDoubleWithinRelative(product_by_dtrmv[j], product_by_dgemv[j], 1000*std::numeric_limits<double>::epsilon())) {
        ++total_errors_dtrmv_T;
      }
    }
  }

  if (total_errors_dsymv != 0) {
    OL_ERROR_PRINTF("dsymv failed\n");
  }
  if (total_errors_dtrmv != 0) {
    OL_ERROR_PRINTF("dtrmv failed\n");
  }
  if (total_errors_dtrmv_T != 0) {
    OL_ERROR_PRINTF("dtrmv_T failed\n");
  }

  return total_errors_dsymv + total_errors_dtrmv + total_errors_dtrmv_T;
}

/*!\rst
  Check that matrix-transpose works.

  \return
    number of entries where ``A`` and ``A^T`` do not match
\endrst*/
OL_WARN_UNUSED_RESULT int TestMatrixTranspose() noexcept {
  const int size_m = 3;
  const int size_n = 5;
  int total_errors = 0;

  std::vector<double> matrix(size_m*size_n);
  std::vector<double> matrix_T(size_m*size_n);
  std::vector<double> product_matrix(size_n*size_n);
  UniformRandomGenerator uniform_generator(34187);

  BuildRandomVector(size_m*size_n, -1.0, 1.0, &uniform_generator, matrix.data());
  MatrixTranspose(matrix.data(), size_m, size_n, matrix_T.data());

  for (int j = 0; j < size_n; ++j) {
    for (int i = 0; i < size_m; ++i) {
      if (CheckDoubleWithin(matrix[j*size_m + i], matrix_T[i*size_n + j], 0.0) == false) {
        ++total_errors;
        break;
      }
    }
  }

  return total_errors;
}

/*!\rst
  Check that ``A = PLU`` factorization works.

  Test is conducted using a case from Trefethen's "Numerical Linear Algebra", 1997.

  \return
    number of cases where PLU fails.
\endrst*/
OL_WARN_UNUSED_RESULT int TestPLUFactor() noexcept {
  const int size = 4;
  int total_errors;
  int pivot[size] = {0};
  const int truth_pivot[size] = {1, 2, 3, 4};
  double matrix[Square(size)] = {2.0, 1.0, 1.0, 0.0, 4.0, 3.0, 3.0, 1.0, 8.0, 7.0, 9.0, 5.0, 6.0, 7.0, 9.0, 8.0};
  const double truth_matrix[Square(size)] = {2.0, 0.5, 0.5, 0.0, 4.0, 1.0, 1.0, 1.0, 8.0, 3.0, 2.0, 1.0, 6.0, 4.0, 2.0, 2.0};

  total_errors = ComputePLUFactorization(size, pivot, matrix);
  for (int i = 0; i < size; ++i) {
    if (!CheckIntEquals(pivot[i], truth_pivot[i])) {
      ++total_errors;
    }
  }

  for (int i = 0; i < Square(size); ++i) {
    if (!CheckDoubleWithinRelative(matrix[i], truth_matrix[i], std::numeric_limits<double>::epsilon())) {
      ++total_errors;
    }
  }
  return total_errors;
}

/*!\rst
  Check that solving ``Ax = b`` works when using PLU factorization + backsolves.

  Test is conducted using a case from Trefethen's "Numerical Linear Algebra", 1997.

  \return
    number of cases where PLU fails.
\endrst*/
OL_WARN_UNUSED_RESULT int TestPLUSolve() noexcept {
  const int size = 4;
  int total_errors;
  int pivot[size] = {0};
  double matrix[Square(size)] = {2.0, 1.0, 1.0, 0.0, 4.0, 3.0, 3.0, 1.0, 8.0, 7.0, 9.0, 5.0, 6.0, 7.0, 9.0, 8.0};
  double rhs1[size] = {58.0, 56.0, 70.0, 49.0};
  const double truth_rhs1[size] = {1.0, 2.0, 3.0, 4.0};
  double rhs2[size] = {64.0, 71.0, 91.0, 73.0};
  const double truth_rhs2[size] = {-5.0, 2.0, 3.0, 7.0};
  double rhs3[size] = {-20.0, -16.0, -20.0, -6.0};
  const double truth_rhs3[size] = {4.0, -2.0, -4.0, 2.0};

  total_errors = ComputePLUFactorization(size, pivot, matrix);
  PLUMatrixVectorSolve(size, matrix, pivot, rhs1);
  PLUMatrixVectorSolve(size, matrix, pivot, rhs2);
  PLUMatrixVectorSolve(size, matrix, pivot, rhs3);

  for (int i = 0; i < size; ++i) {
    if (!CheckDoubleWithinRelative(rhs1[i], truth_rhs1[i], std::numeric_limits<double>::epsilon())) {
      ++total_errors;
    }
    if (!CheckDoubleWithinRelative(rhs2[i], truth_rhs2[i], std::numeric_limits<double>::epsilon())) {
      ++total_errors;
    }
    if (!CheckDoubleWithinRelative(rhs3[i], truth_rhs3[i], std::numeric_limits<double>::epsilon())) {
      ++total_errors;
    }
  }

  return total_errors;
}

/*!\rst
  Test vector norm.

  For several problem sizes, check:

  1. zeros have ``norm = 0``
  2. ``\|\alpha\| = \alpha`` where \alpha is scalar
  3. Scaling a vector by its own norm results in a vector with ``norm = 1.0``.
  4. Columns of a matrix whose columns are *known* to have unit-norm.

  \return
    number of cases where the norm is wrong
\endrst*/
OL_WARN_UNUSED_RESULT int TestNorm() noexcept {
  int total_errors = 0;
  const int num_sizes = 3;
  const int sizes[num_sizes] = {11, 100, 1007};
  UniformRandomGenerator uniform_generator(34187);

  for (int i = 0; i < num_sizes; ++i) {
    std::vector<double> vector(sizes[i], 0.0);
    // zero vector has zero norm
    double norm = VectorNorm(vector.data(), vector.size());
    if (!CheckDoubleWithin(norm, 0.0, 0.0)) {
      ++total_errors;
    }

    BuildRandomVector(sizes[i], -1.0, 1.0, &uniform_generator, vector.data());

    // norm of nothing element is 0
    norm = VectorNorm(vector.data(), 0);
    if (!CheckDoubleWithin(norm, 0.0, 0.0)) {
      ++total_errors;
    }

    // norm of 1 element is |that element|
    norm = VectorNorm(vector.data(), 1);
    if (!CheckDoubleWithinRelative(norm, std::fabs(vector[0]), std::numeric_limits<double>::epsilon())) {
      ++total_errors;
    }

    // unit vectors have unit norm
    norm = VectorNorm(vector.data(), vector.size());
    std::vector<double> unit_vector(vector);
    VectorScale(unit_vector.size(), 1.0/norm, unit_vector.data());
    double unit_norm = VectorNorm(unit_vector.data(), unit_vector.size());
    if (!CheckDoubleWithinRelative(unit_norm, 1.0, std::numeric_limits<double>::epsilon())) {
      ++total_errors;
    }

    if (i < num_sizes - 1) {  // don't do the largest case
      std::vector<double> orthogonal_matrix(Square(sizes[i]));
      BuildOrthogonalSymmetricMatrix(sizes[i], orthogonal_matrix.data());
      for (int j = 0; j < sizes[i]; ++j) {
        double column_norm = VectorNorm(orthogonal_matrix.data() + j*sizes[i], sizes[i]);
        if (!CheckDoubleWithinRelative(column_norm, 1.0, std::sqrt(static_cast<double>(sizes[i]))*std::numeric_limits<double>::epsilon())) {
          ++total_errors;
        }
      }
    }
  }

  return total_errors;
}

/*!\rst
  Test several vector (BLAS-1) functions:

  1. scale: ``y = \alpha * y``
  2. AXPY: ``y = \alpha * x + y``
  3. dot product: ``c = x^T * y``, ``c`` is scalar
  4. norm: ``\|x\|``
\endrst*/
OL_WARN_UNUSED_RESULT int TestVectorFunctions() noexcept {
  int total_errors = 0;
  const int size = 4;
  const double orig_input[size] = {1.5, 2.0, 0.0, -3.2};
  double input[size] = {0};
  const double positive_scale = 2.0;
  const double negative_scale = -3.0;
  const double zero_scale = 0.0;

  // test VectorScale
  {
    const double result_positive_scale[size] = {3.0, 4.0, 0.0, -6.4};
    const double result_negative_scale[size] = {-4.5, -6.0, 0.0, 9.6};
    const double result_zero_scale[size] = {0.0, 0.0, 0.0, 0.0};

    std::copy(orig_input, orig_input + size, input);
    VectorScale(size, positive_scale, input);
    for (int i = 0; i < size; ++i) {
      if (!CheckDoubleWithinRelative(input[i], result_positive_scale[i], std::numeric_limits<double>::epsilon())) {
        ++total_errors;
      }
    }

    std::copy(orig_input, orig_input + size, input);
    VectorScale(size, negative_scale, input);
    for (int i = 0; i < size; ++i) {
      if (!CheckDoubleWithinRelative(input[i], result_negative_scale[i], std::numeric_limits<double>::epsilon())) {
        ++total_errors;
      }
    }

    std::copy(orig_input, orig_input + size, input);
    VectorScale(size, zero_scale, input);
    for (int i = 0; i < size; ++i) {
      if (!CheckDoubleWithinRelative(input[i], result_zero_scale[i], std::numeric_limits<double>::epsilon())) {
        ++total_errors;
      }
    }
  }

  // test VectorAXPY
  {
    const double vector_x[size] = {0.0, 0.5, -1.3, 3.0};
    const double result_positive_scale[size] = {1.5, 3.0, -2.6, 2.8};
    const double result_negative_scale[size] = {1.5, 0.5, 3.9, -12.2};
    const double result_zero_scale[size] = {1.5, 2.0, 0.0, -3.2};

    std::copy(orig_input, orig_input + size, input);
    VectorAXPY(size, positive_scale, vector_x, input);
    for (int i = 0; i < size; ++i) {
      if (!CheckDoubleWithinRelative(input[i], result_positive_scale[i], std::numeric_limits<double>::epsilon())) {
        ++total_errors;
      }
    }

    std::copy(orig_input, orig_input + size, input);
    VectorAXPY(size, negative_scale, vector_x, input);
    for (int i = 0; i < size; ++i) {
      if (!CheckDoubleWithinRelative(input[i], result_negative_scale[i], std::numeric_limits<double>::epsilon())) {
        ++total_errors;
      }
    }

    std::copy(orig_input, orig_input + size, input);
    VectorAXPY(size, zero_scale, vector_x, input);
    for (int i = 0; i < size; ++i) {
      if (!CheckDoubleWithinRelative(input[i], result_zero_scale[i], std::numeric_limits<double>::epsilon())) {
        ++total_errors;
      }
    }
  }

  // test dot product
  {
    const double input2[size] = {0.0};
    const double input3[size] = {0.0, 0.5, -1.3, 3.0};
    double output;
    const double result2 = 0.0;
    const double result3 = 2.0*0.5 - 3.2*3.0;
    output = DotProduct(orig_input, input2, size);
    if (!CheckDoubleWithinRelative(output, result2, std::numeric_limits<double>::epsilon())) {
      ++total_errors;
    }
    output = DotProduct(orig_input, input3, size);
    if (!CheckDoubleWithinRelative(output, result3, std::numeric_limits<double>::epsilon())) {
      ++total_errors;
    }
    // check symmetry
    output = DotProduct(input3, orig_input, size);
    if (!CheckDoubleWithinRelative(output, result3, std::numeric_limits<double>::epsilon())) {
      ++total_errors;
    }
  }

  // test norm
  {
    const double truth_norm_input = 4.060788100849391;
    double norm = VectorNorm(orig_input, size);
    if (!CheckDoubleWithinRelative(norm, truth_norm_input, std::numeric_limits<double>::epsilon())) {
      ++total_errors;
    }

    std::copy(orig_input, orig_input + size, input);
    VectorScale(size, positive_scale, input);
    norm = VectorNorm(input, size);
    if (!CheckDoubleWithinRelative(norm, positive_scale*truth_norm_input, std::numeric_limits<double>::epsilon())) {
      ++total_errors;
    }

    std::copy(orig_input, orig_input + size, input);
    VectorScale(size, negative_scale, input);
    norm = VectorNorm(input, size);
    if (!CheckDoubleWithinRelative(norm, -negative_scale*truth_norm_input, std::numeric_limits<double>::epsilon())) {
      ++total_errors;
    }

    std::copy(orig_input, orig_input + size, input);
    VectorScale(size, zero_scale, input);
    norm = VectorNorm(input, size);
    if (!CheckDoubleWithin(norm, 0.0, 0.0)) {
      ++total_errors;
    }

    total_errors += TestNorm();
  }
  return total_errors;
}

/*!\rst
  Test that outerproduct is working with some small hand-checked cases.

  \return
    number of entries where the outerproduct is invalid
\endrst*/
OL_WARN_UNUSED_RESULT int TestOuterProduct() noexcept {
  int total_errors = 0;
  const int size_m = 2;
  const int size_n = 3;
  const double vector_v[size_m] = {1.2, -2.1};
  const double vector_u[size_n] = {-2.7, 0.0, 3.3};
  double positive_scale = 1.0;
  double zero_scale = 0.0;
  const double outer_prod[size_m*size_n] = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0};
  double input[size_m*size_n] = {0.0};
  const double result_positive_scale[size_m*size_n] = {0.0 + (-2.7*1.2), 1.0 + (-2.7*-2.1), 2.0, 3.0, 4.0 + (3.3*1.2), 5.0 + (3.3*-2.1)};

  std::copy(outer_prod, outer_prod + size_m*size_n, input);
  OuterProduct(size_m, size_n, positive_scale, vector_v, vector_u, input);
  for (int i = 0; i < size_m*size_n; ++i) {
    if (!CheckDoubleWithinRelative(input[i], result_positive_scale[i], std::numeric_limits<double>::epsilon())) {
      ++total_errors;
    }
  }

  std::copy(outer_prod, outer_prod + size_m*size_n, input);
  OuterProduct(size_m, size_n, zero_scale, vector_v, vector_u, input);
  for (int i = 0; i < size_m*size_n; ++i) {
    if (!CheckDoubleWithinRelative(input[i], outer_prod[i], std::numeric_limits<double>::epsilon())) {
      ++total_errors;
    }
  }

  return total_errors;
}

/*!\rst
  Test matrix trace and ``tr(AB)``.

  \return
    number of cases where trace functions fail
\endrst*/
OL_WARN_UNUSED_RESULT int TestMatrixTrace() noexcept {
  int total_errors = 0;

  UniformRandomGenerator uniform_generator(34187);

  // test MatrixTrace
  {
    const int size = 4;
    double matrix[Square(size)];
    BuildRandomVector(Square(size), -1.0, 1.0, &uniform_generator, matrix);
    // replace diagonal with known values
    matrix[0*4 + 0] = 1.5;
    matrix[1*4 + 1] = -2.3;
    matrix[2*4 + 2] = 0.0;
    matrix[3*4 + 3] = 3.1;
    const double result = 1.5 - 2.3 + 0.0 + 3.1;
    double output = MatrixTrace(matrix, size);
    if (!CheckDoubleWithinRelative(output, result, std::numeric_limits<double>::epsilon())) {
      ++total_errors;
    }
  }

  // test TraceOfGeneralMatrixMatrixMultiply
  // test by forming random matrices, evaluating the matrix product, and comparing traces of
  // the explicit and shortcut solutions
  {
    double trace_by_explicit_product;
    double trace_by_shortcut;
    // loop over several sizes to make sure we hit all loop unroll paths
    for (int i = 10; i < 15; ++i) {
      std::vector<double> matrix_A(i*i);
      std::vector<double> matrix_B(i*i);
      std::vector<double> matrix_C(i*i);

      BuildRandomVector(Square(i), -1.0, 1.0, &uniform_generator, matrix_A.data());
      BuildRandomVector(Square(i), -1.0, 1.0, &uniform_generator, matrix_B.data());
      GeneralMatrixMatrixMultiply(matrix_A.data(), 'N', matrix_B.data(), 1.0, 0.0, i, i, i, matrix_C.data());
      trace_by_explicit_product = MatrixTrace(matrix_C.data(), i);
      trace_by_shortcut = TraceOfGeneralMatrixMatrixMultiply(matrix_A.data(), matrix_B.data(), i);

      if (!CheckDoubleWithinRelative(trace_by_shortcut, trace_by_explicit_product, 2.0e-14)) {
        ++total_errors;
      }
    }
  }
  return total_errors;
}

}  // end unnamed namespace

int RunLinearAlgebraTests() {
  int current_errors = 0;
  int total_errors = 0;

  OL_VERBOSE_PRINTF("\nLinear Algebra Unit Tests\n\n");

  current_errors = TestVectorFunctions();
  total_errors += current_errors;
  if (current_errors != 0) {
    OL_PARTIAL_FAILURE_PRINTF("vector function errors = %d\n", current_errors);
  }

  current_errors = TestOuterProduct();
  total_errors += current_errors;
  if (current_errors != 0) {
    OL_PARTIAL_FAILURE_PRINTF("outer product errors = %d\n", current_errors);
  }

  current_errors = TestMatrixTrace();
  total_errors += current_errors;
  if (current_errors != 0) {
    OL_PARTIAL_FAILURE_PRINTF("matrix trace errors = %d\n", current_errors);
  }

  current_errors = TestCholesky();
  total_errors += current_errors;
  if (current_errors != 0) {
    OL_PARTIAL_FAILURE_PRINTF("cholesky errors = %d\n", current_errors);
  }

  current_errors = TestSPDLinearSolvers();
  total_errors += current_errors;
  if (current_errors != 0) {
    OL_PARTIAL_FAILURE_PRINTF("linear solve errors = %d\n", current_errors);
  }

  current_errors = TestGeneralMatrixVectorMultiply();
  total_errors += current_errors;
  if (current_errors != 0) {
    OL_PARTIAL_FAILURE_PRINTF("dgemv errors = %d\n", current_errors);
  }

  current_errors = TestGeneralMatrixMatrixMultiply();
  total_errors += current_errors;
  if (current_errors != 0) {
    OL_PARTIAL_FAILURE_PRINTF("dgemm errors = %d\n", current_errors);
  }

  current_errors = TestSpecialMatrixVectorMultiply();
  total_errors += current_errors;
  if (current_errors != 0) {
    OL_PARTIAL_FAILURE_PRINTF("dsymv, dtrmv errors = %d\n", current_errors);
  }

  current_errors = TestMatrixTranspose();
  total_errors += current_errors;
  if (current_errors != 0) {
    OL_PARTIAL_FAILURE_PRINTF("MatrixTranspose errors = %d\n", current_errors);
  }

  current_errors = TestPLUFactor();
  total_errors += current_errors;
  if (current_errors != 0) {
    OL_PARTIAL_FAILURE_PRINTF("PLU Factorization errors = %d\n", current_errors);
  }

  current_errors = TestPLUSolve();
  total_errors += current_errors;
  if (current_errors != 0) {
    OL_PARTIAL_FAILURE_PRINTF("PLU Solve errors = %d\n", current_errors);
  }

  return total_errors;
}

}  // end namespace optimal_learning
