/*!
  \file gpp_linear_algebra_test.hpp
  \rst
  Functions for testing gpp_linear_algebra and supporting utilities.

  Includes Build* functions that build matrices with various properties.  For example,
  * Well-conditioned SPD matrices
  * Orthogonal matrices
  * Ill-conditioned SPD

  and so forth. These are in turn used as test inputs for the linear algebra routines.

  The linear algebra tests are all called through RunLinearAlgebraTests(); these generally involve tests like checking
  against hand-verified examples, verifying special matrix properties, testing special cases against general cases,
  and verifying analytic mathematical bounds on residual/error.
\endrst*/

#ifndef MOE_OPTIMAL_LEARNING_CPP_GPP_LINEAR_ALGEBRA_TEST_HPP_
#define MOE_OPTIMAL_LEARNING_CPP_GPP_LINEAR_ALGEBRA_TEST_HPP_

#include "gpp_common.hpp"

namespace optimal_learning {

struct UniformRandomGenerator;

/*!\rst
  Utility function to generate a ``m x m`` identity matrix.
  matrix is overwritten.

  \param
    :size_m: dimension of matrix
    :matrix[size_m][size_m]: pointer to space for identity matrix
  \output
    :matrix[size_m][size_m]: identity matrix of order size_m
\endrst*/
OL_NONNULL_POINTERS void BuildIdentityMatrix(int size_m, double * restrict matrix) noexcept;

/*!\rst
  Builds the "Prolate" matrix.  This is a generally ill-conditioned, symmetric matrix.  With the default parameter,
  the condition number is about ``10^(n/2)``; e.g., with n=10, the condition number is ``1.84 x 10^6``.

  This matrix is also Toeplitz (constant along diagonals).

  For ``0 < alpha <= 0.5``, the prolate matrix is SPD.  At ``alpha = 0``, it is singular; at ``alpha = 0.5``, it is the identity.

  This matches the result of:
  ``gallery('prolate', size, alpha)`` in MATLAB.  As in MATLAB, the default parameter is 0.25.
  See implementation for matrix definition.

  \param
    :alpha: prolate parameter
    :size: dimension of prolate matrix
  \output
    :prolate_matrix[size][size]: the prolate matrix with parameter ``alpha``, dimension ``size``
\endrst*/
static constexpr double kProlateDefaultParameter = 0.25;
OL_NONNULL_POINTERS void BuildProlateMatrix(double alpha, int size, double * restrict prolate_matrix) noexcept;

/*!\rst
  Builds the "Moler" matrix.  This is a generally ill-conditioned, SPD matrix.  Roughly speaking, with
  the default parameter, the condition number is about ``10^(n/2)``; e.g., with ``n=10``, the condition number is ``3.68 x 10^6``.

  .. Note:: some parameter choices (near 0) result in near-identity matrices.

  This matrix has generally has small eigenvalue, with the others clustered in a comparatively small
  range away from the smallest.

  This matches the result of:
  ``gallery('moler', size, alpha)`` in MATLAB.  As in MATLAB, the default parameter is -1.0.
  See implementation for matrix definition.

  \param
    :alpha: moler matrix parameter
    :size: dimension of moler matrix
  \output
    :moler_matrix[size][size]: the moler matrix with parameter ``alpha``, dimension ``size``
\endrst*/
static constexpr double kMolerDefaultParameter = -1.0;
OL_NONNULL_POINTERS void BuildMolerMatrix(double alpha, int size, double * restrict moler_matrix) noexcept;

/*!\rst
  Builds an orthogonal (unitary) and symmetric matrix (thus involutary):
  ``Q * Q^T = I`` and ``Q = Q^T``.

  ``Q`` is NOT SPD.

  \param
    :size: dimension of matrix
  \output
    :orthog_symm_matrix[size][size]: size x size matrix satisfying the above properties
\endrst*/
OL_NONNULL_POINTERS void BuildOrthogonalSymmetricMatrix(int size, double * restrict orthog_symm_matrix) noexcept;

/*!\rst
  Builds a symmetric matrix with random entries.  Entries are chosen uniformly at random in the range
  ``[left_bound, right_bound]``.

  Matrix condition number is generally ``~ 10 * size``.  HOWEVER since this is a *random* matrix, extremely
  poor conditioning is possible (even singular matrices are possible, however unlikely).

  \param
    :size: dimension of matrix
    :left_bound: lower bound for matrix entries
    :right_bound: upper bound for matrix_entries
    :uniform_generator[1]: a UniformRandomGenerator object providing the random engine for uniform random numbers
  \output
    :uniform_generator[1]: UniformRandomGenerator object will have its state changed due to random draws
    :symmetric_matrix[size][size]: a random symmetric matrix
\endrst*/
OL_NONNULL_POINTERS void BuildRandomSymmetricMatrix(int size, double left_bound, double right_bound, UniformRandomGenerator * uniform_generator, double * restrict symmetric_matrix)noexcept;

/*!\rst
  Builds a lower triangular matrix with random entries; ``entries \in [0,1]``.  The strict upper triangle is set to 0.

  Matrix condition number is generally ``~ 10 * size``.  HOWEVER since this is a *random* matrix, extremely
  poor conditioning is possible (even singular matrices are possible, however unlikely).

  Building an SPD matrix by:
  ``A = BuildRandomLowerTriangularMatrix``
  ``SPD = A * A^T``
  usually leads to ill-conditioned SPD matrices because ``cond(SPD)`` is ``cond(A)^2``.

  \param
    :size: dimension of matrix
    :uniform_generator[1]: a UniformRandomGenerator object providing the random engine for uniform random numbers
  \output
    :uniform_generator[1]: UniformRandomGenerator object will have its state changed due to random draws
    :lower_triangular_matrix[size][size]: a random, lower triangular matrix
\endrst*/
OL_NONNULL_POINTERS void BuildRandomLowerTriangularMatrix(int size, UniformRandomGenerator * uniform_generator, double * restrict lower_triangular_matrix) noexcept;

/*!\rst
  Builds an SPD matrix with random entries; ``entries \in [0,1]``.

  Matrix condition number is generally ``100 * size^2`` with high variance.  Since this is a random matrix,
  extremely poor conditioning is possible (even singular matrices are possible, however unlikely).

  Builds SPD matrix by computing ``L * L^T``, where ``L`` comes from BuildRandomLowerTriangularMatrix().

  \param
    :size: dimension of matrix
    :uniform_generator[1]: a UniformRandomGenerator object providing the random engine for uniform random numbers
  \output
    :uniform_generator[1]: UniformRandomGenerator object will have its state changed due to random draws
    :spd_matrix[size][size]: a random, SPD matrix
\endrst*/
OL_NONNULL_POINTERS void BuildRandomSPDMatrix(int size, UniformRandomGenerator * uniform_generator, double * restrict spd_matrix) noexcept;

/*!\rst
  Generates matrix ``F`` (as a function of ``x``) such that
  ``F * x = [ ||x||_2; zeros(n-1,1) ]``
  ``F`` is orthogonal (unitary).  ``F`` is symmetric but never SPD.

  See implementation for the definition of ``F``.

  The eigenvalues of ``F`` are ``+/- 1``; 1 has multiplicity ``n-1``, -1 has multiplicity 1 (hence not SPD).

  \param
    :vector[size]: vector x used to construct F so that F * x = ||x||_2 * e_0
    :size: dimension of matrix, vector
  \output
    :householder[size][size]: householder matrix, F
\endrst*/
OL_NONNULL_POINTERS void BuildHouseholderReflectorMatrix(double const * restrict vector, int size, double * restrict householder) noexcept;

/*!\rst
  Builds a vector with random entries.  ``Entries \in [left_bound, right_bound]``.

  A random (non-symmetric) ``m X n`` matrix can be built too:
  ``BuildRandomVector(m*n, matrix);``

  \param
    :size: number of elements in vector
    :uniform_generator[1]: a UniformRandomGenerator object providing the random engine for uniform random numbers
  \output
    :uniform_generator[1]: UniformRandomGenerator object will have its state changed due to random draws
    :vector[size]: array filled with random entries
\endrst*/
OL_NONNULL_POINTERS void BuildRandomVector(int size, double left_bound, double right_bound, UniformRandomGenerator * uniform_generator, double * restrict vector) noexcept;

/*!\rst
  Checks if the input matrix is symmetric to within tolerance.

  If the matrix was explicitly constructed to be symmetric (e.g., ``A_{j,i}`` copied from ``A_{i,j}``), then
  tolerance should be set to 0.0.  Otherwise tolerance should usually be near std::numeric_limits<double>::epsilon().

  \param
    :matrix[size][size]: matrix to check for symmetry
    :size: dimension of matrix
    :tolerance: amount of element-wise non-symmetry allowed
  \return
    true if matrix is symmetric
\endrst*/
OL_NONNULL_POINTERS OL_WARN_UNUSED_RESULT bool CheckMatrixIsSymmetric(double const * restrict matrix, int size, double tolerance) noexcept;

/*!\rst
  Runs a battery of tests on (supporting) linear algebra routines:
  * cholesky factorization
  * Solving ``A * x = b`` when ``A`` is SPD
  * Formation of ``A^-1``
  * ``y = A * x``, general ``A``
  * ``C = A * B``, general ``A, B``
  * ``y = A * x``, ``A = A^T``
  * ``y = A * x``, ``A`` lower triangular
  * computing `A^T``

  \return
    number of test failures: 0 if all is working well.
\endrst*/
OL_WARN_UNUSED_RESULT int RunLinearAlgebraTests();

}  // end namespace optimal_learning

#endif  // MOE_OPTIMAL_LEARNING_CPP_GPP_LINEAR_ALGEBRA_TEST_HPP_
