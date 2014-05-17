/*!
  \file gpp_linear_algebra-inl.hpp
  \rst
  The "inline header" for gpp_linear_algebra.  This file contains inline function definitions and template definitions,
  particularly for functions/templates that contain more complex logic making them too lengthy/cumbersome for gpp_linear_algebra.hpp.
\endrst*/

#ifndef MOE_OPTIMAL_LEARNING_CPP_GPP_LINEAR_ALGEBRA_INL_HPP_
#define MOE_OPTIMAL_LEARNING_CPP_GPP_LINEAR_ALGEBRA_INL_HPP_

#include "gpp_common.hpp"

namespace optimal_learning {

/*!\rst
  Computes ``A = alpha*v*u^T + A``, where ``v * u^T`` is the outer product of ``v`` with ``u``, with
  ``v \in R^m`` and ``u \in R^n`` (aka ``A_{ij} += v_i * u_j``)
  If ``v == u``, then this update is symmetric and semi-definite (positive or negative depends on sign of ``alpha``).

  Mathematically, the update to ``A`` is rank 1.  Numerically, it will almost never be.

  \param
    :size_m: length of ``v``
    :size_n: length of ``u``
    :alpha: scaling factor
    :vector_v[size_m]: the vector ``v``
    :vector_u[size_n]: the vector ``u``
  \output
    :outerprod[size_m][size_n]: ``A`` such that ``A_{ij} = v_i * v_j``
\endrst*/
inline OL_NONNULL_POINTERS void OuterProduct(int size_m, int size_n, double alpha, double const * restrict vector_v, double const * restrict vector_u, double * restrict outer_prod) noexcept {
  for (int i = 0; i < size_n; ++i) {
    double temp = alpha*vector_u[i];
    for (int j = 0; j < size_m; ++j) {
      outer_prod[j] += vector_v[j]*temp;
    }
    outer_prod += size_m;
  }
}

/*!\rst
  Computes the trace of a matrix, ``tr(A) = \sum_{i = 1}^n A_ii``

  \param
    :A[size_m][size_m]: matrix to take the trace of
    :size_m: dimension of A
  \return
    the trace of ``A``
\endrst*/
inline OL_NONNULL_POINTERS OL_WARN_UNUSED_RESULT double MatrixTrace(double const * restrict A, int size_m) noexcept {
  double trace = 0.0;
  for (int i = 0; i < size_m; ++i) {
    trace += A[0];
    A += size_m + 1;
  }
  return trace;
}

/*!\rst
  Computes ``tr(A*B)`` without explicitly forming ``A*B``

  Naively, computing ``tr(A*B)`` would involve: ``C_{i,k} = A_{i,j}*B_{j,k}``, ``trace = C_{i,i}``
  Spelled out::

    for i = 1:n
      for j = 1:n
        for k = 1:n
         C_{i,k} += A_{i,j} * B_{j,k}

  Then,
  trace = ``\sum_{i = 1}^n C_{i,i}``
  This has cost ``O(n^3)``, dominated by the matrix-matrix product.

  Notice that since we only need the diagonal of ``C`` to compute the trace, there is no need to form ``C_{1,10}``, for example.
  Thus the above loop can be reorganized:
  ``trace = A_{i,j} * B_{j,i}``

  For the pseudocode, we take the original code and set ``k = i``::

    for i = 1:n
      for j = 1:n
          trace += A_{i,j} * B_{j,i}

  Which is ``O(n^2)``.

  Finally, for matrix-products ``A*B``, when ``A`` is stored column-major, we use the view of matrix-vector multiply,
  ``Ax``, as a weighted sum of the columns of ``A``.  This improves memory access patterns.  But here, such an arrangement
  is impossible and we have to access ``A`` in a "bad" ordering.

  ``A, B`` do not need to be square (as long as their product, ``A*B``, is square), but we make this assumption for convenience.

  \param
    :A[size][size]: left matrix
    :B[size][size]: right matrix
    :size: dimension of matrices
  \return
    ``tr(A*B)``
\endrst*/
inline OL_NONNULL_POINTERS OL_WARN_UNUSED_RESULT double TraceOfGeneralMatrixMatrixMultiply(double const * restrict A, double const * restrict B, int size) noexcept {
  double trace = 0.0;

  if (size & 1) {
    for (int j = 0; j < size; ++j) {
      trace += A[0] * B[j];
      A += size;
    }
    A -= size*size;
    A += 1;
    B += size;
  }

  double const * restrict Apos2 = A + 1;
  double const * restrict Bpos2 = B + size;
  for (int i=(size & 1); i < size; i+=2) {
    for (int j = 0; j < size; ++j) {
      trace += A[0] * B[j];
      trace += Apos2[0] * Bpos2[j];
      A += size;
      Apos2 += size;
    }
    A -= size*size;
    A += 2;
    Apos2 -= size*size;
    Apos2 += 2;

    B += 2*size;
    Bpos2 += 2*size;
  }
  return trace;
}

}  // end namespace optimal_learning

#endif  // MOE_OPTIMAL_LEARNING_CPP_GPP_LINEAR_ALGEBRA_INL_HPP_
