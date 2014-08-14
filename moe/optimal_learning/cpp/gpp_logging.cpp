/*!
  \file gpp_logging.cpp
  \rst
  Utilities for printing commonly used structures. We put printing code here to hide ``std::printf()`` calls.
\endrst*/

#include "gpp_logging.hpp"

#include <cstdio>

#include "gpp_common.hpp"
#include "gpp_geometry.hpp"

namespace optimal_learning {

/*!\rst
  Since matrices are stored column-major and the natural screen-printed formatting
  is by rows, we need to access the matrix in transposed order.
\endrst*/
void PrintMatrix(double const * restrict matrix, int num_rows, int num_cols) noexcept {
  for (int i = 0; i < num_rows; ++i) {
    for (int j = 0; j < num_cols; ++j) {
      std::printf("%.18E ", matrix[j*num_rows + i]);
    }
    std::printf("\n");
  }
}

/*!\rst
  Opposite PrintMatrix(), the screen formatted ordering here is the same as the
  matrix storage ordering.
\endrst*/
void PrintMatrixTrans(double const * restrict matrix, int num_rows, int num_cols) noexcept {
  // prints a matrix to stdout
  for (int i = 0; i < num_rows; ++i) {
    for (int j = 0; j < num_cols; ++j) {
      std::printf("%.18E ", matrix[i*num_cols + j]);
    }
    std::printf("\n");
  }
}

void PrintDomainBounds(ClosedInterval const * restrict domain_bounds, int dim) {
  for (int i = 0; i < dim; ++i) {
    std::printf("dim %d: bounds = [%.18E, %.18E]\n", i, domain_bounds[i].min, domain_bounds[i].max);
  }
}

}  // end namespace optimal_learning
