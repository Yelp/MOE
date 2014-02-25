// gpp_logging.cpp
/*
  Utilities for printing commonly used structures. We put printing code here to hide std::printf() calls.
*/

#include <cstdio>

#include "gpp_common.hpp"
#include "gpp_geometry.hpp"

namespace optimal_learning {

void PrintDomainBounds(ClosedInterval const * restrict domain_bounds, int dim) {
  for (int i = 0; i < dim; ++i) {
    std::printf("dim %d: bounds = [%.18E, %.18E]\n", i, domain_bounds[i].min, domain_bounds[i].max);
  }
}

}  // end namespace optimal_learning
