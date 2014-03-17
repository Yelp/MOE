// gpp_geometry_test.hpp
/*
  Tests for gpp_geometry.hpp: utilities for performing d-dimensional geometry.
*/

#ifndef OPTIMAL_LEARNING_EPI_SRC_CPP_GPP_GEOMETRY_TEST_HPP_
#define OPTIMAL_LEARNING_EPI_SRC_CPP_GPP_GEOMETRY_TEST_HPP_

#include "gpp_common.hpp"

namespace optimal_learning {

/*
  Tests that the member functions of ClosedInterval work correctly.
  RETURNS:
  number of test failures: 0 if all geometry functions are working properly
*/
OL_WARN_UNUSED_RESULT int ClosedIntervalTests();

/*
  Tests the various geometry utilities in gpp_geometry.hpp; e.g., distance from point to plane,
  projection from point to plane, hypercube/simplex intersection

  RETURNS:
  number of test failures: 0 if all geometry functions are working properly
*/
OL_WARN_UNUSED_RESULT int GeometryToolsTests();

}  // end namespace optimal_learning

#endif  // OPTIMAL_LEARNING_EPI_SRC_CPP_GPP_GEOMETRY_TEST_HPP_
