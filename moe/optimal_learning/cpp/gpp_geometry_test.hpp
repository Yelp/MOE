/*!
  \file gpp_geometry_test.hpp
  \rst
  Tests for gpp_geometry.hpp: utilities for performing d-dimensional geometry.
\endrst*/

#ifndef MOE_OPTIMAL_LEARNING_CPP_GPP_GEOMETRY_TEST_HPP_
#define MOE_OPTIMAL_LEARNING_CPP_GPP_GEOMETRY_TEST_HPP_

#include "gpp_common.hpp"

namespace optimal_learning {

/*!\rst
  Tests that the member functions of ClosedInterval work correctly.

  \return
    number of test failures: 0 if all geometry functions are working properly
\endrst*/
OL_WARN_UNUSED_RESULT int ClosedIntervalTests();

/*!\rst
  Tests the various geometry utilities in gpp_geometry.hpp; e.g., distance from point to plane,
  projection from point to plane, hypercube/simplex intersection

  \return
    number of test failures: 0 if all geometry functions are working properly
\endrst*/
OL_WARN_UNUSED_RESULT int GeometryToolsTests();

}  // end namespace optimal_learning

#endif  // MOE_OPTIMAL_LEARNING_CPP_GPP_GEOMETRY_TEST_HPP_
