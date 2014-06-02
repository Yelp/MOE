/*!
  \file gpp_test_utils.hpp
  \rst
  Test for gpp_test_utils.hpp: utilities that are useful in writing other tests, like pinging
  error checking, etc.
\endrst*/

#ifndef MOE_OPTIMAL_LEARNING_CPP_GPP_TEST_UTILS_TEST_HPP_
#define MOE_OPTIMAL_LEARNING_CPP_GPP_TEST_UTILS_TEST_HPP_

#include "gpp_common.hpp"

namespace optimal_learning {

/*!\rst
  Checks the correctness of the functions in gpp_test_utils.hpp.

  \return
    number of test failures: 0 if all domain functions are working properly
\endrst*/
OL_WARN_UNUSED_RESULT int TestUtilsTests();

}  // end namespace optimal_learning

#endif  // MOE_OPTIMAL_LEARNING_CPP_GPP_TEST_UTILS_TEST_HPP_
