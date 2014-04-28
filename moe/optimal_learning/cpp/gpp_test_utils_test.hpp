// gpp_test_utils.hpp
/*
  Test for gpp_test_utils.hpp: utilities that are useful in writing other tests, like pinging
  error checking, etc.
*/

#ifndef OPTIMAL_LEARNING_EPI_SRC_CPP_GPP_TEST_UTILS_TEST_HPP_
#define OPTIMAL_LEARNING_EPI_SRC_CPP_GPP_TEST_UTILS_TEST_HPP_

#include "gpp_common.hpp"

namespace optimal_learning {

/*
  Checks the correctness of the functions in gpp_test_utils.hpp.

  RETURNS:
  number of test failures: 0 if all domain functions are working properly
*/
OL_WARN_UNUSED_RESULT int TestUtilsTests();

}  // end namespace optimal_learning

#endif  // OPTIMAL_LEARNING_EPI_SRC_CPP_GPP_TEST_UTILS_TEST_HPP_
