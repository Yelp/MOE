// gpp_cov_test.hpp
/*
  Simple function call to run unit tests for covariance functions defined in gpp_covariance.cpp.
  This function consists of a battery of ping tests that verify the correctness of derivatives and hessians
  of the various covariance functions available in gpp_covariance.hpp.
*/

#ifndef OPTIMAL_LEARNING_EPI_SRC_CPP_GPP_COVARIANCE_TEST_HPP_
#define OPTIMAL_LEARNING_EPI_SRC_CPP_GPP_COVARIANCE_TEST_HPP_

#include "gpp_common.hpp"

namespace optimal_learning {

/*
  Ping tests the covariance functions implemented in gpp_covariance.cpp
  Currently the only covariance option is SquareExponential.

  See gpp_test_utils.hpp for further details on ping testing.

  RETURNS:
  number of test failures: 0 if all is working well.
*/
OL_WARN_UNUSED_RESULT int RunCovarianceTests();

}  // end namespace optimal_learning

#endif  // OPTIMAL_LEARNING_EPI_SRC_CPP_GPP_COVARIANCE_TEST_HPP_
