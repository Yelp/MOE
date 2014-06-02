/*!
  \file gpp_covariance_test.hpp
  \rst
  Simple function call to run unit tests for covariance functions defined in gpp_covariance.cpp.
  This function consists of a battery of ping tests that verify the correctness of derivatives and hessians
  of the various covariance functions available in gpp_covariance.hpp.
\endrst*/

#ifndef MOE_OPTIMAL_LEARNING_CPP_GPP_COVARIANCE_TEST_HPP_
#define MOE_OPTIMAL_LEARNING_CPP_GPP_COVARIANCE_TEST_HPP_

#include "gpp_common.hpp"

namespace optimal_learning {

/*!\rst
  Ping tests the covariance functions implemented in gpp_covariance.cpp
  Currently the only covariance option is SquareExponential.

  See gpp_test_utils.hpp for further details on ping testing.

  \return
    number of test failures: 0 if all is working well.
\endrst*/
OL_WARN_UNUSED_RESULT int RunCovarianceTests();

}  // end namespace optimal_learning

#endif  // MOE_OPTIMAL_LEARNING_CPP_GPP_COVARIANCE_TEST_HPP_
