/*!
  \file gpp_expected_improvement_gpu_test.hpp
  \rst
  Functions for testing expected improvement functions on GPU.

  Tests are broken into two main groups:

  * consistency test against analytical 1,0-EI result
  * compare with CPU(MC) results

  .. Note:: These tests do not run if GPU computation (``OL_GPU_ENABLED``) is disabled.
\endrst*/

#ifndef MOE_OPTIMAL_LEARNING_CPP_GPP_EXPECTED_IMPROVEMENT_GPU_TEST_HPP_
#define MOE_OPTIMAL_LEARNING_CPP_GPP_EXPECTED_IMPROVEMENT_GPU_TEST_HPP_

#include "gpp_common.hpp"

namespace optimal_learning {

/*!\rst
  Invoke all tests for GPU functions.

  \return
    number of test failures: 0 if all is working well.
\endrst*/
OL_WARN_UNUSED_RESULT int RunGPUTests();

}  // end namespace optimal_learning

#endif  // MOE_OPTIMAL_LEARNING_CPP_GPP_EXPECTED_IMPROVEMENT_GPU_TEST_HPP_
