/*!
  \file gpp_expected_improvement_gpu_test.hpp
  \rst
  Functions for testing expected improvement functions on GPU.

  Tests are broken into two main groups:

  * consistency test against analytical 1,0-EI result
  * compare with CPU(MC) results

\endrst*/

#ifndef MOE_OPTIMAL_LEARNING_CPP_GPP_EXPECTED_IMPROVEMENT_GPU_TEST_HPP_
#define MOE_OPTIMAL_LEARNING_CPP_GPP_EXPECTED_IMPROVEMENT_GPU_TEST_HPP_

#include <vector>

#include "gpp_common.hpp"
#include "gpp_domain.hpp"
#include "gpp_exception.hpp"
#include "gpp_expected_improvement_gpu.hpp"
#include "gpp_math.hpp"
#include "gpp_test_utils.hpp"

namespace optimal_learning {
/*!\rst
  Tests that the general EI + grad EI computation (using MC integration) is consistent
  with the special analytic case of EI when there is only *ONE* potential point
  to sample.

  \return
    number of test failures: 0 if all is working well.
\endrst*/
OL_WARN_UNUSED_RESULT int RunCudaEIConsistencyTests();

/*!\rst
  Tests that the general EI + grad EI computation on CPU (using MC integration) is consistent
  with the computation on GPU. We use exactly the same sequences of normal random numbers on
  CPU and GPU so that they are supposed to output the same result even if the number of MC
  iterations is small.

  \return
    number of test failures: 0 if all is working well.
\endrst*/
OL_WARN_UNUSED_RESULT int RunCudaEIvsCpuEITests();

}  // end namespace optimal_learning
#endif  // MOE_OPTIMAL_LEARNING_CPP_GPP_EXPECTED_IMPROVEMENT_GPU_TEST_HPP_

