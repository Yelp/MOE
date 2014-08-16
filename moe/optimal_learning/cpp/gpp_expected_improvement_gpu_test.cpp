/*!
  \file gpp_expected_improvement_gpu_test.cpp
  \rst
  Routines to test the functions in gpp_expected_improvement_gpu.cpp.

  The tests verify ExpectedImprovementGPUEvaluator from gpp_expected_improvement_gpu.cpp.

  1. Monte-Carlo EI vs analytic EI validation: the monte-carlo versions are run to "high" accuracy and checked against
     analytic formulae when applicable

  2. GPU EI vs CPU EI: both use monte-carlo version and run consistency check on various random sample points
\endrst*/

#include "gpp_expected_improvement_gpu_test.hpp"

#include <vector>
#include <algorithm>

#include <boost/random/uniform_real.hpp>  // NOLINT(build/include_order)

#include "gpp_common.hpp"
#include "gpp_exception.hpp"
#include "gpp_expected_improvement_gpu.hpp"
#include "gpp_math.hpp"
#include "gpp_random.hpp"
#include "gpp_test_utils.hpp"

namespace optimal_learning {

#ifdef OL_GPU_ENABLED

namespace {

/*!\rst
  Test that the EI + grad EI computation (using MC integration) is consistent
  with the special analytic case of EI when there is only *ONE* potential point
  to sample.

  Generates a set of 40 random test cases for expected improvement with only one potential sample.
  The general EI (which uses MC integration) is evaluated to reasonably high accuracy (while not taking too long to run)
  and compared against the analytic formula version for consistency.  The gradients (spatial) of EI are also checked.

  \return
    number of cases where analytic and monte-carlo EI do not match
\endrst*/
int RunCudaEIConsistencyTests() {
  int total_errors = 0;

  int which_gpu = 1;
  const int num_mc_iter = 20000000;
  const int dim = 3;
  const int num_being_sampled = 0;
  const int num_to_sample = 1;
  const int num_sampled = 7;

  double alpha = 2.80723;
  // set best_so_far to be larger than max(points_sampled_value) (but don't make it huge or stability will be suffer)
  double best_so_far = 10.0;
  bool configure_for_gradients = true;

  UniformRandomGenerator uniform_generator(31278);
  boost::uniform_real<double> uniform_double(0.5, 2.5);

  MockExpectedImprovementEnvironment ei_environment;

  std::vector<double> lengths(dim);
  std::vector<double> noise_variance(num_sampled, 0.0);
  std::vector<double> grad_ei_cuda(dim);
  std::vector<double> grad_ei_one_potential_sample(dim);
  double ei_cuda;
  double ei_one_potential_sample;

  for (int i = 0; i < 40; ++i) {
    ei_environment.Initialize(dim, num_to_sample, num_being_sampled, num_sampled, &uniform_generator);
    for (int j = 0; j < dim; ++j) {
      lengths[j] = uniform_double(uniform_generator.engine);
    }
    SquareExponential sqexp_covariance(dim, alpha, lengths);
    GaussianProcess gaussian_process(sqexp_covariance, ei_environment.points_sampled(), ei_environment.points_sampled_value(), noise_variance.data(), dim, num_sampled);

    OnePotentialSampleExpectedImprovementEvaluator one_potential_sample_ei_evaluator(gaussian_process, best_so_far);
    OnePotentialSampleExpectedImprovementEvaluator::StateType one_potential_sample_ei_state(one_potential_sample_ei_evaluator, ei_environment.points_to_sample(), configure_for_gradients);

    CudaExpectedImprovementEvaluator cuda_ei_evaluator(gaussian_process, num_mc_iter, best_so_far, which_gpu);
    CudaExpectedImprovementEvaluator::StateType cuda_ei_state(cuda_ei_evaluator, ei_environment.points_to_sample(), ei_environment.points_being_sampled(), num_to_sample, num_being_sampled, configure_for_gradients, &uniform_generator);

    ei_cuda = cuda_ei_evaluator.ComputeObjectiveFunction(&cuda_ei_state);
    cuda_ei_evaluator.ComputeGradObjectiveFunction(&cuda_ei_state, grad_ei_cuda.data());
    ei_one_potential_sample = one_potential_sample_ei_evaluator.ComputeObjectiveFunction(&one_potential_sample_ei_state);
    one_potential_sample_ei_evaluator.ComputeGradObjectiveFunction(&one_potential_sample_ei_state, grad_ei_one_potential_sample.data());

    int ei_errors_this_iteration = 0;
    if (!CheckDoubleWithinRelative(ei_cuda, ei_one_potential_sample, 5.0e-3)) {
      ++ei_errors_this_iteration;
    }
    if (ei_errors_this_iteration != 0) {
      OL_PARTIAL_FAILURE_PRINTF("in EI on iteration %d\n", i);
    }
    total_errors += ei_errors_this_iteration;

    int grad_ei_errors_this_iteration = 0;
    for (int j = 0; j < dim; ++j) {
      if (!CheckDoubleWithinRelative(grad_ei_cuda[j], grad_ei_one_potential_sample[j], 4.5e-3)) {
        ++grad_ei_errors_this_iteration;
      }
    }

    if (grad_ei_errors_this_iteration != 0) {
      OL_PARTIAL_FAILURE_PRINTF("in EI gradients on iteration %d\n", i);
    }
    total_errors += grad_ei_errors_this_iteration;
  }

  if (total_errors != 0) {
    OL_PARTIAL_FAILURE_PRINTF("comparing MC EI to analytic EI failed with %d total_errors\n\n", total_errors);
  } else {
    OL_PARTIAL_SUCCESS_PRINTF("comparing MC EI to analytic EI passed\n");
  }
  return total_errors;
}

/*!\rst
  Tests that the general EI + grad EI computation on CPU (using MC integration) is consistent
  with the computation on GPU. We use exactly the same sequences of normal random numbers on
  CPU and GPU so that they are supposed to output the same result even if the number of MC
  iterations is small.

  Generates a set of 10 random test cases for genral q,p-EI computed on cpu vs gpu.
  The computations on cpu and gpu use the same set of normal random numbers for MC
  simulation, so that we can make sure the outputs should be consistent, even with
  a relative small number of MC iteration.

  \return
    number of cases where outputs from cpu and gpu do not match.
\endrst*/
int RunCudaEIvsCpuEITests() {
  int total_errors = 0;

  int which_gpu = 1;
  const int num_mc_iter = 40000;
  const int dim = 3;
  const int num_being_sampled = 4;
  const int num_to_sample = 4;
  const int num_sampled = 20;

  double alpha = 2.80723;
  // set best_so_far to be larger than max(points_sampled_value) (but don't make it huge or stability will be suffer)
  double best_so_far = 10.0;
  bool configure_for_gradients = true;
  bool configure_for_test = true;

  UniformRandomGenerator uniform_generator(31278);
  boost::uniform_real<double> uniform_double(0.5, 2.5);

  MockExpectedImprovementEnvironment ei_environment;

  std::vector<double> lengths(dim);
  std::vector<double> noise_variance(num_sampled, 0.0);
  std::vector<double> grad_ei_cpu(dim*num_to_sample);
  std::vector<double> grad_ei_gpu(dim*num_to_sample);
  std::vector<double> normal_random_table;
  double ei_cpu;
  double ei_gpu;
  int cpu_num_iter;

  for (int i = 0; i < 10; ++i) {
    ei_environment.Initialize(dim, num_to_sample, num_being_sampled, num_sampled, &uniform_generator);
    for (int j = 0; j < dim; ++j) {
      lengths[j] = uniform_double(uniform_generator.engine);
    }
    SquareExponential sqexp_covariance(dim, alpha, lengths);
    GaussianProcess gaussian_process(sqexp_covariance, ei_environment.points_sampled(), ei_environment.points_sampled_value(), noise_variance.data(), dim, num_sampled);

    CudaExpectedImprovementEvaluator cuda_ei_evaluator(gaussian_process, num_mc_iter, best_so_far, which_gpu);
    CudaExpectedImprovementEvaluator::StateType cuda_ei_state(cuda_ei_evaluator, ei_environment.points_to_sample(), ei_environment.points_being_sampled(), num_to_sample, num_being_sampled, configure_for_gradients, &uniform_generator, configure_for_test);

    ei_gpu = cuda_ei_evaluator.ComputeObjectiveFunction(&cuda_ei_state);

    // setup cpu EI computation
    normal_random_table = cuda_ei_state.random_number_ei;
    cpu_num_iter = normal_random_table.size()/ (num_being_sampled + num_to_sample);
    NormalRNGSimulator normal_rng_for_ei(normal_random_table);
    ExpectedImprovementEvaluator ei_evaluator_for_ei(gaussian_process, cpu_num_iter, best_so_far);
    ExpectedImprovementEvaluator::StateType ei_state_for_ei(ei_evaluator_for_ei, ei_environment.points_to_sample(), ei_environment.points_being_sampled(), num_to_sample, num_being_sampled, false, &normal_rng_for_ei);
    ei_cpu = ei_evaluator_for_ei.ComputeObjectiveFunction(&ei_state_for_ei);

    // setup cpu gradEI computation
    cuda_ei_evaluator.ComputeGradObjectiveFunction(&cuda_ei_state, grad_ei_gpu.data());

    normal_random_table = cuda_ei_state.random_number_grad_ei;
    cpu_num_iter = normal_random_table.size()/ (num_being_sampled + num_to_sample);
    NormalRNGSimulator normal_rng_for_grad_ei(normal_random_table);
    ExpectedImprovementEvaluator ei_evaluator_for_grad_ei(gaussian_process, cpu_num_iter, best_so_far);
    ExpectedImprovementEvaluator::StateType ei_state_for_grad_ei(ei_evaluator_for_grad_ei, ei_environment.points_to_sample(), ei_environment.points_being_sampled(), num_to_sample, num_being_sampled, true, &normal_rng_for_grad_ei);
    ei_evaluator_for_grad_ei.ComputeGradObjectiveFunction(&ei_state_for_grad_ei, grad_ei_cpu.data());

    int ei_errors_this_iteration = 0;
    if (!CheckDoubleWithinRelative(ei_cpu, ei_gpu, 1.0e-12)) {
      ++ei_errors_this_iteration;
    }
    if (ei_errors_this_iteration != 0) {
      OL_PARTIAL_FAILURE_PRINTF("in EI on iteration %d\n", i);
    }
    total_errors += ei_errors_this_iteration;

    int grad_ei_errors_this_iteration = 0;
    for (int j = 0; j < dim*num_to_sample; ++j) {
      if (!CheckDoubleWithinRelative(grad_ei_cpu[j], grad_ei_gpu[j], 1.0e-12)) {
        ++grad_ei_errors_this_iteration;
      }
    }

    if (grad_ei_errors_this_iteration != 0) {
      OL_PARTIAL_FAILURE_PRINTF("in EI gradients on iteration %d\n", i);
    }
    total_errors += grad_ei_errors_this_iteration;
  }

  if (total_errors != 0) {
    OL_PARTIAL_FAILURE_PRINTF("comparing cpu EI to gpu EI failed with %d total_errors\n\n", total_errors);
  } else {
    OL_PARTIAL_SUCCESS_PRINTF("comparing cpu EI to gpu EI passed\n");
  }
  return total_errors;
}

}  // end unnamed namespace

/*!\rst
  Invoke all tests for GPU functions.

  \return
    number of test failures: 0 if all is working well.
\endrst*/
int RunGPUTests() {
  int total_errors = 0;
  int error = RunCudaEIConsistencyTests();
  if (error != 0) {
    OL_FAILURE_PRINTF("analytic, Cuda EI do not match for 1 potential sample case\n");
  } else {
    OL_SUCCESS_PRINTF("analytic, Cuda EI match for 1 potential sample case\n");
  }
  total_errors += error;

  error = RunCudaEIvsCpuEITests();
  if (error != 0) {
    OL_FAILURE_PRINTF("cudaEI vs cpuEI consistency check failed\n");
  } else {
    OL_SUCCESS_PRINTF("cudaEI vs cpuEI consistency check successed\n");
  }
  total_errors += error;
  return total_errors;
}

#else  // OL_GPU_ENABLED

int RunGPUTests() {
  OL_WARNING_PRINTF("no gpu component is enabled, this test did not run.\n");
  return 0;
}

#endif  // OL_GPU_ENABLED

}  // end namespace optimal_learning
