/*!
  \file gpp_expected_improvement_gpu_test.cpp
  \rst
  Routines to test the functions in gpp_expected_improvement_gpu.cpp.

  The tests verify ExpectedImprovementGPUEvaluator from gpp_expected_improvement_gpu.cpp.

  1. Ping testing (verifying analytic gradient computation against finite difference approximations)

  2. Monte-Carlo EI vs analytic EI validation: the monte-carlo versions are run to "high" accuracy and checked against
     analytic formulae when applicable

  3. GPU EI vs CPU EI: both use monte-carlo version and run consistency check on various random sample points
\endrst*/

#include "gpp_expected_improvement_gpu_test.hpp"

#include <vector>
#include <algorithm>

#include <boost/random/uniform_real.hpp>  // NOLINT(build/include_order)

#include "gpp_common.hpp"
#include "gpp_exception.hpp"
#include "gpp_math.hpp"
#include "gpp_random.hpp"
#include "gpp_test_utils.hpp"

namespace optimal_learning {

#ifdef OL_GPU_ENABLED

/*!\rst
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

  MockExpectedImprovementEnvironment EI_environment;

  std::vector<double> lengths(dim);
  std::vector<double> noise_variance(num_sampled, 0.0);
  std::vector<double> grad_EI_cuda(dim);
  std::vector<double> grad_EI_one_potential_sample(dim);
  double EI_cuda;
  double EI_one_potential_sample;

  for (int i = 0; i < 40; ++i) {
    EI_environment.Initialize(dim, num_to_sample, num_being_sampled, num_sampled, &uniform_generator);
    for (int j = 0; j < dim; ++j) {
      lengths[j] = uniform_double(uniform_generator.engine);
    }
    SquareExponential sqexp_covariance(dim, alpha, lengths);
    GaussianProcess gaussian_process(sqexp_covariance, EI_environment.points_sampled(), EI_environment.points_sampled_value(), noise_variance.data(), dim, num_sampled);

    OnePotentialSampleExpectedImprovementEvaluator one_potential_sample_ei_evaluator(gaussian_process, best_so_far);
    OnePotentialSampleExpectedImprovementEvaluator::StateType one_potential_sample_ei_state(one_potential_sample_ei_evaluator, EI_environment.points_to_sample(), configure_for_gradients);

    CudaExpectedImprovementEvaluator cuda_ei_evaluator(gaussian_process, num_mc_iter, best_so_far, which_gpu);
    CudaExpectedImprovementEvaluator::StateType cuda_ei_state(cuda_ei_evaluator, EI_environment.points_to_sample(), EI_environment.points_being_sampled(), num_to_sample, num_being_sampled, configure_for_gradients, &uniform_generator);

    EI_cuda = cuda_ei_evaluator.ComputeObjectiveFunction(&cuda_ei_state);
    cuda_ei_evaluator.ComputeGradObjectiveFunction(&cuda_ei_state, grad_EI_cuda.data());
    EI_one_potential_sample = one_potential_sample_ei_evaluator.ComputeObjectiveFunction(&one_potential_sample_ei_state);
    one_potential_sample_ei_evaluator.ComputeGradObjectiveFunction(&one_potential_sample_ei_state, grad_EI_one_potential_sample.data());

    int ei_errors_this_iteration = 0;
    if (!CheckDoubleWithinRelative(EI_cuda, EI_one_potential_sample, 5.0e-3)) {
      ++ei_errors_this_iteration;
    }
    if (ei_errors_this_iteration != 0) {
      OL_PARTIAL_FAILURE_PRINTF("in EI on iteration %d\n", i);
    }
    total_errors += ei_errors_this_iteration;

    int grad_ei_errors_this_iteration = 0;
    for (int j = 0; j < dim; ++j) {
      if (!CheckDoubleWithinRelative(grad_EI_cuda[j], grad_EI_one_potential_sample[j], 4.5e-3)) {
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

  MockExpectedImprovementEnvironment EI_environment;

  std::vector<double> lengths(dim);
  std::vector<double> noise_variance(num_sampled, 0.0);
  std::vector<double> grad_EI_cpu(dim*num_to_sample);
  std::vector<double> grad_EI_gpu(dim*num_to_sample);
  std::vector<double> normal_random_table;
  double EI_cpu;
  double EI_gpu;
  int cpu_num_iter;

  for (int i = 0; i < 10; ++i) {
    EI_environment.Initialize(dim, num_to_sample, num_being_sampled, num_sampled, &uniform_generator);
    for (int j = 0; j < dim; ++j) {
      lengths[j] = uniform_double(uniform_generator.engine);
    }
    SquareExponential sqexp_covariance(dim, alpha, lengths);
    GaussianProcess gaussian_process(sqexp_covariance, EI_environment.points_sampled(), EI_environment.points_sampled_value(), noise_variance.data(), dim, num_sampled);

    CudaExpectedImprovementEvaluator cuda_ei_evaluator(gaussian_process, num_mc_iter, best_so_far, which_gpu);
    CudaExpectedImprovementEvaluator::StateType cuda_ei_state(cuda_ei_evaluator, EI_environment.points_to_sample(), EI_environment.points_being_sampled(), num_to_sample, num_being_sampled, configure_for_gradients, &uniform_generator, configure_for_test);

    EI_gpu = cuda_ei_evaluator.ComputeObjectiveFunction(&cuda_ei_state);

    // setup cpu EI computation
    normal_random_table = cuda_ei_state.random_number_EI;
    cpu_num_iter = normal_random_table.size()/ (num_being_sampled + num_to_sample);
    NormalRNGSimulator normal_rng_forEI(normal_random_table);
    ExpectedImprovementEvaluator ei_evaluator_forEI(gaussian_process, cpu_num_iter, best_so_far);
    ExpectedImprovementEvaluator::StateType ei_state_forEI(ei_evaluator_forEI, EI_environment.points_to_sample(), EI_environment.points_being_sampled(), num_to_sample, num_being_sampled, false, &normal_rng_forEI);
    EI_cpu = ei_evaluator_forEI.ComputeObjectiveFunction(&ei_state_forEI);

    // setup cpu gradEI computation
    cuda_ei_evaluator.ComputeGradObjectiveFunction(&cuda_ei_state, grad_EI_gpu.data());

    normal_random_table = cuda_ei_state.random_number_gradEI;
    cpu_num_iter = normal_random_table.size()/ (num_being_sampled + num_to_sample);
    NormalRNGSimulator normal_rng_forGradEI(normal_random_table);
    ExpectedImprovementEvaluator ei_evaluator_forGradEI(gaussian_process, cpu_num_iter, best_so_far);
    ExpectedImprovementEvaluator::StateType ei_state_forGradEI(ei_evaluator_forGradEI, EI_environment.points_to_sample(), EI_environment.points_being_sampled(), num_to_sample, num_being_sampled, true, &normal_rng_forGradEI);
    ei_evaluator_forGradEI.ComputeGradObjectiveFunction(&ei_state_forGradEI, grad_EI_cpu.data());

    int ei_errors_this_iteration = 0;
    if (!CheckDoubleWithinRelative(EI_cpu, EI_gpu, 1.0e-12)) {
      ++ei_errors_this_iteration;
    }
    if (ei_errors_this_iteration != 0) {
      OL_PARTIAL_FAILURE_PRINTF("in EI on iteration %d\n", i);
    }
    total_errors += ei_errors_this_iteration;

    int grad_ei_errors_this_iteration = 0;
    for (int j = 0; j < dim*num_to_sample; ++j) {
      if (!CheckDoubleWithinRelative(grad_EI_cpu[j], grad_EI_gpu[j], 1.0e-12)) {
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

#else

int RunCudaEIConsistencyTests() {
  OL_WARNING_PRINTF("no gpu component is enabled, this test did not run.\n");
  return 0;
}

int RunCudaEIvsCpuEITests() {
  OL_WARNING_PRINTF("no gpu component is enabled, this test did not run.\n");
  return 0;
}

#endif

}  // end namespace optimal_learning
