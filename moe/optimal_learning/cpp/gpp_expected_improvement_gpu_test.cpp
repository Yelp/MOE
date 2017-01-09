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

/*!\rst
  This function is the same as ``ExpectedImprovementOptimizationMultipleSamplesTest``
  in ``gpp_math_test.cpp``. Refer to ``gpp_math_test.cpp`` for detailed documentation.
\endrst*/
int CudaExpectedImprovementOptimizationMultipleSamplesTest() {
  using DomainType = TensorProductDomain;
  const int dim = 3;

  int total_errors = 0;
  int current_errors = 0;

  // gradient descent parameters
  const double gamma = 0.5;
  const double pre_mult = 1.0;
  const double max_relative_change = 1.0;
  const double tolerance = 1.0e-5;
  const int max_gradient_descent_steps = 100;
  const int max_num_restarts = 3;
  const int num_steps_averaged = 0;
  const int num_multistarts = 20;
  GradientDescentParameters gd_params(num_multistarts, max_gradient_descent_steps,
                                      max_num_restarts, num_steps_averaged, gamma, pre_mult,
                                      max_relative_change, tolerance);

  // grid search parameters
  int num_grid_search_points = 5000;

  // q,p-EI computation parameters
  const int num_to_sample = 3;
  const int num_being_sampled = 0;

  std::vector<double> points_being_sampled(dim*num_being_sampled);
  int max_int_steps = 10000;

  // random number generators
  UniformRandomGenerator uniform_generator(314);
  boost::uniform_real<double> uniform_double_hyperparameter(0.4, 1.3);
  boost::uniform_real<double> uniform_double_lower_bound(-5.0, -4.5);
  boost::uniform_real<double> uniform_double_upper_bound(4.0, 4.5);

  static const int kMaxNumThreads = 1;
  ThreadSchedule thread_schedule(kMaxNumThreads, omp_sched_static);

  const int num_sampled = 50;
  std::vector<double> noise_variance(num_sampled, 0.002);
  MockGaussianProcessPriorData<DomainType> mock_gp_data(SquareExponential(dim, 1.0, 1.0),
                                                        noise_variance, dim, num_sampled,
                                                        uniform_double_lower_bound,
                                                        uniform_double_upper_bound,
                                                        uniform_double_hyperparameter,
                                                        &uniform_generator);

  // we will optimize over the expanded region
  std::vector<ClosedInterval> domain_bounds(mock_gp_data.domain_bounds);
  ExpandDomainBounds(1.5, &domain_bounds);
  DomainType domain(domain_bounds.data(), dim);

  const int which_gpu = 0;

  // optimize EI using grid search to set the baseline
  bool found_flag = false;
  std::vector<double> grid_search_best_point_set(dim*num_to_sample);
  CudaComputeOptimalPointsToSampleViaLatinHypercubeSearch(*mock_gp_data.gaussian_process_ptr, domain,
                                                          thread_schedule, points_being_sampled.data(),
                                                          num_grid_search_points, num_to_sample,
                                                          num_being_sampled, mock_gp_data.best_so_far,
                                                          max_int_steps, which_gpu, &found_flag,
                                                          &uniform_generator, grid_search_best_point_set.data());
  if (!found_flag) {
    ++total_errors;
  }

  // optimize EI using gradient descent
  found_flag = false;
  bool lhc_search_only = false;
  std::vector<double> best_points_to_sample(dim*num_to_sample);
  CudaComputeOptimalPointsToSample(*mock_gp_data.gaussian_process_ptr, gd_params, domain,
                                   thread_schedule, points_being_sampled.data(),
                                   num_to_sample, num_being_sampled, mock_gp_data.best_so_far,
                                   max_int_steps, lhc_search_only,
                                   num_grid_search_points, which_gpu, &found_flag,
                                   &uniform_generator, best_points_to_sample.data());
  if (!found_flag) {
    ++total_errors;
  }

  // check points are in domain
  RepeatedDomain<DomainType> repeated_domain(domain, num_to_sample);
  if (!repeated_domain.CheckPointInside(best_points_to_sample.data())) {
    ++current_errors;
  }
#ifdef OL_ERROR_PRINT
  if (current_errors != 0) {
    OL_ERROR_PRINTF("ERROR: points were not in domain!  points:\n");
    PrintMatrixTrans(best_points_to_sample.data(), num_to_sample, dim);
    OL_ERROR_PRINTF("domain:\n");
    PrintDomainBounds(domain_bounds.data(), dim);
  }
#endif
  total_errors += current_errors;

  // check points are distinct; points within tolerance are considered non-distinct
  const double distinct_point_tolerance = 1.0e-5;
  current_errors = CheckPointsAreDistinct(best_points_to_sample.data(), num_to_sample, dim, distinct_point_tolerance);
#ifdef OL_ERROR_PRINT
  if (current_errors != 0) {
    OL_ERROR_PRINTF("ERROR: points were not distinct!  points:\n");
    PrintMatrixTrans(best_points_to_sample.data(), num_to_sample, dim);
  }
#endif
  total_errors += current_errors;

  // results
  double ei_optimized, ei_grid_search;
  std::vector<double> grad_ei(dim*num_to_sample);

  // set up evaluators and state to check results
  double tolerance_result = tolerance;
  {
    max_int_steps = 100000000;  // evaluate the final results with high accuracy
    tolerance_result = 7e-3;  // reduce b/c we cannot achieve full accuracy in the monte-carlo case
    // while still having this test run in a reasonable amt of time
    bool configure_for_gradients = true;
    CudaExpectedImprovementEvaluator ei_evaluator(*mock_gp_data.gaussian_process_ptr,
                                                  max_int_steps, mock_gp_data.best_so_far, which_gpu);
    CudaExpectedImprovementEvaluator::StateType ei_state(ei_evaluator, best_points_to_sample.data(),
                                                         points_being_sampled.data(), num_to_sample,
                                                         num_being_sampled, configure_for_gradients,
                                                         &uniform_generator);

    ei_optimized = ei_evaluator.ComputeExpectedImprovement(&ei_state);
    ei_evaluator.ComputeGradExpectedImprovement(&ei_state, grad_ei.data());

    CudaExpectedImprovementEvaluator::StateType ei_state_grid_search(ei_evaluator,
                                                                     grid_search_best_point_set.data(),
                                                                     points_being_sampled.data(), num_to_sample,
                                                                     num_being_sampled, configure_for_gradients,
                                                                     &uniform_generator);
    ei_grid_search = ei_evaluator.ComputeExpectedImprovement(&ei_state_grid_search);
  }

  printf("optimized EI: %.18E, grid_search_EI: %.18E\n", ei_optimized, ei_grid_search);
  printf("grad_EI: "); PrintMatrixTrans(grad_ei.data(), num_to_sample, dim);

  if (ei_optimized < ei_grid_search) {
    ++total_errors;
  }

  current_errors = 0;
  for (const auto& entry : grad_ei) {
    if (!CheckDoubleWithinRelative(entry, 0.0, tolerance_result)) {
      ++current_errors;
    }
  }
  total_errors += current_errors;

  return total_errors;
}

/*!\rst
  This function tests GPU optimizer in the case of 1-EI, in which GPU optimizer
  merely calls optimizer in ``gpp_math.hpp`` since we are not using MC simulation
  anyway. The function definition is mostly copied from ``ExpectedImprovementOptimizationTestCore`` in ``gpp_math_test.cpp`` by preserving
  its 1-EI part.
\endrst*/
int CudaExpectedImprovementOptimizationAnalyticTest() {
  using DomainType = TensorProductDomain;
  const int dim = 3;
  const int which_gpu = 0;

  int total_errors = 0;
  int current_errors = 0;

  // gradient descent parameters
  const double gamma = 0.5;
  const double pre_mult = 1.4;
  const double max_relative_change = 1.0;
  const double tolerance = 1.0e-7;
  const int max_gradient_descent_steps = 1000;
  const int max_num_restarts = 10;
  const int num_steps_averaged = 0;
  const int num_multistarts = 20;
  GradientDescentParameters gd_params(num_multistarts, max_gradient_descent_steps,
                                      max_num_restarts, num_steps_averaged,
                                      gamma, pre_mult, max_relative_change, tolerance);

  // grid search parameters
  int num_grid_search_points = 10000;

  // 1,p-EI computation parameters
  const int num_to_sample = 1;
  int num_being_sampled = 0;
  int max_int_steps = 6000;

  // random number generators
  UniformRandomGenerator uniform_generator(314);
  boost::uniform_real<double> uniform_double_hyperparameter(0.4, 1.3);
  boost::uniform_real<double> uniform_double_lower_bound(-2.0, 0.5);
  boost::uniform_real<double> uniform_double_upper_bound(1.0, 2.5);

  static const int kMaxNumThreads = 4;
  ThreadSchedule thread_schedule(kMaxNumThreads, omp_sched_static);

  int num_sampled = 20;

  std::vector<double> noise_variance(num_sampled, 0.002);
  MockGaussianProcessPriorData<DomainType> mock_gp_data(SquareExponential(dim, 1.0, 1.0), noise_variance,
                                                        dim, num_sampled, uniform_double_lower_bound,
                                                        uniform_double_upper_bound,
                                                        uniform_double_hyperparameter,
                                                        &uniform_generator);

  // we will optimize over the expanded region
  std::vector<ClosedInterval> domain_bounds(mock_gp_data.domain_bounds);
  ExpandDomainBounds(1.5, &domain_bounds);
  DomainType domain(domain_bounds.data(), dim);

  std::vector<double> points_being_sampled(dim*num_being_sampled);

  // optimize EI
  bool found_flag = false;
  std::vector<double> grid_search_best_point(dim*num_to_sample);
  CudaComputeOptimalPointsToSampleViaLatinHypercubeSearch(*mock_gp_data.gaussian_process_ptr, domain,
                                                          thread_schedule, points_being_sampled.data(),
                                                          num_grid_search_points, num_to_sample,
                                                          num_being_sampled, mock_gp_data.best_so_far,
                                                          max_int_steps, which_gpu, &found_flag,
                                                          &uniform_generator, grid_search_best_point.data());
  if (!found_flag) {
    ++total_errors;
  }

  std::vector<double> next_point(dim*num_to_sample);
  found_flag = false;
  CudaComputeOptimalPointsToSampleWithRandomStarts(*mock_gp_data.gaussian_process_ptr, gd_params,
                                                   domain, thread_schedule, points_being_sampled.data(),
                                                   num_to_sample, num_being_sampled,
                                                   mock_gp_data.best_so_far, max_int_steps,
                                                   which_gpu, &found_flag,
                                                   &uniform_generator, next_point.data());
  if (!found_flag) {
    ++total_errors;
  }

  printf("next best point  : "); PrintMatrixTrans(next_point.data(), num_to_sample, dim);
  printf("grid search point: "); PrintMatrixTrans(grid_search_best_point.data(), num_to_sample, dim);

  // results
  double ei_optimized, ei_grid_search;
  std::vector<double> grad_ei(dim*num_to_sample);

  // set up evaluators and state to check results
  double tolerance_result = tolerance;
  bool configure_for_gradients = true;

  OnePotentialSampleExpectedImprovementEvaluator ei_evaluator(*mock_gp_data.gaussian_process_ptr,
                                                              mock_gp_data.best_so_far);
  OnePotentialSampleExpectedImprovementEvaluator::StateType ei_state(ei_evaluator, next_point.data(),
                                                                     configure_for_gradients);

  ei_optimized = ei_evaluator.ComputeExpectedImprovement(&ei_state);
  ei_evaluator.ComputeGradExpectedImprovement(&ei_state, grad_ei.data());

  ei_state.SetCurrentPoint(ei_evaluator, grid_search_best_point.data());
  ei_grid_search = ei_evaluator.ComputeExpectedImprovement(&ei_state);

  printf("optimized EI: %.18E, grid_search_EI: %.18E\n", ei_optimized, ei_grid_search);
  printf("grad_EI: "); PrintMatrixTrans(grad_ei.data(), num_to_sample, dim);

  if (ei_optimized < ei_grid_search) {
    ++total_errors;
  }

  current_errors = 0;
  for (const auto& entry : grad_ei) {
    if (!CheckDoubleWithinRelative(entry, 0.0, tolerance_result)) {
      ++current_errors;
    }
  }
  total_errors += current_errors;

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
    OL_SUCCESS_PRINTF("cudaEI vs cpuEI consistency check succeeded\n");
  }
  total_errors += error;

  error = CudaExpectedImprovementOptimizationAnalyticTest();
  if (error != 0) {
    OL_FAILURE_PRINTF("GPU 1-EI optimization failed\n");
  } else {
    OL_SUCCESS_PRINTF("GPU 1-EI optimization succeeded\n");
  }
  total_errors += error;

  error = CudaExpectedImprovementOptimizationMultipleSamplesTest();
  if (error != 0) {
    OL_FAILURE_PRINTF("GPU optimizer failed\n");
  } else {
    OL_SUCCESS_PRINTF("GPU optimizer succeeded\n");
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
