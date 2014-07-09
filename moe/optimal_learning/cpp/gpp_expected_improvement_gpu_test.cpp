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
#include "gpp_expected_improvement_gpu.hpp"
#include "gpp_math.hpp"
#include "gpp_random.hpp"
#include "gpp_test_utils.hpp"

namespace optimal_learning {
// PingCudaExpectedImprovement function def begins
PingCudaExpectedImprovement::PingCudaExpectedImprovement(double const * restrict lengths, double const * restrict points_being_sampled, double const * restrict points_sampled, double const * restrict points_sampled_value, double alpha, double best_so_far, int dim, int num_to_sample, int num_being_sampled, int num_sampled, int num_mc_iter)
  : dim_(dim),
    num_to_sample_(num_to_sample),
    num_being_sampled_(num_being_sampled),
    num_sampled_(num_sampled),
    gradients_already_computed_(false),
    noise_variance_(num_sampled_, 0.0),
    points_sampled_(points_sampled, points_sampled + dim_*num_sampled_),
    points_sampled_value_(points_sampled_value, points_sampled_value + num_sampled_),
    points_being_sampled_(points_being_sampled, points_being_sampled + num_being_sampled_*dim_),
    grad_EI_(num_to_sample_*dim_),
    sqexp_covariance_(dim_, alpha, lengths),
    gaussian_process_(sqexp_covariance_, points_sampled_.data(), points_sampled_value_.data(), noise_variance_.data(), dim_, num_sampled_), ei_evaluator_(gaussian_process_, num_mc_iter, best_so_far) {
}

void PingCudaExpectedImprovement::GetInputSizes(int * num_rows, int * num_cols) const noexcept {
  *num_rows = dim_;
  *num_cols = num_to_sample_;
}

int PingCudaExpectedImprovement::GetGradientsSize() const noexcept {
  return dim_*GetOutputSize();
}

int PingCudaExpectedImprovement::GetOutputSize() const noexcept {
  return num_to_sample_;
}

void PingCudaExpectedImprovement::EvaluateAndStoreAnalyticGradient(double const * restrict points_to_sample, double * restrict gradients) noexcept {
  if (gradients_already_computed_ == true) {
    OL_WARNING_PRINTF("WARNING: grad_EI data already set.  Overwriting...\n");
  }
  gradients_already_computed_ = true;

  UniformRandomGenerator uniform_rng(3141);
  bool configure_for_gradients = true;
  CudaExpectedImprovementEvaluator::StateType ei_state(ei_evaluator_, points_to_sample, points_being_sampled_.data(), num_to_sample_, num_being_sampled_, configure_for_gradients, &uniform_rng);
  ei_evaluator_.ComputeGradExpectedImprovement(&ei_state, grad_EI_.data());

  if (gradients != nullptr) {
    std::copy(grad_EI_.begin(), grad_EI_.end(), gradients);
  }
}

double PingCudaExpectedImprovement::GetAnalyticGradient(int row_index, int column_index, int OL_UNUSED(output_index)) const {
  if (gradients_already_computed_ == false) {
    OL_THROW_EXCEPTION(OptimalLearningException, "PingExpectedImprovement::GetAnalyticGradient() called BEFORE EvaluateAndStoreAnalyticGradient. NO DATA!");
  }

  return grad_EI_[column_index*dim_ + row_index];
}

void PingCudaExpectedImprovement::EvaluateFunction(double const * restrict points_to_sample, double * restrict function_values) const noexcept {
  UniformRandomGenerator uniform_rng(3141);
  bool configure_for_gradients = false;
  CudaExpectedImprovementEvaluator::StateType ei_state(ei_evaluator_, points_to_sample, points_being_sampled_.data(), num_to_sample_, num_being_sampled_, configure_for_gradients, &uniform_rng);
  *function_values = ei_evaluator_.ComputeExpectedImprovement(&ei_state);
}
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

    CudaExpectedImprovementEvaluator cuda_ei_evaluator(gaussian_process, num_mc_iter, best_so_far);
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
int RunCudaEIvsCpuEI() {
  int total_errors = 0;

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

    CudaExpectedImprovementEvaluator cuda_ei_evaluator(gaussian_process, num_mc_iter, best_so_far);
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

/*!\rst
  At the moment, this test is very bare-bones.  It checks:

  1. method succeeds
  2. points returned are all inside the specified domain
  3. points returned are not within epsilon of each other (i.e., distinct)
  4. result of gradient-descent optimization is *no worse* than result of a random search
  5. final grad EI is sufficiently small

  The test sets up a toy problem by repeatedly drawing from a GP with made-up hyperparameters.
  Then it runs EI optimization, attempting to sample 3 points simultaneously.
\endrst*/
int CudaExpectedImprovementOptimizationMultipleSamplesTest() {
  using DomainType = TensorProductDomain;
  const int dim = 3;

  int total_errors = 0;
  int current_errors = 0;

  // gradient descent parameters
  const double gamma = 0.5;
  const double pre_mult = 1.5;
  const double max_relative_change = 1.0;
  const double tolerance = 1.0e-5;
  const int max_gradient_descent_steps = 250;
  const int max_num_restarts = 3;
  const int num_multistarts = 20;
  GradientDescentParameters gd_params(num_multistarts, max_gradient_descent_steps, max_num_restarts, gamma, pre_mult, max_relative_change, tolerance);

  // grid search parameters
  int num_grid_search_points = 1000;

  // q,p-EI computation parameters
  const int num_to_sample = 3;
  const int num_being_sampled = 0;

  std::vector<double> points_being_sampled(dim*num_being_sampled);
  int max_int_steps = 10000000;

  // random number generators
  UniformRandomGenerator uniform_generator(314);
  boost::uniform_real<double> uniform_double_hyperparameter(0.4, 1.3);
  boost::uniform_real<double> uniform_double_lower_bound(-2.0, 0.5);
  boost::uniform_real<double> uniform_double_upper_bound(1.0, 2.5);

  const int64_t pi_array[] = {314, 3141, 31415, 314159, 3141592, 31415926, 314159265, 3141592653, 31415926535, 314159265359};
  static const int kMaxNumThreads = 4;

  const int num_sampled = 20;
  std::vector<double> noise_variance(num_sampled, 0.002);
  MockGaussianProcessPriorData<DomainType> mock_gp_data(SquareExponential(dim, 1.0, 1.0), noise_variance, dim, num_sampled, uniform_double_lower_bound, uniform_double_upper_bound, uniform_double_hyperparameter, &uniform_generator);

  // we will optimize over the expanded region
  std::vector<ClosedInterval> domain_bounds(mock_gp_data.domain_bounds);
  ExpandDomainBounds(1.5, &domain_bounds);
  DomainType domain(domain_bounds.data(), dim);

  int which_gpu = 0;
  // optimize EI using grid search to set the baseline
  bool found_flag = false;
  std::vector<double> grid_search_best_point_set(dim*num_to_sample);
  CudaComputeOptimalPointsToSampleViaLatinHypercubeSearch(*mock_gp_data.gaussian_process_ptr, domain, points_being_sampled.data(), num_grid_search_points, num_to_sample, num_being_sampled, mock_gp_data.best_so_far, max_int_steps, which_gpu, kMaxNumThreads, &found_flag, &uniform_generator, grid_search_best_point_set.data());
  if (!found_flag) {
    ++total_errors;
  }

  // optimize EI using gradient descent
  found_flag = false;
  bool lhc_search_only = false;
  std::vector<double> best_points_to_sample(dim*num_to_sample);
  CudaComputeOptimalPointsToSample(*mock_gp_data.gaussian_process_ptr, gd_params, domain, points_being_sampled.data(), num_to_sample, num_being_sampled, mock_gp_data.best_so_far, max_int_steps, which_gpu, kMaxNumThreads, lhc_search_only, num_grid_search_points, &found_flag, &uniform_generator, best_points_to_sample.data());
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
    tolerance_result = 2.0e-3;  // reduce b/c we cannot achieve full accuracy in the monte-carlo case
    // while still having this test run in a reasonable amt of time
    bool configure_for_gradients = true;
    CudaExpectedImprovementEvaluator ei_evaluator(*mock_gp_data.gaussian_process_ptr, max_int_steps, mock_gp_data.best_so_far, which_gpu);
    CudaExpectedImprovementEvaluator::StateType ei_state(ei_evaluator, best_points_to_sample.data(), points_being_sampled.data(), num_to_sample, num_being_sampled, configure_for_gradients, &uniform_generator);

    ei_optimized = ei_evaluator.ComputeExpectedImprovement(&ei_state);
    ei_evaluator.ComputeGradExpectedImprovement(&ei_state, grad_ei.data());

    CudaExpectedImprovementEvaluator::StateType ei_state_grid_search(ei_evaluator, grid_search_best_point_set.data(), points_being_sampled.data(), num_to_sample, num_being_sampled, configure_for_gradients, &uniform_generator);
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

#else

int RunCudaEIConsistencyTests() {
  OL_WARNING_PRINTF("no gpu component is enabled, this test did not run.\n");
  return 0;
}

int RunCudaEIvsCpuEI() {
  OL_WARNING_PRINTF("no gpu component is enabled, this test did not run.\n");
  return 0;
}

int CudaExpectedImprovementOptimizationMultipleSamplesTest() {
  OL_WARNING_PRINTF("no gpu component is enabled, this test did not run.\n");
  return 0;
}
#endif

}  // end namespace optimal_learning
