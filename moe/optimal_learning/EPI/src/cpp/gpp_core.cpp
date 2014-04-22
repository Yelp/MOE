// gpp_core.cpp
/*
  File with main() used by eliu for testing/debugging C++ OL components.

  The code in this file is ad-hoc and purely used as a quick way to call and run the C++
  code from the command-line (i.e., independent of python).
*/

#include <sys/time.h>

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <functional>
#include <numeric>
#include <random>
#include <vector>

#include "gpp_common.hpp"
#include "gpp_covariance.hpp"
#include "gpp_covariance_test.hpp"
#include "gpp_domain.hpp"
#include "gpp_domain_test.hpp"
#include "gpp_exception.hpp"
#include "gpp_geometry_test.hpp"
#include "gpp_heuristic_expected_improvement_optimization.hpp"
#include "gpp_heuristic_expected_improvement_optimization_test.hpp"
#include "gpp_linear_algebra_test.hpp"
#include "gpp_logging.hpp"
#include "gpp_math.hpp"
#include "gpp_math_test.hpp"
#include "gpp_model_selection_and_hyperparameter_optimization.hpp"
#include "gpp_model_selection_and_hyperparameter_optimization_test.hpp"
#include "gpp_optimization.hpp"
#include "gpp_optimization_parameters.hpp"
#include "gpp_optimization_test.hpp"
#include "gpp_random.hpp"
#include "gpp_random_test.hpp"
#include "gpp_test_utils.hpp"
#include "gpp_test_utils_test.hpp"

using namespace optimal_learning;  // NOLINT, i'm lazy in this file which has no external linkage anyway

// -1: mirroring the "Within python" example currently at: https://github.com/sc932/MOE
// 0: random GP prior, optimize hyperparameter, build GP, optimize EI
// 1: runs Scott's Branin optimization
// 2: tests get_expected_EI, ComputeMeanOfPoints, and get_var_of_points on simple input
// 3: speed test EI eval
// 4: unit tests
// 5: speed test multistart newton hyper
// 6: speed test GD hyper
// 7: speed test multistart GD hyper
// 8: speed test log likelihood eval

#define OL_MODE -1
#if OL_MODE == -1

double function_to_minimize(double const * restrict point, UniformRandomGenerator * uniform_generator) {
  boost::uniform_real<double> uniform_double(-0.02, 0.02);
  return std::sin(point[0])*std::cos(point[1]) + std::cos(point[0] + point[1]) + uniform_double(uniform_generator->engine);
}

int main() {
  using DomainType = TensorProductDomain;
  using HyperparameterDomainType = TensorProductDomain;
  using CovarianceClass = SquareExponential;  // see gpp_covariance.hpp for other options

  int dim = 2;
  int num_to_sample = 0;
  int num_sampled = 21;

  std::vector<double> points_sampled(num_sampled*dim);
  std::vector<double> points_sampled_value(num_sampled);
  std::vector<double> noise_variance(num_sampled, 0.01);  // each entry must be >= 0.0
  std::vector<double> points_to_sample(num_to_sample*dim);  // each entry must be >= 0.0

  std::vector<ClosedInterval> domain_bounds = {
    {0.0, 2.0},
    {0.0,  4.0}};
  DomainType domain(domain_bounds.data(), dim);

  int num_mc_iterations=100000;

  UniformRandomGenerator uniform_generator(314);  // set to mode 0 to generate seeds automatically

  // gradient descent parameters
  const double gamma = 0.9;
  const double pre_mult = 1.0;
  const double max_relative_change = 1.0;
  const double tolerance = 1.0e-7;
  const int max_gradient_descent_steps = 1000;
  const int max_num_restarts = 3;
  const int num_multistarts = 40;
  GradientDescentParameters gd_params(num_multistarts, max_gradient_descent_steps, max_num_restarts, gamma, pre_mult, max_relative_change, tolerance);

  // booststrap w/some data
  points_sampled[0] = 0.0; points_sampled[1] = 0.0;
  points_sampled_value[0] = 1.0;

  double default_signal_variance = 1.0;
  double default_length_scale = 0.2;
  CovarianceClass covariance(dim, default_signal_variance, default_length_scale);

  // only 1 point sampled so far
  GaussianProcess gaussian_process(covariance, points_sampled.data(), points_sampled_value.data(), noise_variance.data(), dim, 0);

  // 20 more samples
  double best_so_far = *std::min_element(points_sampled_value.begin(), points_sampled_value.begin() + 0 + 1);
  int max_num_threads = 1;
  bool lhc_search_only = false;
  int num_lhc_samples = 0;
  int num_samples_to_generate = 1;
  bool found_flag = false;
  std::vector<double> best_points_to_sample(num_samples_to_generate*dim);
  for (int i = 1; i < num_sampled; ++i) {
    // std::vector<double> temp(dim, 0.0);
    // OnePotentialSampleExpectedImprovementEvaluator ei_evaluator(gaussian_process, best_so_far);
    // OnePotentialSampleExpectedImprovementState ei_state(ei_evaluator, temp.data(), 1, true, nullptr);
    // std::vector<double> grad_ei(dim);
    // double ei = ei_evaluator.ComputeExpectedImprovement(&ei_state);
    // printf("ei = %.18E\n", ei);
    // ei_evaluator.ComputeGradExpectedImprovement(&ei_state, grad_ei.data());
    // PrintMatrix(grad_ei.data(), 1, dim);

    ComputeOptimalSetOfPointsToSample(gaussian_process, gd_params, domain, points_to_sample.data(), num_to_sample, best_so_far, num_mc_iterations, max_num_threads, lhc_search_only, num_lhc_samples, num_samples_to_generate, &found_flag, &uniform_generator, nullptr, best_points_to_sample.data());
    printf("%d: found_flag = %d\n", i, found_flag);

    points_sampled_value[i] = function_to_minimize(best_points_to_sample.data(), &uniform_generator);
    // add function value back into the GP
    gaussian_process.AddPointToGP(best_points_to_sample.data(), points_sampled_value[i], noise_variance[i]);

    best_so_far = *std::min_element(points_sampled_value.begin(), points_sampled_value.begin() + i + 1);
  }

  PrintMatrix(best_points_to_sample.data(), 1, dim);

  return 0;
}

#endif

#if OL_MODE == 0

int main() {
  using DomainType = TensorProductDomain;
  using HyperparameterDomainType = TensorProductDomain;


  // the "spatial" dimension, aka the number of independent (experiment) parameters
  static const int dim = 3;  // > 0

  // number of concurrent samples running alongside the optimization
  static const int num_to_sample = 0;  // >= 0

  // number of points that we have already sampled; i.e., size of the training set
  static const int num_sampled = 10;  // >= 0

  // specifies the domain of each independent variable in (min, max) pairs
  // std::vector<double> domain_bounds = {0.1, 0.9,
  //                               0.2,  0.7,
  //                               0.05,  0.4};
  std::vector<ClosedInterval> domain_bounds = {
    {0.15, 0.4},
    {0.2,  0.35},
    {0.05,  0.4}};
  DomainType domain(domain_bounds.data(), dim);

  std::vector<double> points_sampled(num_sampled*dim);

  std::vector<double> points_sampled_value(num_sampled);

  // default to 0 noise
  std::vector<double> noise_variance(num_sampled, 0.0);  // each entry must be >= 0.0

  // covariance selection
  using CovarianceClass = SquareExponential;  // see gpp_covariance.hpp for other options

  UniformRandomGenerator uniform_generator(314);  // set to mode 0 to generate seeds automatically
  // arbitrary hyperparameters used to generate data
  std::vector<double> hyperparameters_original(1 + dim);
  // generate randomly
  boost::uniform_real<double> uniform_double_for_alpha(0.01, 0.08);
  boost::uniform_real<double> uniform_double_for_hyperparameter(0.05, 0.1);
  for (auto& hyperparameter : hyperparameters_original) {
    hyperparameter = uniform_double_for_hyperparameter(uniform_generator.engine);
  }
  // std::fill(hyperparameters_original.begin(), hyperparameters_original.end(), 0.03);
  hyperparameters_original[0] = uniform_double_for_alpha(uniform_generator.engine);

  CovarianceClass covariance_original(dim, hyperparameters_original[0], hyperparameters_original.data() + 1);
  int num_hyperparameters = covariance_original.GetNumberOfHyperparameters();

  PrintMatrix(hyperparameters_original.data(), 1.0, num_hyperparameters);

  // Generate data that will be used to build the GP
  // set noise
  boost::uniform_real<double> uniform_double_for_noise(0.001, 0.008);
  for (auto& noise : noise_variance) {
    noise = uniform_double_for_noise(uniform_generator.engine);
  }

  // use latin hypercube sampling to get a reasonable distribution of training point locations
  domain.GenerateUniformPointsInDomain(num_sampled, &uniform_generator, points_sampled.data());

  // build an empty GP: since num_sampled (last arg) is 0, none of the data arrays will be used here
  GaussianProcess gp_generator(covariance_original, points_sampled.data(), points_sampled_value.data(), noise_variance.data(), dim, 0);

  for (int j = 0; j < num_sampled; ++j) {
    // draw function value from the GP
    points_sampled_value.data()[j] = gp_generator.SamplePointFromGP(points_sampled.data() + dim*j, noise_variance.data()[j]);
    // add function value back into the GP
    gp_generator.AddPointToGP(points_sampled.data() + dim*j, points_sampled_value.data()[j], noise_variance.data()[j]);
  }

  // set up unbounded hyperparameter domain
  std::vector<ClosedInterval> hyperparameter_log_domain_bounds(num_hyperparameters);
  // for (int i = 0; i < num_hyperparameters; ++i) {
  //   hyperparameter_log_domain_bounds[2*i + 0] = -2.0;
  //   hyperparameter_log_domain_bounds[2*i + 1] = 0.0;
  // }
  // hyperparameter_log_domain_bounds[0] = -4.0; hyperparameter_log_domain_bounds[1] = 0.0;
  ClosedInterval derp[] = {
    {0.00059040912125259625, 1.0},
    {0.01, 0.25},
    {0.01, 0.14999999999999997},
    {0.01, 0.34048385431803763}};
  for (int i = 0; i < num_hyperparameters; ++i) {
    hyperparameter_log_domain_bounds[i].min = std::log10(derp[i].min);
    hyperparameter_log_domain_bounds[i].max = std::log10(derp[i].max);
  }
  // HyperparameterDomainType hyperparameter_domain(hyperparameter_log_domain_bounds.data(), num_hyperparameters);

  // Log Likelihood eval
  using LogLikelihoodEvaluator = LogMarginalLikelihoodEvaluator;
  // log likelihood evaluator object
  LogLikelihoodEvaluator log_marginal_eval(points_sampled.data(), points_sampled_value.data(), noise_variance.data(), dim, num_sampled);

  int newton_max_num_steps = 100;  // max number of newton steps
  double num_multistarts_newton = 100;
  double gamma_newton = 1.1;  // newton diagonal dominance scale-down factor (see newton docs for details)
  double pre_mult_newton = 1.0e-1;  // newton diagonal dominance scaling factor (see newton docs for details)
  double max_relative_change_newton = 1.0;
  double tolerance_newton = 1.0e-8;
  NewtonParameters newton_parameters(num_multistarts_newton, newton_max_num_steps, gamma_newton, pre_mult_newton, max_relative_change_newton, tolerance_newton);

#define OL_TEST_CONSTANT_LIAR
#ifdef OL_TEST_CONSTANT_LIAR
  {
    std::vector<double> new_newton_hyperparameters(num_hyperparameters);
    int max_num_threads = 4;
    bool found_flag = false;
    uniform_generator.SetExplicitSeed(314);
    MultistartNewtonHyperparameterOptimization(log_marginal_eval, covariance_original, newton_parameters, hyperparameter_log_domain_bounds.data(), max_num_threads, &found_flag, &uniform_generator, new_newton_hyperparameters.data());
    printf("newton found = %d\n", found_flag);

    PrintDomainBounds(hyperparameter_log_domain_bounds.data(), num_hyperparameters);
    PrintMatrix(new_newton_hyperparameters.data(), 1.0, num_hyperparameters);

    CovarianceClass covariance_final(dim, new_newton_hyperparameters[0], new_newton_hyperparameters.data() + 1);
    typename LogLikelihoodEvaluator::StateType log_marginal_state_newton_optimized_hyper(log_marginal_eval, covariance_final);
    double newton_log_marginal_opt = log_marginal_eval.ComputeLogLikelihood(log_marginal_state_newton_optimized_hyper);
    printf("newton optimized log marginal likelihood = %.18E\n", newton_log_marginal_opt);

    std::vector<double> grad_log_marginal_opt(num_hyperparameters);
    log_marginal_eval.ComputeGradLogLikelihood(&log_marginal_state_newton_optimized_hyper, grad_log_marginal_opt.data());
    printf("grad log likelihood: ");
    PrintMatrix(grad_log_marginal_opt.data(), 1, num_hyperparameters);

    // gradient descent parameters
    const double gamma = 0.9;
    const double pre_mult = 1.0;
    const double max_relative_change = 1.0;
    const double tolerance = 1.0e-7;
    const int max_gradient_descent_steps = 1000;
    const int max_num_restarts = 20;
    const int num_multistarts = 100;
    GradientDescentParameters gd_params(num_multistarts, max_gradient_descent_steps, max_num_restarts, gamma, pre_mult, max_relative_change, tolerance);

    double best_so_far = *std::min_element(points_sampled_value.begin(), points_sampled_value.end());
    // EI computation parameters
    double lie_value = best_so_far;  // Uses the "CL-min" version of constant liar
    double lie_noise_variance = 0.0;
    ConstantLiarEstimationPolicy constant_liar_policy(lie_value, lie_noise_variance);

    // number of simultaneous samples
    const int num_samples_to_generate = 3;
    std::vector<double> best_points_to_sample(dim*num_samples_to_generate);

    GaussianProcess gaussian_process(covariance_final, points_sampled.data(), points_sampled_value.data(), noise_variance.data(), dim, num_sampled);

    bool grid_search_only = false;
    int num_grid_search_points = 10000;
    found_flag = false;
    uniform_generator.SetExplicitSeed(31415);
    ComputeConstantLiarSetOfPointsToSample(gaussian_process, gd_params, domain, constant_liar_policy, best_so_far, max_num_threads, grid_search_only, num_grid_search_points, num_samples_to_generate, &found_flag, &uniform_generator, best_points_to_sample.data());
    PrintMatrixTrans(best_points_to_sample.data(), num_samples_to_generate, dim);

    printf("hi\n");
    // test Estimation Policies
    ConstantLiarEstimationPolicy constant_liar(2.0, 1.0);
    FunctionValue cl_value;
    cl_value = constant_liar.ComputeEstimate(gaussian_process, best_points_to_sample.data(), 0);
    printf("value: %.18E, var = %.18E\n", cl_value.function_value, cl_value.noise_variance);

    double std_deviation_coef = -0.2;
    KrigingBelieverEstimationPolicy kriging_believer(std_deviation_coef, 1.2);
    FunctionValue kb_value;
    kb_value = kriging_believer.ComputeEstimate(gaussian_process, best_points_to_sample.data(), 0);
    printf("value: %.18E, var = %.18E\n", kb_value.function_value, kb_value.noise_variance);

    {
      PointsToSampleState gaussian_process_state(gaussian_process, best_points_to_sample.data(), 1, false);

      double kriging_noise_variance = 1.2;
      double kriging_function_value;
      gaussian_process.ComputeMeanOfPoints(gaussian_process_state, &kriging_function_value);
      if (std_deviation_coef != 0.0) {
        // Only compute variance (expensive) if we are going to use it.
        double gp_variance;
        gaussian_process.ComputeVarianceOfPoints(&gaussian_process_state, &gp_variance);
        kriging_function_value += std_deviation_coef * std::sqrt(gp_variance);
      }
      printf("value: %.18E, var = %.18E\n", kriging_function_value, kriging_noise_variance);
    }

    {
      SquareExponential covariance(dim, kPi, 0.2);
      std::vector<ClosedInterval> domain_bounds(dim);
      for (auto& bounds : domain_bounds) {
        bounds = {0.0, 1.0};
      }
      DomainType domain_gp_source(domain_bounds.data(), dim);

      int num_sampled = 20;  // need to keep this similar to the number of multistarts
      std::vector<double> points_sampled(num_sampled*dim);
      std::vector<double> points_sampled_value(num_sampled, 1.0);
      std::vector<double> noise_variance(num_sampled, 0.0);

      // generate random sampling points
      domain_gp_source.GenerateUniformPointsInDomain(num_sampled, &uniform_generator, points_sampled.data());

      GaussianProcess gaussian_process(covariance, points_sampled.data(), points_sampled_value.data(), noise_variance.data(), dim, num_sampled);

      {
        PointsToSampleState gaussian_process_state(gaussian_process, points_sampled.data(), 1, false);
        double mean;
        double variance;
        gaussian_process.ComputeMeanOfPoints(gaussian_process_state, &mean);
        gaussian_process.ComputeVarianceOfPoints(&gaussian_process_state, &variance);
        printf("mean = %.18E, mean-1 = %.18E, variance = %.18E, variance-pi = %.18E\n", mean, mean - 1.0, variance, variance - kPi);
      }

      {
        std::vector<double> point(dim, 0.5);
        PointsToSampleState gaussian_process_state(gaussian_process, point.data(), 1, false);
        double mean;
        double variance;
        gaussian_process.ComputeMeanOfPoints(gaussian_process_state, &mean);
        gaussian_process.ComputeVarianceOfPoints(&gaussian_process_state, &variance);
        printf("mean = %.18E, mean-1 = %.18E, variance = %.18E, variance-pi = %.18E\n", mean, mean - 1.0, variance, variance - kPi);
      }

      {
        std::vector<double> point(dim, 3);
        PointsToSampleState gaussian_process_state(gaussian_process, point.data(), 1, false);
        double mean;
        double variance;
        gaussian_process.ComputeMeanOfPoints(gaussian_process_state, &mean);
        gaussian_process.ComputeVarianceOfPoints(&gaussian_process_state, &variance);
        printf("mean = %.18E, mean-1 = %.18E, variance = %.18E, variance-pi = %.18E\n", mean, mean - 1.0, variance, variance - kPi);
      }
    }
  }
#else
  {
    printf("TENSOR PRODUCT:\n");
    std::vector<double> new_newton_hyperparameters(num_hyperparameters);
    int max_num_threads = 4;
    bool found_flag = false;
    uniform_generator.SetExplicitSeed(314);
    MultistartNewtonHyperparameterOptimization(log_marginal_eval, covariance_original, newton_parameters, hyperparameter_log_domain_bounds.data(), max_num_threads, &found_flag, &uniform_generator, new_newton_hyperparameters.data());
    printf("newton found = %d\n", found_flag);

    PrintDomainBounds(hyperparameter_log_domain_bounds.data(), num_hyperparameters);
    PrintMatrix(new_newton_hyperparameters.data(), 1.0, num_hyperparameters);

    CovarianceClass covariance_final(dim, new_newton_hyperparameters[0], new_newton_hyperparameters.data() + 1);
    typename LogLikelihoodEvaluator::StateType log_marginal_state_newton_optimized_hyper(log_marginal_eval, covariance_final);
    double newton_log_marginal_opt = log_marginal_eval.ComputeLogLikelihood(log_marginal_state_newton_optimized_hyper);
    printf("newton optimized log marginal likelihood = %.18E\n", newton_log_marginal_opt);

    std::vector<double> grad_log_marginal_opt(num_hyperparameters);
    log_marginal_eval.ComputeGradLogLikelihood(&log_marginal_state_newton_optimized_hyper, grad_log_marginal_opt.data());
    printf("grad log likelihood: ");
    PrintMatrix(grad_log_marginal_opt.data(), 1, num_hyperparameters);

    // gradient descent parameters
    const double gamma = 0.9;
    const double pre_mult = 1.0;
    const double max_relative_change = 1.0;
    const double tolerance = 1.0e-7;
    const int max_gradient_descent_steps = 1000;
    const int max_num_restarts = 20;
    const int num_multistarts = 100;
    GradientDescentParameters gd_params(num_multistarts, max_gradient_descent_steps, max_num_restarts, gamma, pre_mult, max_relative_change, tolerance);

    double best_so_far = *std::min_element(points_sampled_value.begin(), points_sampled_value.end());
    found_flag = false;
    int max_int_steps = 1000;
    std::vector<double> next_point(dim);
    std::vector<double> points_to_sample;

    GaussianProcess gaussian_process(covariance_final, points_sampled.data(), points_sampled_value.data(), noise_variance.data(), dim, num_sampled);

    // uniform_generator.SetExplicitSeed(314);
    ClosedInterval derp2[] = {
      {0.15, 0.4},
      {0.2, 0.35},
      {0.05951614568196238, 0.4}};
    TensorProductDomain ei_domain(derp2, dim);
    ComputeOptimalPointToSampleWithRandomStarts(gaussian_process, gd_params, ei_domain, points_to_sample.data(), num_to_sample, best_so_far, max_int_steps, max_num_threads, &found_flag, &uniform_generator, nullptr, next_point.data());
    printf("EI found: %d\n", found_flag);
    printf("next best point  : "); PrintMatrix(next_point.data(), 1, dim);

    double ei_optimized, ei_grid_search;
    std::vector<double> grad_ei(dim);

    // set up evaluators and state to check results
    std::vector<double> union_of_points((num_to_sample+1)*dim);
    std::copy(next_point.begin(), next_point.end(), union_of_points.begin());
    std::copy(points_to_sample.begin(), points_to_sample.end(), union_of_points.begin() + dim);

    double tolerance_result = tolerance;
    if (1) {
      OnePotentialSampleExpectedImprovementEvaluator ei_evaluator(gaussian_process, best_so_far);
      OnePotentialSampleExpectedImprovementEvaluator::StateType ei_state(ei_evaluator, union_of_points.data(), num_to_sample + 1, true, nullptr);

      ei_optimized = ei_evaluator.ComputeExpectedImprovement(&ei_state);
      ei_evaluator.ComputeGradExpectedImprovement(&ei_state, grad_ei.data());
    }

    printf("optimized EI: %.18E\n", ei_optimized);
    printf("grad_EI: "); PrintMatrix(grad_ei.data(), 1, dim);
  }


  {
    printf("SIMPLEX:\n");
    std::vector<double> new_newton_hyperparameters(num_hyperparameters);
    int max_num_threads = 4;
    bool found_flag = false;
    uniform_generator.SetExplicitSeed(3141);
    MultistartNewtonHyperparameterOptimization(log_marginal_eval, covariance_original, newton_parameters, hyperparameter_log_domain_bounds.data(), max_num_threads, &found_flag, &uniform_generator, new_newton_hyperparameters.data());
    printf("newton found = %d\n", found_flag);

    PrintDomainBounds(hyperparameter_log_domain_bounds.data(), num_hyperparameters);
    PrintMatrix(new_newton_hyperparameters.data(), 1.0, num_hyperparameters);

    CovarianceClass covariance_final(dim, new_newton_hyperparameters[0], new_newton_hyperparameters.data() + 1);
    typename LogLikelihoodEvaluator::StateType log_marginal_state_newton_optimized_hyper(log_marginal_eval, covariance_final);
    double newton_log_marginal_opt = log_marginal_eval.ComputeLogLikelihood(log_marginal_state_newton_optimized_hyper);
    printf("newton optimized log marginal likelihood = %.18E\n", newton_log_marginal_opt);

    std::vector<double> grad_log_marginal_opt(num_hyperparameters);
    log_marginal_eval.ComputeGradLogLikelihood(&log_marginal_state_newton_optimized_hyper, grad_log_marginal_opt.data());
    printf("grad log likelihood: ");
    PrintMatrix(grad_log_marginal_opt.data(), 1, num_hyperparameters);

    // gradient descent parameters
    const double gamma = 0.9;
    const double pre_mult = 1.0;
    const double max_relative_change = 1.0;
    const double tolerance = 1.0e-7;
    const int max_gradient_descent_steps = 1000;
    const int max_num_restarts = 20;
    const int num_multistarts = 100;
    GradientDescentParameters gd_params(num_multistarts, max_gradient_descent_steps, max_num_restarts, gamma, pre_mult, max_relative_change, tolerance);

    double best_so_far = *std::min_element(points_sampled_value.begin(), points_sampled_value.end());
    found_flag = false;
    int max_int_steps = 1000;
    std::vector<double> next_point(dim);
    std::vector<double> points_to_sample;

    GaussianProcess gaussian_process(covariance_final, points_sampled.data(), points_sampled_value.data(), noise_variance.data(), dim, num_sampled);

    // uniform_generator.SetExplicitSeed(314);
    ClosedInterval derp2[] = {
      {0.15, 0.4},
      {0.2, 0.35},
      {0.05951614568196238, 0.4}};
    SimplexIntersectTensorProductDomain ei_domain(derp2, dim);
    ComputeOptimalPointToSampleWithRandomStarts(gaussian_process, gd_params, ei_domain, points_to_sample.data(), num_to_sample, best_so_far, max_int_steps, max_num_threads, &found_flag, &uniform_generator, nullptr, next_point.data());
    printf("EI found: %d\n", found_flag);
    printf("next best point  : "); PrintMatrix(next_point.data(), 1, dim);

    double ei_optimized, ei_grid_search;
    std::vector<double> grad_ei(dim);

    // set up evaluators and state to check results
    std::vector<double> union_of_points((num_to_sample+1)*dim);
    std::copy(next_point.begin(), next_point.end(), union_of_points.begin());
    std::copy(points_to_sample.begin(), points_to_sample.end(), union_of_points.begin() + dim);

    double tolerance_result = tolerance;
    if (1) {
      OnePotentialSampleExpectedImprovementEvaluator ei_evaluator(gaussian_process, best_so_far);
      OnePotentialSampleExpectedImprovementEvaluator::StateType ei_state(ei_evaluator, union_of_points.data(), num_to_sample + 1, true, nullptr);

      ei_optimized = ei_evaluator.ComputeExpectedImprovement(&ei_state);
      ei_evaluator.ComputeGradExpectedImprovement(&ei_state, grad_ei.data());
    }

    printf("optimized EI: %.18E\n", ei_optimized);
    printf("grad_EI: "); PrintMatrix(grad_ei.data(), 1, dim);
  }
#endif
  // double derp2[] = {0.15, 0.4, 0.2, 0.35, 0.05951614568196238, 0.4};
  // SimplexIntersectTensorProductDomain simplex_domain(derp2, dim);

  printf("domain = [");
  for (int i = 0; i < dim; ++i) {
    printf("[%.18E, %.18E, ], ", domain_bounds[i].min, domain_bounds[i].max);
  }
  printf("]\n");

  printf("points_sampled = [");
  for (int i = 0; i < num_sampled; ++i) {
    printf("[");
    for (int j = 0; j < dim; ++j) {
      printf("%.18E, ", points_sampled[i*dim + j]);
    }
    printf("], ");
  }
  printf("]\n");

  printf("points_sampled_value = [");
  for (int i = 0; i < num_sampled; ++i) {
    printf("%.18E, ", points_sampled_value[i]);
  }
  printf("]\n");

  printf("noise_variance = [");
  for (int i = 0; i < num_sampled; ++i) {
    printf("%.18E, ", noise_variance[i]);
  }
  printf("]\n");
}

#elif OL_MODE == 1


void rotate_point_list(double * restrict points_to_sample, int list_len, int dim) {
  for (int i = 1; i < list_len; i++) {
    for (int j = 0; j < dim; j++) {
      points_to_sample[(i-1)*dim + j] = points_to_sample[i*dim + j];
    }
  }
}

double branin_func(double *x) {
  if (x[0] > 10.0 || x[0] < -5.0 || x[1] < 0.0 || x[1] > 15.0) {
    printf("Branin function outside domain at (%.18E,%.18E)\n", x[0], x[1]);
  }
  return pow(x[1] - (5.1/(4.0*kPi*kPi))*pow(x[0], 2.0) + (5.0/kPi)*x[0] - 6.0, 2.0) + 10.0*(1.0 - 1.0/(8.0*kPi))*cos(x[0]) + 10.0;
}

// OL_FUNC_MODE 0: branin
// OL_FUNC_MODE 1: GPP
#define OL_FUNC_MODE 1

void run_core_test(int *processor_count_list, int num_processor_count_list, int num_samples) {
  using DomainType = TensorProductDomain;
  using HyperparameterDomainType = TensorProductDomain;
  int procs;
  int dim = 2;

  std::vector<ClosedInterval> domain_bounds = {
    {-5.0, 10.0},
    {0.0, 15.0}};
  DomainType domain(domain_bounds.data(), dim);

  // sample stencil of 9 points
  int stencil_rows = 3;
  int stencil_columns = 3;
  std::vector<double> points_sampled((stencil_rows*stencil_columns + num_samples)*dim);
  std::vector<double> points_sampled_value(stencil_rows*stencil_columns + num_samples);

  int num_total_samples = stencil_rows*stencil_columns + num_samples;
  std::vector<double> noise_variance(num_total_samples);
  for (int i = 0; i < num_total_samples; ++i) {
    noise_variance[i] = 1.0e-1;
  }

  int max_procs = processor_count_list[0];
  for (int i = 1; i < num_processor_count_list; i++) {
    if (processor_count_list[i] > max_procs) {
      max_procs = processor_count_list[i];
    }
  }

  std::vector<double> points_to_sample(max_procs*dim);

//   SquareExponentialSingleLength covariance(dim, 1.0, 2.0);
//   SquareExponentialSingleLength covariance_perturbed(dim, 1.1, 2.1);

  SquareExponential covariance(dim, 1.0, 2.0);
  SquareExponential covariance_perturbed(dim, 2.5, 4.9);
  // SquareExponential covariance_perturbed(dim, 12.5, 14.9);
  // SquareExponential covariance_perturbed(dim, 25.5, 45.9);

//   MaternNu1p5 covariance(dim, 1.0, 2.0);
//   MaternNu1p5 covariance_perturbed(dim, 2.5, 2.9);

  // MaternNu2p5 covariance(dim, 1.0, 2.0);
  // MaternNu2p5 covariance_perturbed(dim, 2.5, 4.9);

  UniformRandomGenerator uniform_generator(314);

#if 0
  for (int i = 0; i < stencil_rows; i++) {
    for (int j = 0; j < stencil_columns; j++) {
      points_sampled[(i*stencil_columns + j)*dim + 0] = 7.5*static_cast<double>(i) - 5.0;
      points_sampled[(i*stencil_columns + j)*dim + 1] = 7.5*static_cast<double>(j);
    }
  }
#else
  domain.GenerateUniformPointsInDomain(stencil_rows*stencil_columns, &uniform_generator, points_sampled.data());
#endif

  GaussianProcess gp(covariance, points_sampled.data(), points_sampled_value.data(), noise_variance.data(), dim, 0);

#if OL_FUNC_MODE == 0
  for (int j = 0; j < stencil_rows*stencil_columns; ++j) {
    points_sampled_value[j] = branin_func(points_sampled.data() + (j)*dim);
    gp.AddPointToGP(points_sampled.data() + dim*(j), points_sampled_value.data()[j], noise_variance.data()[j]);
  }
#elif OL_FUNC_MODE == 1
  for (int j = 0; j < stencil_rows*stencil_columns; ++j) {
    points_sampled_value.data()[j] = gp.SamplePointFromGP(points_sampled.data() + dim*(j), noise_variance.data()[j]);
    gp.AddPointToGP(points_sampled.data() + dim*(j), points_sampled_value.data()[j], noise_variance.data()[j]);
  }
#else
  exit(-1);
#endif

  std::vector<ClosedInterval> hyperparameter_domain_bounds(covariance_perturbed.GetNumberOfHyperparameters(), {1.0e-10, 1.0e10});
  HyperparameterDomainType hyperparameter_domain(hyperparameter_domain_bounds.data(), covariance_perturbed.GetNumberOfHyperparameters());

  std::vector<double> new_hyperparameters(covariance.GetNumberOfHyperparameters());
  const int hyperparameter_max_num_steps = 1000;
  const double gamma_hyper = 0.5;
  const double pre_mult_hyper = 1.0;
  const int max_num_restarts = 10;
  const double max_relative_change = 0.02;
  const double tolerance_gd = 1.0e-7;
  GradientDescentParameters gd_parameters(1, hyperparameter_max_num_steps, max_num_restarts, gamma_hyper, pre_mult_hyper, max_relative_change, tolerance_gd);

  printf("LOG MARGINAL HYPERPARAM OPT:\n");
  LogMarginalLikelihoodEvaluator log_marginal_eval(points_sampled.data(), points_sampled_value.data(), noise_variance.data(), dim, stencil_rows*stencil_columns);
  RestartedGradientDescentHyperparameterOptimization(log_marginal_eval, covariance_perturbed, gd_parameters, hyperparameter_domain, new_hyperparameters.data());
  // covariance.GetHyperparameters(new_hyperparameters.data());

  LogMarginalLikelihoodState log_marginal_state_initial_hyper(log_marginal_eval, covariance_perturbed);
  double log_marginal = log_marginal_eval.ComputeLogLikelihood(log_marginal_state_initial_hyper);
  printf("perturbed log marginal likelihood = %.18E\n", log_marginal);

  LogMarginalLikelihoodState log_marginal_state_generation_hyper(log_marginal_eval, covariance);
  log_marginal = log_marginal_eval.ComputeLogLikelihood(log_marginal_state_generation_hyper);
  printf("generation log marginal likelihood = %.18E\n", log_marginal);

  covariance.SetHyperparameters(new_hyperparameters.data());
  LogMarginalLikelihoodState log_marginal_state_log_marginal_optimized_hyper(log_marginal_eval, covariance);
  // getchar();

  double log_marginal_opt = log_marginal_eval.ComputeLogLikelihood(log_marginal_state_log_marginal_optimized_hyper);
  printf("optimized log marginal likelihood = %.18E\n", log_marginal_opt);

  std::vector<double> grad_log_marginal(covariance.GetNumberOfHyperparameters());
  std::vector<double> grad_log_marginal_opt(covariance.GetNumberOfHyperparameters());
  log_marginal_eval.ComputeGradLogLikelihood(&log_marginal_state_log_marginal_optimized_hyper, grad_log_marginal_opt.data());

  std::vector<double> grad_loo_likelihood_marginal_opt(covariance.GetNumberOfHyperparameters());
  LeaveOneOutLogLikelihoodEvaluator loo_marginal_eval(points_sampled.data(), points_sampled_value.data(), noise_variance.data(), dim, stencil_rows*stencil_columns);
  LeaveOneOutLogLikelihoodState loo_marginal_state_log_marginal_optimized_hyper(loo_marginal_eval, covariance);

  loo_marginal_eval.ComputeGradLogLikelihood(&loo_marginal_state_log_marginal_optimized_hyper, grad_loo_likelihood_marginal_opt.data());

  LeaveOneOutLogLikelihoodState loo_marginal_state_perturbed_hyper(loo_marginal_eval, covariance_perturbed);
  double loo_likelihood_perturbed = loo_marginal_eval.ComputeLogLikelihood(loo_marginal_state_perturbed_hyper);
  printf("perturbed loo: %.18E\n", loo_likelihood_perturbed);

  double loo_likelihood_marginal_opt = loo_marginal_eval.ComputeLogLikelihood(loo_marginal_state_log_marginal_optimized_hyper);
  printf("marginal optimized loo: %.18E\n", loo_likelihood_marginal_opt);

  std::vector<double> loo_new_hyperparameters(covariance.GetNumberOfHyperparameters());
  printf("LEAVE ONE OUT HYPERPARAM OPT:\n");
  RestartedGradientDescentHyperparameterOptimization(loo_marginal_eval, covariance_perturbed, gd_parameters, hyperparameter_domain, loo_new_hyperparameters.data());
  covariance.SetHyperparameters(loo_new_hyperparameters.data());

  LeaveOneOutLogLikelihoodState loo_marginal_state_loo_optimized_hyper(loo_marginal_eval, covariance);
  double loo_likelihood_loo_opt = loo_marginal_eval.ComputeLogLikelihood(loo_marginal_state_loo_optimized_hyper);

  LogMarginalLikelihoodState log_marginal_state_loo_optimized_hyper(log_marginal_eval, covariance);
  double log_marginal_loo_opt = log_marginal_eval.ComputeLogLikelihood(log_marginal_state_loo_optimized_hyper);

  std::vector<double> grad_log_marginal_loo_opt(covariance.GetNumberOfHyperparameters());
  log_marginal_eval.ComputeGradLogLikelihood(&log_marginal_state_loo_optimized_hyper, grad_log_marginal_loo_opt.data());

  std::vector<double> grad_loo_likelihood_loo_opt(covariance.GetNumberOfHyperparameters());
  loo_marginal_eval.ComputeGradLogLikelihood(&loo_marginal_state_loo_optimized_hyper, grad_loo_likelihood_loo_opt.data());

  printf("log likelihood hyper: "); PrintMatrix(new_hyperparameters.data(), 1, covariance.GetNumberOfHyperparameters());
  printf("loo likelihood hyper: "); PrintMatrix(loo_new_hyperparameters.data(), 1, covariance.GetNumberOfHyperparameters());
  printf("log likelihood: initial: %.18E, likelihood opt: %.18E, loo opt: %.18E\n", log_marginal, log_marginal_opt, log_marginal_loo_opt);
  printf("grad log likelihood: likelihood opt: "); PrintMatrix(grad_log_marginal_opt.data(), 1, covariance.GetNumberOfHyperparameters());
  printf("grad loo likelihood: likelihood opt: "); PrintMatrix(grad_loo_likelihood_marginal_opt.data(), 1, covariance.GetNumberOfHyperparameters());
  printf("loo likelihood: initial: %.18E, likelihood opt: %.18E, loo opt: %.18E\n", loo_likelihood_perturbed, loo_likelihood_marginal_opt, loo_likelihood_loo_opt);
  printf("grad log likelihood: loo opt: "); PrintMatrix(grad_log_marginal_loo_opt.data(), 1, covariance.GetNumberOfHyperparameters());
  printf("grad loo likelihood: loo opt: "); PrintMatrix(grad_loo_likelihood_loo_opt.data(), 1, covariance.GetNumberOfHyperparameters());

  covariance.SetHyperparameters(new_hyperparameters.data());
  // getchar();

  // printf(OL_ANSI_COLOR_GREEN "SUCCESS: " OL_ANSI_COLOR_RESET "Tests passed\n");
  // printf(OL_ANSI_COLOR_RED "FAILURE: " OL_ANSI_COLOR_RESET "Tests failed\n");
  // printf(OL_ANSI_COLOR_BLACK "FAILURE: " OL_ANSI_COLOR_RESET "Tests failed\n");
  // printf(OL_ANSI_COLOR_YELLOW "FAILURE: " OL_ANSI_COLOR_RESET "Tests failed\n");
  // printf(OL_ANSI_COLOR_BLUE "FAILURE: " OL_ANSI_COLOR_RESET "Tests failed\n");
  // printf(OL_ANSI_COLOR_MAGENTA "FAILURE: " OL_ANSI_COLOR_RESET "Tests failed\n");
  // printf(OL_ANSI_COLOR_CYAN "FAILURE: " OL_ANSI_COLOR_RESET "Tests failed\n");
  // printf(OL_ANSI_COLOR_WHITE "FAILURE: " OL_ANSI_COLOR_RESET "Tests failed\n");

  // printf(OL_ANSI_COLOR_BOLDGREEN "SUCCESS: " OL_ANSI_COLOR_RESET "Tests passed\n");
  // printf(OL_ANSI_COLOR_BOLDRED "FAILURE: " OL_ANSI_COLOR_RESET "Tests failed\n");
  // printf(OL_ANSI_COLOR_BOLDBLACK "FAILURE: " OL_ANSI_COLOR_RESET "Tests failed\n");
  // printf(OL_ANSI_COLOR_BOLDYELLOW "FAILURE: " OL_ANSI_COLOR_RESET "Tests failed\n");
  // printf(OL_ANSI_COLOR_BOLDBLUE "FAILURE: " OL_ANSI_COLOR_RESET "Tests failed\n");
  // printf(OL_ANSI_COLOR_BOLDMAGENTA "FAILURE: " OL_ANSI_COLOR_RESET "Tests failed\n");
  // printf(OL_ANSI_COLOR_BOLDCYAN "FAILURE: " OL_ANSI_COLOR_RESET "Tests failed\n");
  // printf(OL_ANSI_COLOR_BOLDWHITE "FAILURE: " OL_ANSI_COLOR_RESET "Tests failed\n");

  // exit(-1);

  double best_so_far = 10.3079084864;
  const double gamma = 0.1;
  const double pre_mult = 1.0;
  const double max_relative_change_ei = 1.0;
  const double tolerance_ei = 1.0e-9;

  const int num_multistarts = 5;
  const int max_num_steps = 200;
  const int max_int_steps = 100;
  const int max_num_restarts_ei = 5;
  int num_sampled = stencil_rows*stencil_columns;
  GradientDescentParameters gd_params_ei(num_multistarts, max_num_steps, max_num_restarts_ei, gamma, pre_mult, max_relative_change_ei, tolerance_ei);
  time_t time0;

  gp.SetCovarianceHyperparameters(new_hyperparameters.data());

  int64_t pi_array[] = {314, 3141, 31415, 314159, 3141592, 31415926, 314159265, 3141592653, 31415926535, 314159265359};
  int max_num_threads = 1;
  std::vector<NormalRNG> normal_rng_vec(max_num_threads);
  for (int i = 0; i < max_num_threads; ++i) {
    normal_rng_vec[i].SetExplicitSeed(pi_array[i]);
  }

  bool found_flag = false;
  for (int i = 0; i < num_processor_count_list; i++) {
    procs = processor_count_list[i];

    // fill up queue
    for (int j = 0; j < procs; j++) {
      ComputeOptimalPointToSampleWithRandomStarts(gp, gd_params_ei, domain, points_to_sample.data(), j, best_so_far, max_int_steps, max_num_threads, &found_flag, &uniform_generator, normal_rng_vec.data(), points_to_sample.data() + j*dim);
      printf("points_to_sample so far\n");
      PrintMatrixTrans(points_to_sample.data(), j+1, dim);
    }
    printf("Queue filled.\n");

    for (int j = 0; j < num_samples; j++) {
      time0 = time(nullptr);
      // sample the first point
      printf("attempting to shift to sampled\n");
      for (int k = 0; k < dim; k++) {
        points_sampled[(stencil_rows*stencil_columns+j)*dim + k] = points_to_sample[0*dim + k];
      }
      printf("atempting to acertain value\n");
#if OL_FUNC_MODE == 0
      points_sampled_value[stencil_rows*stencil_columns+j] = branin_func(points_sampled.data() + (stencil_rows*stencil_columns+j)*dim);
#elif OL_FUNC_MODE == 1
      points_sampled_value[stencil_rows*stencil_columns + j] = gp.SamplePointFromGP(points_sampled.data() + dim*(stencil_rows*stencil_columns + j), noise_variance[stencil_rows*stencil_columns + j]);
      gp.AddPointToGP(points_sampled.data() + dim*(stencil_rows*stencil_columns + j), points_sampled_value[stencil_rows*stencil_columns + j], noise_variance[stencil_rows*stencil_columns + j]);
#endif

      printf("checking against best_so_far\n");
      if (points_sampled_value[stencil_rows*stencil_columns+j] < best_so_far) {
        best_so_far = points_sampled_value[stencil_rows*stencil_columns+j];
      }
      num_sampled++;

      printf("sampled point val = %.18E, best_so_far = %.18E\n", points_sampled_value[stencil_rows*stencil_columns+j], best_so_far);
      // rotate points_to_sample
      printf("rotating points_to_sample\n");
      rotate_point_list(points_to_sample.data(), procs, dim);
      // pick a new point

      // TODO(sclark): tail off correctly so taht we dont sample the end points repeatedly
      if (num_sampled - stencil_rows*stencil_columns + procs - 1 < num_samples) {
        printf("picking a new point\n");
        ComputeOptimalPointToSampleWithRandomStarts(gp, gd_params_ei, domain, points_to_sample.data(), procs-1, best_so_far, max_int_steps, max_num_threads, &found_flag, &uniform_generator, normal_rng_vec.data(), points_to_sample.data() + (procs-1)*dim);
      }

      printf("points_to_sample so far\n");
      PrintMatrixTrans(points_to_sample.data(), procs, dim);
      printf("sample took %d seconds\n", static_cast<int>(time(nullptr) - time0));
    }
  }

  printf("num_sampled = %d, initial points = %d, num_samples = %d\n", num_sampled, stencil_rows*stencil_columns, num_samples);

  if (num_total_samples != num_sampled) {
    printf("ERROR: number of total samples, %d, != size of sampled, %d\n", num_total_samples, num_sampled);
    exit(-1);
  }

  LogMarginalLikelihoodEvaluator log_marginal_eval_final(points_sampled.data(), points_sampled_value.data(), noise_variance.data(), dim, num_sampled);
  LogMarginalLikelihoodState log_marginal_state_final_hyper(log_marginal_eval_final, covariance);
  log_marginal = log_marginal_eval_final.ComputeLogLikelihood(log_marginal_state_final_hyper);
  printf("log marginal likelihood = %.18E\n", log_marginal);

  log_marginal_eval_final.ComputeGradLogLikelihood(&log_marginal_state_final_hyper, grad_log_marginal.data());
  printf("grad log marginal likelihood:\n");
  PrintMatrix(grad_log_marginal.data(), 1, covariance.GetNumberOfHyperparameters());
}


int main() {
  std::vector<int> processor_count_list(1);
  processor_count_list[0] = 4;
  run_core_test(processor_count_list.data(), 1, 24);
  printf("Exited Successfully.\n");

  return 0;
}

#elif OL_MODE == 2

int main() {
  // here we set some configurable parameters
  // feel free to change them (and recompile) as you explore
  // comments next to each parameter will indicate its purpose and domain

  // the "spatial" dimension, aka the number of independent (experiment) parameters
  static const int dim = 2;  // > 0

  // number of concurrent samples running alongside the optimization
  static const int num_to_sample = 0;  // >= 0

  // number of points that we have already sampled; i.e., size of the training set
  static const int num_sampled = 3;  // >= 0

  UniformRandomGenerator uniform_generator(0, 0);  // generate seeds automatically

  // specifies the domain of each independent variable in (min, max) pairs
  std::vector<ClosedInterval> domain_bounds(dim, {1.0, 5.0});
  TensorProductDomain domain(domain_bounds.data(), dim);

  // now we allocate point sets; ALL POINTS MUST LIE INSIDE THE DOMAIN!
  std::vector<double> points_to_sample(num_to_sample*dim);

  std::vector<double> points_sampled(dim*num_sampled);
  points_sampled[0] = 2.0; points_sampled[1] = 3.0; points_sampled[2] = 3.0; points_sampled[3] = 2.5; points_sampled[4] = 4.0; points_sampled[5] = 3.5;

  std::vector<double> points_sampled_value(num_sampled);
  points_sampled_value[0] = -0.971475067584; points_sampled_value[1] = -1.01058365802; points_sampled_value[2] = -0.975903614458;

  // default to 0 noise
  std::vector<double> noise_variance(num_sampled, 0.005);  // each entry must be >= 0.0

  // covariance selection
  using CovarianceClass = SquareExponential;  // see gpp_covariance.hpp for other options

  // arbitrary hyperparameters used to generate data
  std::vector<double> hyperparameters_original(1 + dim);
  // generate randomly
  boost::uniform_real<double> uniform_double_for_hyperparameter(0.5, 1.5);
  for (auto& hyperparameter : hyperparameters_original) {
    hyperparameter = uniform_double_for_hyperparameter(uniform_generator.engine);
  }

  CovarianceClass covariance_original(dim, hyperparameters_original[0], hyperparameters_original.data() + 1);

  // Log Likelihood eval
  using LogLikelihoodEvaluator = LogMarginalLikelihoodEvaluator;
  // log likelihood evaluator object
  LogLikelihoodEvaluator log_marginal_eval(points_sampled.data(), points_sampled_value.data(), noise_variance.data(), dim, num_sampled);

  // multithreading
  int max_num_threads = 4;  // feel free to experiment with different numbers

  // set up RNG containers
  std::vector<NormalRNG> normal_rng_vec(max_num_threads);
  for (int i = 0; i < max_num_threads; ++i) {
    normal_rng_vec[i].SetRandomizedSeed(0, i);  // automatic seed selection (based on current time)
  }

  // Newton setup
  int newton_max_num_steps = 100;  // max number of newton steps
  double gamma_newton = 1.1;  // newton diagonal dominance scale-down factor (see newton docs for details)
  double pre_mult_newton = 1.0e-1;  // newton diagonal dominance scaling factor (see newton docs for details)
  double max_relative_change_newton = 1.0;
  double tolerance_newton = 1.0e-13;
  int newton_num_multistarts = 100;

  NewtonParameters newton_parameters(newton_num_multistarts, newton_max_num_steps, gamma_newton, pre_mult_newton, max_relative_change_newton, tolerance_newton);

  std::vector<ClosedInterval> hyperparameter_domain = {
    {-1.0, 1.0},
    {-2.0, -0.5},
    {-2.0, -0.5}};
  std::vector<double> new_newton_hyperparameters(covariance_original.GetNumberOfHyperparameters());

  for (int k = 0; k < 30; ++k) {
    bool hyperparameters_found = false;
    bool found_flag = false;
    while (hyperparameters_found == false) {
      MultistartNewtonHyperparameterOptimization(log_marginal_eval, covariance_original, newton_parameters, hyperparameter_domain.data(), max_num_threads, &found_flag, &uniform_generator, new_newton_hyperparameters.data());
      hyperparameters_found = true;
      for (const auto& entry : new_newton_hyperparameters) {
        if (entry > 5.0) {
          hyperparameters_found = false;
          break;
        }
      }
    }

    // Now optimize EI using the 'best' hyperparameters
    // set gaussian process's hyperparameters to the result of newton optimization
    CovarianceClass covariance_opt(dim, new_newton_hyperparameters[0], new_newton_hyperparameters.data() + 1);
    GaussianProcess gp_model(covariance_opt, points_sampled.data(), points_sampled_value.data(), noise_variance.data(), dim, num_sampled);


    double best_so_far = *std::min_element(points_sampled_value.begin(), points_sampled_value.end());  // this is simply the best function value seen to date

    // gradient descent parameters
    double gamma = 0.1;  // we decrease step size by a factor of 1/(iteration)^gamma
    double pre_mult = 1.0;  // scaling factor
    double max_relative_change_ei = 1.0;
    double tolerance_ei = 1.0e-7;
    int num_multistarts = 10;  // max number of multistarted locations
    int max_num_steps = 500;  // maximum number of GD iterations per restart
    int max_num_restarts = 20;  // number of restarts to run with GD
    GradientDescentParameters gd_params(num_multistarts, max_num_steps, max_num_restarts, gamma, pre_mult, max_relative_change_ei, tolerance_ei);

    // EI evaluation parameters
    int max_int_steps = 1000;  // number of monte carlo iterations

    // printf(OL_ANSI_COLOR_CYAN "OPTIMIZING EXPECTED IMPROVEMENT... (optimized hyperparameters)\n" OL_ANSI_COLOR_RESET);
    {
      std::vector<double> next_point_winner(dim);
      bool found_flag = false;
      ComputeOptimalPointToSampleWithRandomStarts(gp_model, gd_params, domain, points_to_sample.data(), num_to_sample, best_so_far, max_int_steps, max_num_threads, &found_flag, &uniform_generator, normal_rng_vec.data(), next_point_winner.data());
      // printf(OL_ANSI_COLOR_CYAN "EI OPTIMIZATION FINISHED (optimized hyperparameters).\n" OL_ANSI_COLOR_RESET);
      printf("Next best sample point according to EI (opt hyper):\n");
      PrintMatrix(next_point_winner.data(), 1, dim);
    }
  }
  return 0;
}

#elif OL_MODE == 3

int main() {
  using DomainType = TensorProductDomain;
  const int dim = 3;

  int total_errors = 0;
  int current_errors = 0;

  const double gamma = 0.9;
  const double pre_mult = 0.02;
  const double max_relative_change = 0.99;
  const double tolerance = 1.0e-7;
  const int max_gradient_descent_steps = 1000;
  const int max_num_restarts = 10;
  const int num_multistarts = 20;
  GradientDescentParameters gd_params(num_multistarts, max_gradient_descent_steps, max_num_restarts, gamma, pre_mult, max_relative_change, tolerance);

  // grid search parameters
  int num_grid_search_points = 100000;

  // EI computation parameters
  int num_to_sample = 0;
  int max_int_steps = 1000;

  // random number generators
  UniformRandomGenerator uniform_generator(314);
  boost::uniform_real<double> uniform_double_hyperparameter(0.4, 1.3);
  boost::uniform_real<double> uniform_double_lower_bound(-2.0, 0.5);
  boost::uniform_real<double> uniform_double_upper_bound(2.0, 3.5);

  const int64_t pi_array[] = {314, 3141, 31415, 314159, 3141592, 31415926, 314159265, 3141592653, 31415926535, 314159265359};
  static const int kMaxNumThreads = 4;
  std::vector<NormalRNG> normal_rng_vec(kMaxNumThreads);
  for (int j = 0; j < kMaxNumThreads; ++j) {
    normal_rng_vec[j].SetExplicitSeed(pi_array[j]);
  }

  SquareExponential covariance(dim, 1.0, 1.0);
  std::vector<double> hyperparameters(covariance.GetNumberOfHyperparameters());
  for (auto& entry : hyperparameters) {
    entry = uniform_double_hyperparameter(uniform_generator.engine);
  }
  covariance.SetHyperparameters(hyperparameters.data());

  std::vector<ClosedInterval> domain_bounds(dim);
  for (int i = 0; i < dim; ++i) {
    domain_bounds[i].min = uniform_double_lower_bound(uniform_generator.engine);
    domain_bounds[i].max = uniform_double_upper_bound(uniform_generator.engine);
  }
  DomainType domain_gp_source(domain_bounds.data(), dim);

  int num_sampled;
  int objective_mode = 1;
  if (objective_mode == 0) {
    num_grid_search_points *= 100;
    num_sampled = 20;  // need to keep this similar to the number of multistarts
  } else {
    num_sampled = 80;  // matters less here b/c we end up starting one multistart from the LHC-search optima
  }

  std::vector<double> points_sampled(num_sampled*dim);
  std::vector<double> points_sampled_value(num_sampled);
  std::vector<double> noise_variance(num_sampled, 0.002);

  GaussianProcess gaussian_process(covariance, points_sampled.data(), points_sampled_value.data(), noise_variance.data(), dim, 0);

  // generate random sampling points
  domain_gp_source.GenerateUniformPointsInDomain(num_sampled, &uniform_generator, points_sampled.data());

  // generate the "world"
  for (int j = 0; j < num_sampled; ++j) {
    points_sampled_value.data()[j] = gaussian_process.SamplePointFromGP(points_sampled.data() + dim*j, noise_variance[j]);
    gaussian_process.AddPointToGP(points_sampled.data() + dim*j, points_sampled_value[j], noise_variance[j]);
  }

  // get best point
  double best_so_far = *std::min_element(points_sampled_value.begin(), points_sampled_value.end());

  // expand the domain: we will optimize over the expanded region
  // we want to make sure that optima are not outside the domain since this result is much more
  // difficult to check reliably in testing
  // we also do not want to expand the domain *too* much or update limiting measures will play
  // no role (i.e., problem becomes too easy)
#ifdef OL_VERBOSE_PRINT
  PrintDomainBounds(domain_bounds.data(), dim);  // domain before expansion
#endif
  for (int i = 0; i < dim; ++i) {
    double side_length = domain_bounds[i].Length();
    double midpoint = 0.5*(domain_bounds[i].max + domain_bounds[i].min);
    side_length *= 2.2;
    side_length *= 0.5;
    domain_bounds[i].min = midpoint - side_length;
    domain_bounds[i].max = midpoint + side_length;
  }
#ifdef OL_VERBOSE_PRINT
  PrintDomainBounds(domain_bounds.data(), dim);  // expanded domain
#endif
  DomainType domain(domain_bounds.data(), dim);

  // set up parallel experiments, if any
  if (objective_mode == 0) {
    num_to_sample = 0;
  } else {
    // using MC integration
    num_to_sample = 2;
    max_int_steps = 1000;

    gd_params.max_num_steps = 200;
    gd_params.tolerance = 1.0e-5;
  }
  std::vector<double> points_to_sample(dim*num_to_sample);

  if (objective_mode == 1) {
    // generate two non-trivial parallel samples
    // picking these randomly could place them in regions where EI is 0, which means errors in the computation would
    // likely be masked (making for a bad test)
    bool found_flag = false;
    for (int j = 0; j < num_to_sample; j++) {
      ComputeOptimalPointToSampleWithRandomStarts(gaussian_process, gd_params, domain, points_to_sample.data(), j, best_so_far, max_int_steps, kMaxNumThreads, &found_flag, &uniform_generator, normal_rng_vec.data(), points_to_sample.data() + j*dim);
    }
    printf("setup complete, points_to_sample:\n");
    PrintMatrixTrans(points_to_sample.data(), num_to_sample, dim);
  }

  struct timeval tv0, tv1;

  gettimeofday(&tv0, nullptr);

  time_t c0, c1;
  c0 = clock();

  std::vector<double> next_point(dim);
  bool found_flag = false;
  std::vector<double> grid_search_best_point(dim);
  // ComputeOptimalPointToSampleViaLatinHypercubeSearch(gaussian_process, domain, points_to_sample.data(), num_grid_search_points, num_to_sample, best_so_far, max_int_steps, kMaxNumThreads, &found_flag, &uniform_generator, normal_rng_vec.data(), grid_search_best_point.data());

  std::vector<double> function_values(num_grid_search_points);
  std::vector<double> initial_guesses(dim*num_grid_search_points);
  num_grid_search_points = domain.GenerateUniformPointsInDomain(num_grid_search_points, &uniform_generator, initial_guesses.data());

  EvaluateEIAtPointList(gaussian_process, domain, initial_guesses.data(), points_to_sample.data(), num_grid_search_points, num_to_sample, best_so_far, max_int_steps, kMaxNumThreads, &found_flag, normal_rng_vec.data(), function_values.data(), grid_search_best_point.data());

  gettimeofday(&tv1, nullptr);
  c1 = clock();
  printf("time: %f\n", static_cast<double>((c1 - c0))/static_cast<double>(CLOCKS_PER_SEC));

  int diff = (tv1.tv_sec - tv0.tv_sec) * 1000000;
  diff += tv1.tv_usec - tv0.tv_usec;

  printf("diff = %d, %f\n", diff, static_cast<double>(diff)/1000000.0);

  PrintMatrix(grid_search_best_point.data(), 1, dim);

  // int max_index = 0;
  // double best_value = function_values[0];
  // for (int i = 0; i < num_grid_search_points; ++i) {
  //   if (function_values[i] > best_value) {
  //     best_value = function_values[i];
  //     max_index = i;
  //   }
  // }
  // printf("best = %.18E, index = %d\n", best_value, max_index);
  // PrintMatrix(initial_guesses.data() + max_index*dim, 1, dim);

  return 0;
}

#elif OL_MODE == 4

// 0: ping covariance
// 1: ping mu
// 2: ping var
// 3: ping cholesky
// 4: ping EI
// 5: ping "one point to sample" special case EI
// 6: compare MC EI to analytic EI
// 7: linalg tests
// 8: sampling
// 9: all tests
#define OL_PINGMODE 9

int main() {
  int total_errors = 0;
  int error = 0;

#if OL_PINGMODE == 0
  // error += RunCovarianceTests();
  // error += RunGPPingTests();
  // error += RunLogLikelihoodPingTests();
  // error += HyperparameterLikelihoodOptimizationTest(OptimizerTypes::kNewton, LogLikelihoodTypes::kLogMarginalLikelihood);
  // error += HyperparameterLikelihoodOptimizationTest(OptimizerTypes::kGradientDescent, LogLikelihoodTypes::kLogMarginalLikelihood);
  // error += HyperparameterLikelihoodOptimizationTest(OptimizerTypes::kGradientDescent, LogLikelihoodTypes::kLeaveOneOutLogLikelihood);
  // error += EvaluateLogLikelihoodAtPointListTest();
  // error = RandomNumberGeneratorContainerTest();
  // error += RunOptimizationTests(0);
  // error += RunOptimizationTests(1);
  // error += DomainTests();
  // error += RunEIConsistencyTests();
  // error += MultithreadedEIOptimizationTest(ExpectedImprovementEvaluationMode::kAnalytic);
  // error += MultithreadedEIOptimizationTest(ExpectedImprovementEvaluationMode::kMonteCarlo);
  // error += ExpectedImprovementOptimizationTest(DomainTypes::kTensorProduct, ExpectedImprovementEvaluationMode::kAnalytic);
  // error += ExpectedImprovementOptimizationTest(DomainTypes::kTensorProduct, ExpectedImprovementEvaluationMode::kMonteCarlo);
  // error += ExpectedImprovementOptimizationMultipleSamplesTest();
  // error += ExpectedImprovementOptimizationTest(DomainTypes::kSimplex, ExpectedImprovementEvaluationMode::kAnalytic);
  // error += ExpectedImprovementOptimizationTest(DomainTypes::kSimplex, ExpectedImprovementEvaluationMode::kMonteCarlo);
  // error += EvaluateEIAtPointListTest();
  // error += EstimationPolicyTest();
  error += HeuristicExpectedImprovementOptimizationTest();

  if (error != 0) {
    OL_FAILURE_PRINTF("%d errors\n", error);
  } else {
    OL_SUCCESS_PRINTF("\n");
  }

#elif OL_PINGMODE == 1  // mu
  error = PingGPMeanTest();

  if (error != 0) {
    OL_FAILURE_PRINTF("%d errors\n", error);
  } else {
    OL_SUCCESS_PRINTF("\n");
  }

#elif OL_PINGMODE == 2  // var
  error = PingGPVarianceTest();

  if (error != 0) {
    OL_FAILURE_PRINTF("%d errors\n", error);
  } else {
    OL_SUCCESS_PRINTF("\n");
  }

#elif OL_PINGMODE == 3  // chol
  error = PingGPCholeskyVarianceTest();

  if (error != 0) {
    OL_FAILURE_PRINTF("%d errors\n", error);
  } else {
    OL_SUCCESS_PRINTF("\n");
  }

#elif OL_PINGMODE == 4  // EI
  error = PingEIGeneralTest();

  if (error != 0) {
    OL_FAILURE_PRINTF("\n");
  } else {
    OL_SUCCESS_PRINTF("\n");
  }

#elif OL_PINGMODE == 5  // special case one_to_sample_EI
  error = PingEIOnePotentialSampleTest();

  if (error != 0) {
    OL_FAILURE_PRINTF("\n");
  } else {
    OL_SUCCESS_PRINTF("\n");
  }

#elif OL_PINGMODE == 6  // special case "one to sample" EI
  error = RunEIConsistencyTests();

  if (error != 0) {
    OL_FAILURE_PRINTF("\n");
  } else {
    OL_SUCCESS_PRINTF("\n");
  }

#elif OL_PINGMODE == 7  // linalg tests
  error = RunLinearAlgebraTests();
  if (error != 0) {
    OL_FAILURE_PRINTF("\n");
  } else {
    OL_SUCCESS_PRINTF("\n");
  }

#elif OL_PINGMODE == 8  // sampling tests
  error = RunRandomPointGeneratorTests();
  if (error != 0) {
    OL_FAILURE_PRINTF("various random point sampling\n");
  } else {
    OL_SUCCESS_PRINTF("various random point sampling\n");
  }

#elif OL_PINGMODE == 9  // all
  error = TestUtilsTests();
  if (error != 0) {
    OL_FAILURE_PRINTF("test utils\n");
  } else {
    OL_SUCCESS_PRINTF("test utils\n");
  }
  total_errors += error;

  error = RunLinearAlgebraTests();
  if (error != 0) {
    OL_FAILURE_PRINTF("linear algebra\n");
  } else {
    OL_SUCCESS_PRINTF("linear algebra\n");
  }
  total_errors += error;

  error = RunCovarianceTests();
  if (error != 0) {
    OL_FAILURE_PRINTF("covariance\n");
  } else {
    OL_SUCCESS_PRINTF("covariance\n");
  }
  total_errors += error;

  error = RunGPPingTests();
  if (error != 0) {
    OL_FAILURE_PRINTF("GP ping\n");
  } else {
    OL_SUCCESS_PRINTF("GP ping\n");
  }
  total_errors += error;

  error = RunEIConsistencyTests();
  if (error != 0) {
    OL_FAILURE_PRINTF("EI consistency\n");
  } else {
    OL_SUCCESS_PRINTF("EI consistency\n");
  }
  total_errors += error;

  error = RunLogLikelihoodPingTests();
  if (error != 0) {
    OL_FAILURE_PRINTF("LogLikelihood ping\n");
  } else {
    OL_SUCCESS_PRINTF("LogLikelihood ping\n");
  }
  total_errors += error;

  error = RunRandomPointGeneratorTests();
  if (error != 0) {
    OL_FAILURE_PRINTF("various random point sampling\n");
  } else {
    OL_SUCCESS_PRINTF("various random point sampling\n");
  }
  total_errors += error;

  error = RandomNumberGeneratorContainerTest();
  if (error != 0) {
    OL_FAILURE_PRINTF("random number generator containers\n");
  } else {
    OL_SUCCESS_PRINTF("random number generator containers\n");
  }
  total_errors += error;

  error = DomainTests();
  if (error != 0) {
    OL_FAILURE_PRINTF("domain classes\n");
  } else {
    OL_SUCCESS_PRINTF("domain classes\n");
  }
  total_errors += error;

  error = ClosedIntervalTests();
  if (error != 0) {
    OL_FAILURE_PRINTF("ClosedInterval member functions\n");
  } else {
    OL_SUCCESS_PRINTF("ClosedInterval member functions\n");
  }
  total_errors += error;

  error = GeometryToolsTests();
  if (error != 0) {
    OL_FAILURE_PRINTF("geometry tools\n");
  } else {
    OL_SUCCESS_PRINTF("geometry tools\n");
  }
  total_errors += error;

  error += EstimationPolicyTest();
  if (error != 0) {
    OL_FAILURE_PRINTF("Estimation Policies\n");
  } else {
    OL_SUCCESS_PRINTF("Estimation Policies\n");
  }
  total_errors += error;

  error = RunOptimizationTests(OptimizerTypes::kGradientDescent);
  if (error != 0) {
    OL_FAILURE_PRINTF("quadratic mock gradient descent optimization\n");
  } else {
    OL_SUCCESS_PRINTF("quadratic mock gradient descent optimization\n");
  }
  total_errors += error;

  error = RunOptimizationTests(OptimizerTypes::kNewton);
  if (error != 0) {
    OL_FAILURE_PRINTF("quadratic mock newton optimization\n");
  } else {
    OL_SUCCESS_PRINTF("quadratic mock newton optimization\n");
  }
  total_errors += error;

  error = HyperparameterLikelihoodOptimizationTest(OptimizerTypes::kGradientDescent, LogLikelihoodTypes::kLogMarginalLikelihood);
  if (error != 0) {
    OL_FAILURE_PRINTF("log likelihood hyperparameter optimization\n");
  } else {
    OL_SUCCESS_PRINTF("log likelihood hyperparameter optimization\n");
  }
  total_errors += error;

  error = HyperparameterLikelihoodOptimizationTest(OptimizerTypes::kGradientDescent, LogLikelihoodTypes::kLeaveOneOutLogLikelihood);
  if (error != 0) {
    OL_FAILURE_PRINTF("LOO likelihood hyperparameter optimization\n");
  } else {
    OL_SUCCESS_PRINTF("LOO likelihood hyperparameter optimization\n");
  }
  total_errors += error;

  error = HyperparameterLikelihoodOptimizationTest(OptimizerTypes::kNewton, LogLikelihoodTypes::kLogMarginalLikelihood);
  if (error != 0) {
    OL_FAILURE_PRINTF("log likelihood hyperparameter newton optimization\n");
  } else {
    OL_SUCCESS_PRINTF("log likelihood hyperparameter newton optimization\n");
  }
  total_errors += error;

  error = EvaluateLogLikelihoodAtPointListTest();
  if (error != 0) {
    OL_FAILURE_PRINTF("log likelihood evaluation at point list\n");
  } else {
    OL_SUCCESS_PRINTF("log likelihood evaluation at point list\n");
  }
  total_errors += error;

  error = EvaluateEIAtPointListTest();
  if (error != 0) {
    OL_FAILURE_PRINTF("EI evaluation at point list\n");
  } else {
    OL_SUCCESS_PRINTF("EI evaluation at point list\n");
  }
  total_errors += error;

  error = MultithreadedEIOptimizationTest(ExpectedImprovementEvaluationMode::kAnalytic);
  if (error != 0) {
    OL_FAILURE_PRINTF("analytic EI Optimization single/multithreaded consistency check\n");
  } else {
    OL_SUCCESS_PRINTF("analytic EI single/multithreaded consistency check\n");
  }
  total_errors += error;

  error = MultithreadedEIOptimizationTest(ExpectedImprovementEvaluationMode::kMonteCarlo);
  if (error != 0) {
    OL_FAILURE_PRINTF("EI Optimization single/multithreaded consistency check\n");
  } else {
    OL_SUCCESS_PRINTF("EI single/multithreaded consistency check\n");
  }
  total_errors += error;

  error += HeuristicExpectedImprovementOptimizationTest();
  if (error != 0) {
    OL_FAILURE_PRINTF("Heuristic EI Optimization\n");
  } else {
    OL_SUCCESS_PRINTF("Heuristic EI Optimization\n");
  }
  total_errors += error;

  error = ExpectedImprovementOptimizationTest(DomainTypes::kTensorProduct, ExpectedImprovementEvaluationMode::kAnalytic);
  if (error != 0) {
    OL_FAILURE_PRINTF("analytic EI optimization\n");
  } else {
    OL_SUCCESS_PRINTF("analytic EI optimization\n");
  }
  total_errors += error;

  error = ExpectedImprovementOptimizationTest(DomainTypes::kTensorProduct, ExpectedImprovementEvaluationMode::kMonteCarlo);
  if (error != 0) {
    OL_FAILURE_PRINTF("monte-carlo EI optimization\n");
  } else {
    OL_SUCCESS_PRINTF("monte-carlo EI optimization\n");
  }
  total_errors += error;

  error = ExpectedImprovementOptimizationMultipleSamplesTest();
  if (error != 0) {
    OL_FAILURE_PRINTF("monte-carlo EI optimization for multiple simultaneous experiments\n");
  } else {
    OL_SUCCESS_PRINTF("monte-carlo EI optimization for multiple simultaneous experiments\n");
  }
  total_errors += error;

  error = ExpectedImprovementOptimizationTest(DomainTypes::kSimplex, ExpectedImprovementEvaluationMode::kAnalytic);
  if (error != 0) {
    OL_FAILURE_PRINTF("analytic simplex EI optimization\n");
  } else {
    OL_SUCCESS_PRINTF("analytic simplex EI optimization\n");
  }
  total_errors += error;

  error = ExpectedImprovementOptimizationTest(DomainTypes::kSimplex, ExpectedImprovementEvaluationMode::kMonteCarlo);
  if (error != 0) {
    OL_FAILURE_PRINTF("monte-carlo simplex EI optimization\n");
  } else {
    OL_SUCCESS_PRINTF("monte-carlo simplex EI optimization\n");
  }
  total_errors += error;

#endif

  printf("\nTOTAL FAILURES: %d\n", total_errors);
  return 0;
}

#elif OL_MODE == 5

#define OL_FUNC_MODE 1

int main() {
  using DomainType = TensorProductDomain;
  using HyperparameterDomainType = TensorProductDomain;
  const int dim = 2;

  std::vector<ClosedInterval> domain_bounds = {
    {-5.0, 10.0},
    {0.0, 15.0}};
  DomainType domain(domain_bounds.data(), dim);

  // sample stencil of 9 points
  int stencil_rows = 10;
  int stencil_columns = 10;
  std::vector<double> points_sampled((stencil_rows*stencil_columns)*dim);
  std::vector<double> points_sampled_value(stencil_rows*stencil_columns);

  int num_total_samples = stencil_rows*stencil_columns;
  std::vector<double> noise_variance(num_total_samples);
  for (int i = 0; i < num_total_samples; ++i) {
    // noise_variance[i] = 1.0e-1;
    noise_variance[i] = 1.0e-1;
  }

//   SquareExponentialSingleLength covariance(dim, 1.0, 2.0);
//   SquareExponentialSingleLength covariance_perturbed(dim, 1.1, 2.1);

  SquareExponential covariance(dim, 1.0, 2.0);
// SquareExponential covariance_perturbed(dim, 2.5, 4.9);
  SquareExponential covariance_perturbed(dim, 0.1, 14.9);
  SquareExponential covariance_perturbed2(dim, 0.07, 0.19);
  SquareExponential covariance_perturbed3(dim, 10.21, 16.9);
  SquareExponential covariance_perturbed4(dim, 13.21, 0.06);

//   MaternNu1p5 covariance(dim, 1.0, 2.0);
//   MaternNu1p5 covariance_perturbed(dim, 1.5, 2.9);

  // MaternNu2p5 covariance(dim, 1.0, 2.0);
  // MaternNu2p5 covariance_perturbed(dim, 1.5, 2.9);
  // MaternNu2p5 covariance(dim, 1.0, 2.0);
  // MaternNu2p5 covariance_perturbed(dim, 0.1, 14.9);
  // MaternNu2p5 covariance_perturbed2(dim, 0.07, 0.19);
  // MaternNu2p5 covariance_perturbed3(dim, 10.21, 16.9);
  // MaternNu2p5 covariance_perturbed4(dim, 13.21, 0.06);

  UniformRandomGenerator uniform_generator(314);

  domain.GenerateUniformPointsInDomain(stencil_rows*stencil_columns, &uniform_generator, points_sampled.data());

#if OL_FUNC_MODE == 0
  for (int i = 0; i < stencil_rows*stencil_columns; ++i) {
    points_sampled_value[i] = branin_func(points_sampled.data() + (i)*dim);
  }
#elif OL_FUNC_MODE == 1
  GaussianProcess gp(covariance, points_sampled.data(), points_sampled_value.data(), noise_variance.data(), dim, 0);
  for (int j = 0; j < stencil_rows*stencil_columns; ++j) {
    points_sampled_value.data()[j] = gp.SamplePointFromGP(points_sampled.data() + dim*(j), noise_variance.data()[j]);
    gp.AddPointToGP(points_sampled.data() + dim*(j), points_sampled_value.data()[j], noise_variance.data()[j]);
  }
#else
  exit(-1);
#endif

  std::vector<double> new_hyperparameters(covariance.GetNumberOfHyperparameters());
  const int hyperparameter_max_num_steps = 1000;
  const double gamma_hyper = 0.5;
  const double pre_mult_hyper = 1.0;
  const double max_num_restarts = 10;
  const double max_relative_change = 0.02;
  const double tolerance_gd = 1.0e-7;
  GradientDescentParameters gd_parameters(1, hyperparameter_max_num_steps, max_num_restarts, gamma_hyper, pre_mult_hyper, max_relative_change, tolerance_gd);

  LogMarginalLikelihoodEvaluator log_marginal_eval(points_sampled.data(), points_sampled_value.data(), noise_variance.data(), dim, stencil_rows*stencil_columns);

#define OL_NEWTON_TEST 1
#if OL_NEWTON_TEST == 1
  {
    // SquareExponential covariance_perturbed_newton(dim, 0.1, 1.0);
    double lengths[2] = {0.48, 3.17};
    // double lengths[2] = {2.0, 1.03};
    SquareExponential covariance_perturbed_newton(dim, 0.1, lengths);
    // SquareExponential covariance_perturbed_newton(dim, 2.5, 4.9);
    // SquareExponential covariance_perturbed_newton(dim, 25.5, 45.9);
    // SquareExponential covariance_perturbed_newton(dim, 0.1, 14.9);
    // SquareExponential covariance_perturbed_newton(dim, 0.07, 0.19);
    // SquareExponential covariance_perturbed_newton(dim, 10.21, 16.9);
    // SquareExponential covariance_perturbed_newton(dim, 13.21, 0.06);

    std::vector<double> new_newton_hyperparameters(covariance.GetNumberOfHyperparameters());
    int newton_max_num_steps = 100;
    double pre_mult_newton = 1.0e-1;
    double gamma_newton = 1.05;
    double tolerance_newton = 1.0e-13;
    double max_relative_change_newton = 1.0;

    std::vector<ClosedInterval> hyperparameter_domain_bounds = {
      {-1.0, 1.0},
      {-2.0, 1.0},
      {-2.0, 1.0}};
    HyperparameterDomainType hyperparameter_domain(hyperparameter_domain_bounds.data(), covariance.GetNumberOfHyperparameters());
    int num_multistarts = 16;
    int max_num_threads = 4;

    NewtonParameters newton_parameters(num_multistarts, newton_max_num_steps, gamma_newton, pre_mult_newton, max_relative_change_newton, tolerance_newton);
    bool found_flag = false;
    MultistartNewtonHyperparameterOptimization(log_marginal_eval, covariance_perturbed_newton, newton_parameters, hyperparameter_domain_bounds.data(), max_num_threads, &found_flag, &uniform_generator, new_newton_hyperparameters.data());
    printf("result of newton:\n");
    PrintMatrix(new_newton_hyperparameters.data(), 1, covariance.GetNumberOfHyperparameters());

    covariance.SetHyperparameters(new_newton_hyperparameters.data());
    LogMarginalLikelihoodState log_marginal_state_newton_optimized_hyper(log_marginal_eval, covariance);
    double newton_log_marginal_opt = log_marginal_eval.ComputeLogLikelihood(log_marginal_state_newton_optimized_hyper);
    printf("newton optimized log marginal likelihood = %.18E\n", newton_log_marginal_opt);

    std::vector<double> grad_log_marginal_opt(covariance.GetNumberOfHyperparameters());
    log_marginal_eval.ComputeGradLogLikelihood(&log_marginal_state_newton_optimized_hyper, grad_log_marginal_opt.data());
    printf("grad log likelihood: likelihood opt: "); PrintMatrix(grad_log_marginal_opt.data(), 1, covariance.GetNumberOfHyperparameters());
  }
#else
  {
  std::vector<ClosedInterval> hyperparameter_domain_bounds(covariance_perturbed.GetNumberOfHyperparameters(), {1.0e-10, 1.0e10});
  HyperparameterDomainType hyperparameter_domain(hyperparameter_domain_bounds.data(), covariance.GetNumberOfHyperparameters());

  printf("LOG MARGINAL HYPERPARAM OPT:\n");
  RestartedGradientDescentHyperparameterOptimization(log_marginal_eval, covariance_perturbed, gd_parameters, hyperparameter_domain, new_hyperparameters.data());
  printf("opt hyper: "); PrintMatrix(new_hyperparameters.data(), 1, covariance.GetNumberOfHyperparameters());

  LogMarginalLikelihoodState log_marginal_state_initial_hyper(log_marginal_eval, covariance_perturbed);
  double log_marginal = log_marginal_eval.ComputeLogLikelihood(log_marginal_state_initial_hyper);
  printf("perturbed log marginal likelihood = %.18E\n", log_marginal);

  LogMarginalLikelihoodState log_marginal_state_generation_hyper(log_marginal_eval, covariance);
  log_marginal = log_marginal_eval.ComputeLogLikelihood(log_marginal_state_generation_hyper);
  printf("generation log marginal likelihood = %.18E\n", log_marginal);

  covariance.SetHyperparameters(new_hyperparameters.data());
  // getchar();
  LogMarginalLikelihoodState log_marginal_state_log_marginal_optimized_hyper(log_marginal_eval, covariance);
  // getchar();

  double log_marginal_opt = log_marginal_eval.ComputeLogLikelihood(log_marginal_state_log_marginal_optimized_hyper);
  printf("optimized log marginal likelihood = %.18E\n", log_marginal_opt);

  std::vector<double> grad_log_marginal(covariance.GetNumberOfHyperparameters());
  std::vector<double> grad_log_marginal_opt(covariance.GetNumberOfHyperparameters());
  log_marginal_eval.ComputeGradLogLikelihood(&log_marginal_state_log_marginal_optimized_hyper, grad_log_marginal_opt.data());

  printf("log likelihood hyper: "); PrintMatrix(new_hyperparameters.data(), 1, covariance.GetNumberOfHyperparameters());
  printf("log likelihood: initial: %.18E, likelihood opt: %.18E\n", log_marginal, log_marginal_opt);
  printf("grad log likelihood: likelihood opt: "); PrintMatrix(grad_log_marginal_opt.data(), 1, covariance.GetNumberOfHyperparameters());
  }
#endif
}

#elif OL_MODE == 6

#define OL_FUNC_MODE 1

int main() {
  using DomainType = TensorProductDomain;
  using HyperparameterDomainType = TensorProductDomain;
  const int dim = 2;

  std::vector<ClosedInterval> domain_bounds = {
    {-5.0, 10.0},
    {0.0, 15.0}};
  DomainType domain(domain_bounds.data(), dim);

  // sample stencil of 9 points
  int stencil_rows = 10;
  int stencil_columns = 10;
  std::vector<double> points_sampled((stencil_rows*stencil_columns)*dim);
  std::vector<double> points_sampled_value(stencil_rows*stencil_columns);

  int num_total_samples = stencil_rows*stencil_columns;
  std::vector<double> noise_variance(num_total_samples);
  for (int i = 0; i < num_total_samples; ++i) {
    // noise_variance[i] = 1.0e-1;
    noise_variance[i] = 1.0e-1;
  }

//   SquareExponentialSingleLength covariance(dim, 1.0, 2.0);
//   SquareExponentialSingleLength covariance_perturbed(dim, 1.1, 2.1);

  SquareExponential covariance(dim, 1.0, 2.0);
// SquareExponential covariance_perturbed(dim, 2.5, 4.9);
  SquareExponential covariance_perturbed(dim, 0.1, 14.9);
  SquareExponential covariance_perturbed2(dim, 0.07, 0.19);
  SquareExponential covariance_perturbed3(dim, 10.21, 16.9);
  SquareExponential covariance_perturbed4(dim, 13.21, 0.06);

//   MaternNu1p5 covariance(dim, 1.0, 2.0);
//   MaternNu1p5 covariance_perturbed(dim, 1.5, 2.9);

  // MaternNu2p5 covariance(dim, 1.0, 2.0);
  // MaternNu2p5 covariance_perturbed(dim, 1.5, 2.9);
  // MaternNu2p5 covariance(dim, 1.0, 2.0);
  // MaternNu2p5 covariance_perturbed(dim, 0.1, 14.9);
  // MaternNu2p5 covariance_perturbed2(dim, 0.07, 0.19);
  // MaternNu2p5 covariance_perturbed3(dim, 10.21, 16.9);
  // MaternNu2p5 covariance_perturbed4(dim, 13.21, 0.06);

  UniformRandomGenerator uniform_generator(314);

  domain.GenerateUniformPointsInDomain(stencil_rows*stencil_columns, &uniform_generator, points_sampled.data());

#if OL_FUNC_MODE == 0
  for (int i = 0; i < stencil_rows*stencil_columns; ++i) {
    points_sampled_value[i] = branin_func(points_sampled.data() + (i)*dim);
  }
#elif OL_FUNC_MODE == 1
  GaussianProcess gp(covariance, points_sampled.data(), points_sampled_value.data(), noise_variance.data(), dim, 0);
  for (int j = 0; j < stencil_rows*stencil_columns; ++j) {
    points_sampled_value.data()[j] = gp.SamplePointFromGP(points_sampled.data() + dim*(j), noise_variance.data()[j]);
    gp.AddPointToGP(points_sampled.data() + dim*(j), points_sampled_value.data()[j], noise_variance.data()[j]);
  }
#else
  exit(-1);
#endif

  std::vector<double> new_hyperparameters(covariance.GetNumberOfHyperparameters());
  const int hyperparameter_max_num_steps = 1000;
  const double gamma_hyper = 0.7;
  const double pre_mult_hyper = 1.0;
  const double max_num_restarts = 10;
  const double max_relative_change = 0.02;
  const double tolerance_gd = 1.0e-7;
  GradientDescentParameters gd_parameters(1, hyperparameter_max_num_steps, max_num_restarts, gamma_hyper, pre_mult_hyper, max_relative_change, tolerance_gd);

  std::vector<ClosedInterval> hyperparameter_domain_bounds(covariance_perturbed.GetNumberOfHyperparameters(), {1.0e-10, 1.0e10});
  HyperparameterDomainType hyperparameter_domain(hyperparameter_domain_bounds.data(), covariance.GetNumberOfHyperparameters());

  LogMarginalLikelihoodEvaluator log_marginal_eval(points_sampled.data(), points_sampled_value.data(), noise_variance.data(), dim, stencil_rows*stencil_columns);
  {
  printf("LOG MARGINAL HYPERPARAM OPT:\n");
  RestartedGradientDescentHyperparameterOptimization(log_marginal_eval, covariance_perturbed, gd_parameters, hyperparameter_domain, new_hyperparameters.data());
  printf("opt hyper: "); PrintMatrix(new_hyperparameters.data(), 1, covariance.GetNumberOfHyperparameters());

  LogMarginalLikelihoodState log_marginal_state_initial_hyper(log_marginal_eval, covariance_perturbed);
  double log_marginal = log_marginal_eval.ComputeLogLikelihood(log_marginal_state_initial_hyper);
  printf("perturbed log marginal likelihood = %.18E\n", log_marginal);

  LogMarginalLikelihoodState log_marginal_state_generation_hyper(log_marginal_eval, covariance);
  log_marginal = log_marginal_eval.ComputeLogLikelihood(log_marginal_state_generation_hyper);
  printf("generation log marginal likelihood = %.18E\n", log_marginal);

  covariance.SetHyperparameters(new_hyperparameters.data());
  // getchar();
  LogMarginalLikelihoodState log_marginal_state_log_marginal_optimized_hyper(log_marginal_eval, covariance);
  // getchar();

  double log_marginal_opt = log_marginal_eval.ComputeLogLikelihood(log_marginal_state_log_marginal_optimized_hyper);
  printf("optimized log marginal likelihood = %.18E\n", log_marginal_opt);

  std::vector<double> grad_log_marginal(covariance.GetNumberOfHyperparameters());
  std::vector<double> grad_log_marginal_opt(covariance.GetNumberOfHyperparameters());
  log_marginal_eval.ComputeGradLogLikelihood(&log_marginal_state_log_marginal_optimized_hyper, grad_log_marginal_opt.data());

  printf("log likelihood hyper: "); PrintMatrix(new_hyperparameters.data(), 1, covariance.GetNumberOfHyperparameters());
  printf("log likelihood: initial: %.18E, likelihood opt: %.18E\n", log_marginal, log_marginal_opt);
  printf("grad log likelihood: likelihood opt: "); PrintMatrix(grad_log_marginal_opt.data(), 1, covariance.GetNumberOfHyperparameters());
  }

  {
  printf("LOG MARGINAL HYPERPARAM OPT:\n");
  RestartedGradientDescentHyperparameterOptimization(log_marginal_eval, covariance_perturbed2, gd_parameters, hyperparameter_domain, new_hyperparameters.data());
  printf("opt hyper: "); PrintMatrix(new_hyperparameters.data(), 1, covariance.GetNumberOfHyperparameters());

  LogMarginalLikelihoodState log_marginal_state_initial_hyper(log_marginal_eval, covariance_perturbed2);
  double log_marginal = log_marginal_eval.ComputeLogLikelihood(log_marginal_state_initial_hyper);
  printf("perturbed2 log marginal likelihood = %.18E\n", log_marginal);

  LogMarginalLikelihoodState log_marginal_state_generation_hyper(log_marginal_eval, covariance);
  log_marginal = log_marginal_eval.ComputeLogLikelihood(log_marginal_state_generation_hyper);
  printf("generation log marginal likelihood = %.18E\n", log_marginal);

  covariance.SetHyperparameters(new_hyperparameters.data());
  // getchar();
  LogMarginalLikelihoodState log_marginal_state_log_marginal_optimized_hyper(log_marginal_eval, covariance);
  // getchar();

  double log_marginal_opt = log_marginal_eval.ComputeLogLikelihood(log_marginal_state_log_marginal_optimized_hyper);
  printf("optimized log marginal likelihood = %.18E\n", log_marginal_opt);

  std::vector<double> grad_log_marginal(covariance.GetNumberOfHyperparameters());
  std::vector<double> grad_log_marginal_opt(covariance.GetNumberOfHyperparameters());
  log_marginal_eval.ComputeGradLogLikelihood(&log_marginal_state_log_marginal_optimized_hyper, grad_log_marginal_opt.data());

  printf("log likelihood hyper: "); PrintMatrix(new_hyperparameters.data(), 1, covariance.GetNumberOfHyperparameters());
  printf("log likelihood: initial: %.18E, likelihood opt: %.18E\n", log_marginal, log_marginal_opt);
  printf("grad log likelihood: likelihood opt: "); PrintMatrix(grad_log_marginal_opt.data(), 1, covariance.GetNumberOfHyperparameters());
  }

  {
  printf("LOG MARGINAL HYPERPARAM OPT:\n");
  RestartedGradientDescentHyperparameterOptimization(log_marginal_eval, covariance_perturbed3, gd_parameters, hyperparameter_domain, new_hyperparameters.data());
  printf("opt hyper: "); PrintMatrix(new_hyperparameters.data(), 1, covariance.GetNumberOfHyperparameters());

  LogMarginalLikelihoodState log_marginal_state_initial_hyper(log_marginal_eval, covariance_perturbed3);
  double log_marginal = log_marginal_eval.ComputeLogLikelihood(log_marginal_state_initial_hyper);
  printf("perturbed3 log marginal likelihood = %.18E\n", log_marginal);

  LogMarginalLikelihoodState log_marginal_state_generation_hyper(log_marginal_eval, covariance);
  log_marginal = log_marginal_eval.ComputeLogLikelihood(log_marginal_state_generation_hyper);
  printf("generation log marginal likelihood = %.18E\n", log_marginal);

  covariance.SetHyperparameters(new_hyperparameters.data());
  // getchar();
  LogMarginalLikelihoodState log_marginal_state_log_marginal_optimized_hyper(log_marginal_eval, covariance);
  // getchar();

  double log_marginal_opt = log_marginal_eval.ComputeLogLikelihood(log_marginal_state_log_marginal_optimized_hyper);
  printf("optimized log marginal likelihood = %.18E\n", log_marginal_opt);

  std::vector<double> grad_log_marginal(covariance.GetNumberOfHyperparameters());
  std::vector<double> grad_log_marginal_opt(covariance.GetNumberOfHyperparameters());
  log_marginal_eval.ComputeGradLogLikelihood(&log_marginal_state_log_marginal_optimized_hyper, grad_log_marginal_opt.data());

  printf("log likelihood hyper: "); PrintMatrix(new_hyperparameters.data(), 1, covariance.GetNumberOfHyperparameters());
  printf("log likelihood: initial: %.18E, likelihood opt: %.18E\n", log_marginal, log_marginal_opt);
  printf("grad log likelihood: likelihood opt: "); PrintMatrix(grad_log_marginal_opt.data(), 1, covariance.GetNumberOfHyperparameters());
  }

  {
  printf("LOG MARGINAL HYPERPARAM OPT:\n");
  RestartedGradientDescentHyperparameterOptimization(log_marginal_eval, covariance_perturbed4, gd_parameters, hyperparameter_domain, new_hyperparameters.data());
  printf("opt hyper: "); PrintMatrix(new_hyperparameters.data(), 1, covariance.GetNumberOfHyperparameters());

  LogMarginalLikelihoodState log_marginal_state_initial_hyper(log_marginal_eval, covariance_perturbed4);
  double log_marginal = log_marginal_eval.ComputeLogLikelihood(log_marginal_state_initial_hyper);
  printf("perturbed4 log marginal likelihood = %.18E\n", log_marginal);

  LogMarginalLikelihoodState log_marginal_state_generation_hyper(log_marginal_eval, covariance);
  log_marginal = log_marginal_eval.ComputeLogLikelihood(log_marginal_state_generation_hyper);
  printf("generation log marginal likelihood = %.18E\n", log_marginal);

  covariance.SetHyperparameters(new_hyperparameters.data());
  // getchar();
  LogMarginalLikelihoodState log_marginal_state_log_marginal_optimized_hyper(log_marginal_eval, covariance);
  // getchar();

  double log_marginal_opt = log_marginal_eval.ComputeLogLikelihood(log_marginal_state_log_marginal_optimized_hyper);
  printf("optimized log marginal likelihood = %.18E\n", log_marginal_opt);

  std::vector<double> grad_log_marginal(covariance.GetNumberOfHyperparameters());
  std::vector<double> grad_log_marginal_opt(covariance.GetNumberOfHyperparameters());
  log_marginal_eval.ComputeGradLogLikelihood(&log_marginal_state_log_marginal_optimized_hyper, grad_log_marginal_opt.data());

  printf("log likelihood hyper: "); PrintMatrix(new_hyperparameters.data(), 1, covariance.GetNumberOfHyperparameters());
  printf("log likelihood: initial: %.18E, likelihood opt: %.18E\n", log_marginal, log_marginal_opt);
  printf("grad log likelihood: likelihood opt: "); PrintMatrix(grad_log_marginal_opt.data(), 1, covariance.GetNumberOfHyperparameters());
  }
}

#elif OL_MODE == 7

int main() {
  using DomainType = TensorProductDomain;
  using HyperparameterDomainType = TensorProductDomain;
  const int dim = 2;

  std::vector<ClosedInterval> domain_bounds = {
    {-5.0, 10.0},
    {0.0, 15.0}};
  DomainType domain(domain_bounds.data(), dim);

  // sample stencil of 9 points
  int stencil_rows = 10;
  int stencil_columns = 10;
  std::vector<double> points_sampled((stencil_rows*stencil_columns)*dim);
  std::vector<double> points_sampled_value(stencil_rows*stencil_columns);

  int num_total_samples = stencil_rows*stencil_columns;
  std::vector<double> noise_variance(num_total_samples);
  for (int i = 0; i < num_total_samples; ++i) {
    // noise_variance[i] = 1.0e-1;
    noise_variance[i] = 1.0e-1;
  }

//   SquareExponentialSingleLength covariance(dim, 1.0, 2.0);
//   SquareExponentialSingleLength covariance_perturbed(dim, 1.1, 2.1);

  SquareExponential covariance(dim, 1.0, 2.0);
// SquareExponential covariance_perturbed(dim, 2.5, 4.9);
  SquareExponential covariance_perturbed(dim, 0.1, 14.9);
  SquareExponential covariance_perturbed2(dim, 0.07, 0.19);
  SquareExponential covariance_perturbed3(dim, 10.21, 16.9);
  SquareExponential covariance_perturbed4(dim, 13.21, 0.06);

//   MaternNu1p5 covariance(dim, 1.0, 2.0);
//   MaternNu1p5 covariance_perturbed(dim, 1.5, 2.9);

  // MaternNu2p5 covariance(dim, 1.0, 2.0);
  // MaternNu2p5 covariance_perturbed(dim, 1.5, 2.9);
  // MaternNu2p5 covariance(dim, 1.0, 2.0);
  // MaternNu2p5 covariance_perturbed(dim, 0.1, 14.9);
  // MaternNu2p5 covariance_perturbed2(dim, 0.07, 0.19);
  // MaternNu2p5 covariance_perturbed3(dim, 10.21, 16.9);
  // MaternNu2p5 covariance_perturbed4(dim, 13.21, 0.06);

  UniformRandomGenerator uniform_generator(314);

  domain.GenerateUniformPointsInDomain(stencil_rows*stencil_columns, &uniform_generator, points_sampled.data());

  GaussianProcess gp(covariance, points_sampled.data(), points_sampled_value.data(), noise_variance.data(), dim, 0);
  for (int j = 0; j < stencil_rows*stencil_columns; ++j) {
    points_sampled_value.data()[j] = gp.SamplePointFromGP(points_sampled.data() + dim*(j), noise_variance.data()[j]);
    gp.AddPointToGP(points_sampled.data() + dim*(j), points_sampled_value.data()[j], noise_variance.data()[j]);
  }

  std::vector<double> new_hyperparameters(covariance.GetNumberOfHyperparameters());
  const int num_multistarts = 8;
  const int hyperparameter_max_num_steps = 500;
  const double gamma_hyper = 0.7;
  const double pre_mult_hyper = 1.0;
  const double max_num_restarts = 10;
  const double max_relative_change = 0.02;
  const double tolerance_gd = 1.0e-7;
  GradientDescentParameters gd_parameters(num_multistarts, hyperparameter_max_num_steps, max_num_restarts, gamma_hyper, pre_mult_hyper, max_relative_change, tolerance_gd);

  LogMarginalLikelihoodEvaluator log_marginal_eval(points_sampled.data(), points_sampled_value.data(), noise_variance.data(), dim, stencil_rows*stencil_columns);

  {
    // SquareExponential covariance_perturbed_gd(dim, 0.1, 1.0);
    double lengths[2] = {0.48, 3.17};
    // double lengths[2] = {2.0, 1.03};
    SquareExponential covariance_perturbed_gd(dim, 0.1, lengths);
    // SquareExponential covariance_perturbed_gd(dim, 2.5, 4.9);
    // SquareExponential covariance_perturbed_gd(dim, 25.5, 45.9);
    // SquareExponential covariance_perturbed_gd(dim, 0.1, 14.9);
    // SquareExponential covariance_perturbed_gd(dim, 0.07, 0.19);
    // SquareExponential covariance_perturbed_gd(dim, 10.21, 16.9);
    // SquareExponential covariance_perturbed_gd(dim, 13.21, 0.06);

    std::vector<double> new_gd_hyperparameters(covariance.GetNumberOfHyperparameters());
    std::vector<ClosedInterval> hyperparameter_domain_bounds = {
      {-1.0, 1.0},
      {-2.0, 1.0},
      {-2.0, 1.0}};
    HyperparameterDomainType hyperparameter_domain(hyperparameter_domain_bounds.data(), covariance.GetNumberOfHyperparameters());

    int max_num_threads = 4;

    bool found_flag = false;
    MultistartGradientDescentHyperparameterOptimization(log_marginal_eval, covariance_perturbed_gd, gd_parameters, hyperparameter_domain_bounds.data(), max_num_threads, &found_flag, &uniform_generator, new_gd_hyperparameters.data());
    printf("result of gd:\n");
    PrintMatrix(new_gd_hyperparameters.data(), 1, covariance.GetNumberOfHyperparameters());

    covariance.SetHyperparameters(new_gd_hyperparameters.data());
    LogMarginalLikelihoodState log_marginal_state_gd_optimized_hyper(log_marginal_eval, covariance);
    double gd_log_marginal_opt = log_marginal_eval.ComputeLogLikelihood(log_marginal_state_gd_optimized_hyper);
    printf("gd optimized log marginal likelihood = %.18E\n", gd_log_marginal_opt);

    std::vector<double> grad_log_marginal_opt(covariance.GetNumberOfHyperparameters());
    log_marginal_eval.ComputeGradLogLikelihood(&log_marginal_state_gd_optimized_hyper, grad_log_marginal_opt.data());
    printf("grad log likelihood: likelihood opt: "); PrintMatrix(grad_log_marginal_opt.data(), 1, covariance.GetNumberOfHyperparameters());
  }
}

#elif OL_MODE == 8

int main() {
  using DomainType = TensorProductDomain;
  using HyperparameterDomainType = TensorProductDomain;
  const int dim = 3;

  int total_errors = 0;
  int current_errors = 0;

  // grid search parameters
  int num_grid_search_points = 100000;

  // EI computation parameters
  int num_to_sample = 0;
  int max_int_steps = 1000;

  // random number generators
  UniformRandomGenerator uniform_generator(314);
  boost::uniform_real<double> uniform_double_hyperparameter(0.4, 1.3);
  boost::uniform_real<double> uniform_double_lower_bound(-2.0, 0.5);
  boost::uniform_real<double> uniform_double_upper_bound(2.0, 3.5);

  static const int kMaxNumThreads = 4;

  SquareExponential covariance(dim, 1.0, 1.0);
  std::vector<double> hyperparameters(covariance.GetNumberOfHyperparameters());
  for (auto& entry : hyperparameters) {
    entry = uniform_double_hyperparameter(uniform_generator.engine);
  }
  covariance.SetHyperparameters(hyperparameters.data());

  std::vector<ClosedInterval> domain_bounds(dim);
  for (int i = 0; i < dim; ++i) {
    domain_bounds[i].min = uniform_double_lower_bound(uniform_generator.engine);
    domain_bounds[i].max = uniform_double_upper_bound(uniform_generator.engine);
  }
  DomainType domain_gp_source(domain_bounds.data(), dim);

  int num_sampled;
  int objective_mode = 1;
  if (objective_mode == 0) {
    num_sampled = 20;  // need to keep this similar to the number of multistarts
  } else {
    num_sampled = 80;  // matters less here b/c we end up starting one multistart from the LHC-search optima
  }

  std::vector<double> points_sampled(num_sampled*dim);
  std::vector<double> points_sampled_value(num_sampled);
  std::vector<double> noise_variance(num_sampled, 0.002);

  GaussianProcess gaussian_process(covariance, points_sampled.data(), points_sampled_value.data(), noise_variance.data(), dim, 0);

  // generate random sampling points
  domain_gp_source.GenerateUniformPointsInDomain(num_sampled, &uniform_generator, points_sampled.data());

  // generate the "world"
  for (int j = 0; j < num_sampled; ++j) {
    points_sampled_value.data()[j] = gaussian_process.SamplePointFromGP(points_sampled.data() + dim*j, noise_variance[j]);
    gaussian_process.AddPointToGP(points_sampled.data() + dim*j, points_sampled_value[j], noise_variance[j]);
  }

  // get best point
  double best_so_far = *std::min_element(points_sampled_value.begin(), points_sampled_value.end());

  using LogLikelihoodEvaluator = LogMarginalLikelihoodEvaluator;
  // log likelihood evaluator object
  LogLikelihoodEvaluator log_marginal_eval(points_sampled.data(), points_sampled_value.data(), noise_variance.data(), dim, num_sampled);
  int num_hyperparameters = covariance.GetNumberOfHyperparameters();
  std::vector<ClosedInterval> hyperparameter_log_domain_bounds(num_hyperparameters, {-2.0, 1.0});
  HyperparameterDomainType hyperparameter_log_domain(hyperparameter_log_domain_bounds.data(), num_hyperparameters);

  struct timeval tv0, tv1;

  gettimeofday(&tv0, nullptr);

  time_t c0, c1;
  c0 = clock();

  std::vector<double> next_point(num_hyperparameters);
  bool found_flag = false;
  std::vector<double> grid_search_best_point(num_hyperparameters);

  // LatinHypercubeSearchHyperparameterOptimization(log_marginal_eval, covariance, hyperparameter_log_domain_bounds.data(), num_grid_search_points, kMaxNumThreads, &uniform_generator, grid_search_best_point.data());

  std::vector<double> function_values(num_grid_search_points);
  std::vector<double> initial_guesses(num_hyperparameters*num_grid_search_points);
  num_grid_search_points = hyperparameter_log_domain.GenerateUniformPointsInDomain(num_grid_search_points, &uniform_generator, initial_guesses.data());
  for (auto& point : initial_guesses) {
    point = pow(10.0, point);
  }

  // domain in linear-space
  std::vector<ClosedInterval> hyperparameter_domain_linearspace_bounds(hyperparameter_log_domain_bounds);
  for (auto& interval : hyperparameter_domain_linearspace_bounds) {
    interval.min = std::pow(10.0, interval.min);
    interval.max = std::pow(10.0, interval.max);
  }
  HyperparameterDomainType hyperparameter_domain_linearspace(hyperparameter_domain_linearspace_bounds.data(), num_hyperparameters);

  EvaluateLogLikelihoodAtPointList(log_marginal_eval, covariance, hyperparameter_domain_linearspace, initial_guesses.data(), num_grid_search_points, kMaxNumThreads, function_values.data(), grid_search_best_point.data());

  gettimeofday(&tv1, nullptr);
  c1 = clock();
  printf("time: %f\n", static_cast<double>((c1 - c0))/static_cast<double>(CLOCKS_PER_SEC));

  int diff = (tv1.tv_sec - tv0.tv_sec) * 1000000;
  diff += tv1.tv_usec - tv0.tv_usec;

  printf("diff = %d, %f\n", diff, static_cast<double>(diff)/1000000.0);

  PrintMatrix(grid_search_best_point.data(), 1, num_hyperparameters);

  int max_index = 0;
  double best_value = function_values[0];
  for (int i = 0; i < num_grid_search_points; ++i) {
    if (function_values[i] > best_value) {
      best_value = function_values[i];
      max_index = i;
    }
  }
  printf("best = %.18E, index = %d\n", best_value, max_index);
  PrintMatrix(initial_guesses.data() + max_index*num_hyperparameters, 1, num_hyperparameters);

  return 0;
}

#elif OL_MODE == 9

static constexpr int kNumberOfTests = 30;

int main() {
  struct timeval tv[kNumberOfTests];
  int count[kNumberOfTests], diff;

  gettimeofday(&tv[0], nullptr);

  for (int i = 1; i < kNumberOfTests; i++) {
    gettimeofday(&tv[i], nullptr);
    count[i] = 1;
    while ((tv[i].tv_sec == tv[i-1].tv_sec) &&
           (tv[i].tv_usec == tv[i-1].tv_usec)) {
      gettimeofday(&tv[i], nullptr);
      count[i]++;
    }
  }

  printf("%2d: secs = %d, usecs = %6d\n", 0, tv[0].tv_sec, tv[0].tv_usec);
  for (int i = 1; i < kNumberOfTests; i++) {
    diff = (tv[i].tv_sec - tv[i-1].tv_sec) * 1000000;
    diff += tv[i].tv_usec - tv[i-1].tv_usec;

    printf("%2d: secs = %d, usecs = %6d, count = %5d, diff = %d\n", i, tv[i].tv_sec, tv[i].tv_usec, count[i], diff);
  }

  return 0;
}

#endif
