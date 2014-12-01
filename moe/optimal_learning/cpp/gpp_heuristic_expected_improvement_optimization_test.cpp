/*!
  \file gpp_heuristic_expected_improvement_optimization_test.cpp
  \rst
  Routines to test the functions in gpp_heuristic_expected_improvement_optimization.cpp.
  The tests verify the subclasses of ObjectiveEstimationPolicyInterface and the correctness of
  ComputeHeuristicPointsToSample():

  1. ObjectiveEstimationPolicyInterface

     a. ConstantLiarEstimationPolicy
        Verify that constant liar gives back the same, constant output regardless of inputs (e.g., test against
        invalid inputs).
     b. KrigingBelieverEstimationPolicy
        Kriging's output depends on GP computations (mean, variance). In some special cases, we know these quantities
        analytically, so we test that Kriging gives the expected output in those cases.

  2. ComputeHeuristicPointsToSample
     We have an end-to-end test of this functionality using both ConstantLiar and KrigingBeliever. We check that
     the output is valid (e.g., in the domain, distinct) and that the points correspond to local optima (i.e., each
     round of solving 1-EI succeeded).
\endrst*/

#include "gpp_heuristic_expected_improvement_optimization_test.hpp"

#include <cmath>

#include <algorithm>
#include <limits>
#include <memory>
#include <vector>

#include <boost/random/uniform_real.hpp>  // NOLINT(build/include_order)

#include "gpp_common.hpp"
#include "gpp_covariance.hpp"
#include "gpp_domain.hpp"
#include "gpp_heuristic_expected_improvement_optimization.hpp"
#include "gpp_linear_algebra.hpp"
#include "gpp_logging.hpp"
#include "gpp_math.hpp"
#include "gpp_random.hpp"
#include "gpp_test_utils.hpp"

namespace optimal_learning {

namespace {

/*!\rst
  Checks that ``|value.function_value - truth.function_value| / |truth.function_value| <= tolerance`` (relative error)
  and
  ``value.noise_variance == truth.noise_variance``.

  Currently we never manipulate the ``noise_variance``, so we check the accuracy of this value with a 0 tolerance.

  \param
    :function_value: FunctionValue to be tested
    :function_value_truth: FunctionValue providing the exact/desired result
    :tolerance: permissible relative difference
  \return
    true if value and truth match to within tolerance
\endrst*/
OL_WARN_UNUSED_RESULT bool CheckFunctionValue(const FunctionValue& function_value, const FunctionValue& function_value_truth, double tolerance) {
  return CheckDoubleWithinRelative(function_value.function_value, function_value_truth.function_value, tolerance) && CheckDoubleWithinRelative(function_value.noise_variance, function_value_truth.noise_variance, 0.0);
}

/*!\rst
  Tests ConstantLiarEstimationPolicy (construction and ComputeEstimate() method).
  ConstantLiar is very simple; it always produces the same estimates. So we test against some "bad" inputs;
  e.g., empty gaussian process, invalid sampling point, etc. to ensure that the object never looks at these.

  \return
    number of test failures: 0 if ConstantLiarEstimationPolicy is working correctly
\endrst*/
OL_WARN_UNUSED_RESULT int ConstantLiarPolicyTest() {
  int total_errors = 0;

  const double lie_value_truth = -3.1489;
  const double lie_noise_variance_truth = 1.872;
  FunctionValue function_value_truth(lie_value_truth, lie_noise_variance_truth);

  ConstantLiarEstimationPolicy constant_liar(lie_value_truth, lie_noise_variance_truth);

  // need some dummy variables to invoke constant_liar.ComputeEstimate()
  std::vector<double> dummy;
  int dim = 0;
  SquareExponential covariance(dim, 1.0, 1.0);
  GaussianProcess gaussian_process(covariance, dummy.data(), dummy.data(), dummy.data(), dim, 0);

  // Verify constant_liar 'always' spits out its input values (call twice)
  // using an empty vector
  FunctionValue output1 = constant_liar.ComputeEstimate(gaussian_process, dummy.data(), 0);
  if (!CheckFunctionValue(output1, function_value_truth, 0.0)) {
    total_errors += 1;
  }

  // using a vector of invalid data
  std::vector<double> nan_vector(8, std::numeric_limits<double>::quiet_NaN());
  FunctionValue output2 = constant_liar.ComputeEstimate(gaussian_process, nan_vector.data(), 8976);
  if (!CheckFunctionValue(output2, function_value_truth, 0.0)) {
    total_errors += 1;
  }

  return total_errors;
}

/*!\rst
  Implementation Notes:
  This test relies on two provable properties of 0-noise GPs:

  1. Setting ``Xs = X`` (``points_to_sample = points_sampled``), we always obtain
     ``mu(Xs) = f`` (``points_sampled_value``)
     ``Vars(Xs) = 0_{n,n}`` (``n x n`` zero matrix; ``n = num_sampled``)
  2. For a point ``x`` such that ``dist(x, X) >> max(hyperparameter_length_scales)``,
     ``mus \approx 0`` and ``Vars \approx \alpha``.  (``dist(x, X) = \min_i ||x - X_i||_2``)

  So we will evaluate Kriging Believer using 1. and 2. and verify that the
  expected outcomes are produced.

  Proof of 1. for ``Xs = X_1``, the first point of ``X``.
  Let ``e_i`` be the column vector with a 1 in the ``i``-th slot and 0s otherwise.
  Let ``K`` be the covariance matrix, and let ``K_1, K_2, ... K_n`` denote its *columns*. Say we are using SquareExponential.
  Observe that ``Ks = K_1``.
  and ``Kss = \alpha`` (the signal variance hyperparameter)
  From gpp_math.cpp's file comments:

  * ``mus = Ks^T * K^-1 * f,  (Equation 2, Rasmussen & Williams 2.19)``
  * ``Vars = Kss - Ks^T * K^-1 * Ks, (Equation 3, Rasumussen & Williams 2.19)``

  Using ``Ks^T * K^{-1} = [K^{-1} * Ks]^T = [K^{-1} * K_1]^T = e_1^T``,
  ``mus = e_1^T * f = f_1`` (first entry of ``f``)
  ``Vars = Kss - Ks^T * K^{-1} * Ks = \alpha - e_1^T * K_1^T = \alpha - \alpha = 0``.

  Proof of 2. for ``Xs = x_{far}``, where ``\|x_{far} - X_i\|_2 >> max L \forall i``
  Observe ``Ks \approx 0_n`` (vector of all 0s).
  To see why, consider that ``cov(x, y) = \alpha * exp(-0.5*\sum_{i=0}^{dim} (x_i - y_i)^2/L_i^2)``.
  So for ``x = x_{far}`` and any ``y \in X``, this is ``\alpha * exp(-LARGE) \approx 0``

  And ``Kss = \alpha`` (as before). Hence:
  ``mus \approx 0_n^T * K^{-1} * f = 0``
  ``Vars \approx \alpha - 0_n^T * K^{-1} * 0_n = \alpha``
  The ``\approx`` become "equalities" in finite precision when
  ``dist(x, X) \approx 20 * max(hyperparameter_length_scales)``. (Conservative estimate.)

  Function Docs:
  Tests KrigingBelieverEstimationPolicy (construction and ComputeEstimate() method).
  KrigingBeliever uses GP.Mean() and potentially GP.Variance() as well. It is difficult to know precisely
  what these values are (without just recomputing them exactly the same way that KrigingBeliever does), so we
  check properties 1) and 2) from above. That is, we check:

  1. At point ``X_i`` (for i = 0..num_sampled-1), KrigingBeliever spits out f_i
  2. At random points that are "far away" (see code for details), KrigingBeliever spits out ``sqrt(\alpha) * std_dev_coef``

  \return
    number of test failures: 0 if KrigingBelieverEstimationPolicy is working correctly
\endrst*/
OL_WARN_UNUSED_RESULT int KrigingBelieverPolicyTest() {
  using DomainType = TensorProductDomain;
  const int dim = 3;

  int total_errors = 0;

  // random number generators
  UniformRandomGenerator uniform_generator(3141);
  boost::uniform_real<double> uniform_double_hyperparameter(0.01, 0.3);
  boost::uniform_real<double> uniform_double_lower_bound(-0.1, 0.0);
  boost::uniform_real<double> uniform_double_upper_bound(1.0, 1.1);

  int num_sampled = 20;  // need to keep this similar to the number of multistarts
  std::vector<double> noise_variance(num_sampled, 0.0);
  MockGaussianProcessPriorData<DomainType> mock_gp_data(SquareExponential(dim, 1.0, 1.0), noise_variance, dim, num_sampled, uniform_double_lower_bound, uniform_double_upper_bound, uniform_double_hyperparameter, &uniform_generator);
  double alpha = mock_gp_data.hyperparameters[0];
  double max_length_scale = *std::max_element(mock_gp_data.hyperparameters.begin() + 1, mock_gp_data.hyperparameters.end());

  const double kriging_noise_variance_truth = 1.872;
  // Don't use std_dev_coef here b/c computing GP.Variance() (exact value 0.0) is very unstable
  KrigingBelieverEstimationPolicy kriging_believer_no_var(0.0, kriging_noise_variance_truth);

  // Check property 1)
  for (int i = 0; i < num_sampled; ++i) {
    double truth_function_value = mock_gp_data.gaussian_process_ptr->points_sampled_value()[i];
    FunctionValue function_value_truth(truth_function_value, kriging_noise_variance_truth);

    FunctionValue kriging = kriging_believer_no_var.ComputeEstimate(*mock_gp_data.gaussian_process_ptr, mock_gp_data.gaussian_process_ptr->points_sampled().data() + i*dim, i);

    if (!CheckFunctionValue(kriging, function_value_truth, 8.0*std::numeric_limits<double>::epsilon())) {
      ++total_errors;
    }
  }

  // Check property 2)
  const double kriging_std_dev_coef = 0.25;
  KrigingBelieverEstimationPolicy kriging_believer(kriging_std_dev_coef, kriging_noise_variance_truth);

  double max_points_sampled_value = 0.0;
  for (auto entry : mock_gp_data.gaussian_process_ptr->points_sampled_value()) {
    if (std::fabs(entry) > max_points_sampled_value) {
      max_points_sampled_value = std::fabs(entry);
    }
  }

  double max_domain_length = 0.0;
  std::vector<double> domain_centroid(dim);
  for (int i = 0; i < dim; ++i) {
    if (max_domain_length < mock_gp_data.domain_bounds[i].Length()) {
      max_domain_length = mock_gp_data.domain_bounds[i].Length();
    }
    domain_centroid[i] = 0.5*(mock_gp_data.domain_bounds[i].min + mock_gp_data.domain_bounds[i].max);
  }

  // Pick point (outside domain) so that \mu < 1.0e-40. Produce a rough bound to compute this.
  // Estimate the point as >= m*L away from of points_sampled (L = max length scale).
  // 1.0e-40 >= \alpha * exp(-0.5 * d * (m*L)^2/L^2) * 1_n * max_i (f_i) * 1_n
  // yielding m \approx sqrt(-2/d*ln(1.0e-40 / \alpha / n / max_i (f_i)))
  double mean_tolerance = 1.0e-40;
  double scale_factor = 1.3*sqrt(-2.0/static_cast<double>(dim)*std::log(mean_tolerance / alpha / max_points_sampled_value / static_cast<double>(num_sampled)));

  int num_random_tries = 10;
  boost::uniform_real<double> uniform_unit_interval(std::numeric_limits<double>::min(), 1.0);
  for (int i = 0; i < num_random_tries; ++i) {
    // Generate random point that is far enough from the domain.
    // First, make a random unit vector.
    std::vector<double> random_unit_vector(dim);
    double norm = 0.0;
    for (auto& entry : random_unit_vector) {
      entry = uniform_unit_interval(uniform_generator.engine);
      norm += Square(entry);
    }
    norm = std::sqrt(norm);
    VectorScale(dim, 1.0/norm, random_unit_vector.data());

    // Now travel outward from the centroid in the direction of the unit vector; travel the sum
    // max_domain_length + scale_factor*max_length_scale
    std::vector<double> random_point(domain_centroid);
    VectorAXPY(dim, max_domain_length + scale_factor*max_length_scale, random_unit_vector.data(), random_point.data());

    double truth_function_value = std::sqrt(alpha)*kriging_std_dev_coef;
    FunctionValue function_value_truth(truth_function_value, kriging_noise_variance_truth);
    FunctionValue kriging = kriging_believer.ComputeEstimate(*mock_gp_data.gaussian_process_ptr, random_point.data(), i);

    // We know GP.Mean() to within mean_tolerance (true value is 0.0) and we know GP.Variance() to within Square(mean_tolerance),
    // (due to the way random_point is constructed), so we can afford to be very strict on the computation accuracy.
    if (!CheckFunctionValue(kriging, function_value_truth, Square(mean_tolerance))) {
      ++total_errors;
    }
  }

  return total_errors;
}

}  // end unnamed namespace

int EstimationPolicyTest() {
  int total_errors = 0;
  int current_errors = 0;

  current_errors = ConstantLiarPolicyTest();
  if (current_errors != 0) {
    OL_PARTIAL_FAILURE_PRINTF("ConstantLiar failed with %d errors\n", current_errors);
  } else {
    OL_PARTIAL_SUCCESS_PRINTF("ConstantLiar passed all tests\n");
  }
  total_errors += current_errors;

  current_errors = KrigingBelieverPolicyTest();
  if (current_errors != 0) {
    OL_PARTIAL_FAILURE_PRINTF("KrigingBeliever failed with %d errors\n", current_errors);
  } else {
    OL_PARTIAL_SUCCESS_PRINTF("KrigingBeliever passed all tests\n");
  }
  total_errors += current_errors;

  return total_errors;
}

namespace {

/*!\rst
  This test assumes that ComputeOptimalPointsToSampleWithRandomStarts() with num_being_sampled = 0 (analytic case)
  is working properly, and it assumes EstimationPolicyTest() passes. It checks:

  1. ComputeHeuristicPointsToSample() is working correctly (found_flag is true)
  2. points returned are all inside the specified domain
  3. points returned are not within epsilon of each other (i.e., distinct)
  4. as you add each new "to_sample" point to the GP (one at a time), the gradient of EI is 0 at the next
     "to_sample" point

  The test sets up a toy problem by repeatedly drawing from a GP with made-up hyperparameters.
  Then it runs EI optimization, attempting to sample 3 points simultaneously.

  \param
    :policy_type: which estimation policy (e.g., ConstantLiar, KrigingBeliever) to check in EI optimization
  \return
    number of test failures: 0 if heuristic EI optimization with the specified EstimationPolicy is working properly
\endrst*/
int HeuristicExpectedImprovementOptimizationTestCore(EstimationPolicyTypes policy_type) {
  using DomainType = TensorProductDomain;
  const int dim = 3;

  int total_errors = 0;
  int current_errors = 0;

  // gradient descent parameters
  const double gamma = 0.4;
  const double pre_mult = 1.3;
  const double max_relative_change = 1.0;
  const double tolerance = 1.0e-12;
  const int max_gradient_descent_steps = 300;
  const int max_num_restarts = 5;
  const int num_steps_averaged = 0;
  const int num_multistarts = 20;
  GradientDescentParameters gd_params(num_multistarts, max_gradient_descent_steps,
                                      max_num_restarts, num_steps_averaged, gamma, pre_mult,
                                      max_relative_change, tolerance);

  static const int kMaxNumThreads = 4;
  ThreadSchedule thread_schedule(kMaxNumThreads, omp_sched_dynamic);

  // grid search parameters
  bool grid_search_only = false;
  int num_grid_search_points = 10000;

  // random number generators
  UniformRandomGenerator uniform_generator(314);
  boost::uniform_real<double> uniform_double_hyperparameter(0.4, 1.3);
  boost::uniform_real<double> uniform_double_lower_bound(-2.0, 0.5);
  boost::uniform_real<double> uniform_double_upper_bound(2.0, 3.5);

  int num_sampled = 20;  // need to keep this similar to the number of multistarts
  std::vector<double> noise_variance(num_sampled, 0.002);
  MockGaussianProcessPriorData<DomainType> mock_gp_data(SquareExponential(dim, 1.0, 1.0),
                                                        noise_variance, dim, num_sampled,
                                                        uniform_double_lower_bound,
                                                        uniform_double_upper_bound,
                                                        uniform_double_hyperparameter,
                                                        &uniform_generator);

  // Estimation Policy
  std::unique_ptr<ObjectiveEstimationPolicyInterface> estimation_policy;
  switch (policy_type) {
    case EstimationPolicyTypes::kConstantLiar: {
      double lie_value = mock_gp_data.best_so_far;  // Uses the "CL-min" version of constant liar
      double lie_noise_variance = 0.0;
      estimation_policy.reset(new ConstantLiarEstimationPolicy(lie_value, lie_noise_variance));
      break;
    }
    case EstimationPolicyTypes::kKrigingBeliever: {
      double kriging_std_dev_coef = 0.3;
      double kriging_noise_variance = 0.0;
      estimation_policy.reset(new KrigingBelieverEstimationPolicy(kriging_std_dev_coef, kriging_noise_variance));
      break;
    }
    default: {
      OL_ERROR_PRINTF("%s: INVALID policy_type choice: %d\n", OL_CURRENT_FUNCTION_NAME, policy_type);
      return 1;
    }
  }

  // we will optimize over the expanded region
  std::vector<ClosedInterval> domain_bounds(mock_gp_data.domain_bounds);
  ExpandDomainBounds(1.5, &domain_bounds);
  DomainType domain(domain_bounds.data(), dim);

  // number of simultaneous samples
  const int num_to_sample = 3;
  std::vector<double> best_points_to_sample(dim*num_to_sample);

  // test optimization
  bool found_flag = false;
  ComputeHeuristicPointsToSample(*mock_gp_data.gaussian_process_ptr, gd_params, domain, *estimation_policy,
                                 thread_schedule, mock_gp_data.best_so_far, grid_search_only,
                                 num_grid_search_points, num_to_sample, &found_flag, &uniform_generator,
                                 best_points_to_sample.data());
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

  // check that the optimization succeeded on each output point
  std::vector<double> grad_ei(dim);
  OnePotentialSampleExpectedImprovementEvaluator ei_evaluator(*mock_gp_data.gaussian_process_ptr, mock_gp_data.best_so_far);
  bool configure_for_gradients = true;
  for (int i = 0; i < num_to_sample; ++i) {
    OnePotentialSampleExpectedImprovementEvaluator::StateType ei_state(ei_evaluator,
                                                                       best_points_to_sample.data() + i*dim,
                                                                       configure_for_gradients);
    ei_evaluator.ComputeGradExpectedImprovement(&ei_state, grad_ei.data());

    current_errors = 0;
    for (const auto& entry : grad_ei) {
      if (!CheckDoubleWithinRelative(entry, 0.0, tolerance)) {
        ++current_errors;
      }
    }
    total_errors += current_errors;

    FunctionValue estimate = estimation_policy->ComputeEstimate(*mock_gp_data.gaussian_process_ptr,
                                                                best_points_to_sample.data() + i*dim, i);
    mock_gp_data.gaussian_process_ptr->AddPointsToGP(best_points_to_sample.data() + i*dim,
                                                     &estimate.function_value,
                                                     &estimate.noise_variance, 1);
  }

  return total_errors;
}

}  // end unnamed namespace

int HeuristicExpectedImprovementOptimizationTest() {
  int total_errors = 0;
  int current_errors = 0;

  current_errors = HeuristicExpectedImprovementOptimizationTestCore(EstimationPolicyTypes::kConstantLiar);
  if (current_errors != 0) {
    OL_PARTIAL_FAILURE_PRINTF("ConstantLiar EI Optimization failed with %d errors\n", current_errors);
  } else {
    OL_PARTIAL_SUCCESS_PRINTF("ConstantLiar EI Optimization passed all tests\n");
  }
  total_errors += current_errors;

  current_errors = HeuristicExpectedImprovementOptimizationTestCore(EstimationPolicyTypes::kKrigingBeliever);
  if (current_errors != 0) {
    OL_PARTIAL_FAILURE_PRINTF("KrigingBeliever EI Optimization failed with %d errors\n", current_errors);
  } else {
    OL_PARTIAL_SUCCESS_PRINTF("KrigingBeliever EI Optimization passed all tests\n");
  }
  total_errors += current_errors;

  return total_errors;
}

}  // end namespace optimal_learning
