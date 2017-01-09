/*!
  \file gpp_model_selection_test.cpp
  \rst
  Routines to test the functions in gpp_model_selection.cpp.

  These tests verify LogMarginalLikelihoodEvaluator and LeaveOneOutLogLikelihoodEvaluator and their optimizers:

  1. Ping testing (verifying analytic gradient computation against finite difference approximations)

     a. Following gpp_covariance_test.cpp, we define classes (PingLogLikelihood, PingHessianLogLikelihood) for evaluating
        log likelihood + gradient or log likelihood gradient + hessian (derivs wrt hyperparameters).
     b. Ping for derivative accuracy (PingLogLikelihoodTest, which is general enough for gradients and hessian); this is
        for derivatives wrt hyperparameters.  These are for unit testing analytic derivatives.

  2. Gradient Descent + Newton unit tests: using polynomials and other simple fucntions with analytically known optima
     to verify that the optimizers are performing correctly.
  3. Hyperparameter optimization: we run hyperparameter optimization on toy problems using LML and LOO-CV likelihood
     as objective functions.  Convergence to at least local maxima is verified for both gradient descent and newton optimizers.
     These function as integration tests.
\endrst*/

// #define OL_VERBOSE_PRINT

#include "gpp_model_selection_test.hpp"

#include <cmath>

#include <algorithm>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

#include <boost/random/uniform_real.hpp>  // NOLINT(build/include_order)

#include "gpp_common.hpp"
#include "gpp_covariance.hpp"
#include "gpp_domain.hpp"
#include "gpp_exception.hpp"
#include "gpp_logging.hpp"
#include "gpp_math.hpp"
#include "gpp_mock_optimization_objective_functions.hpp"
#include "gpp_model_selection.hpp"
#include "gpp_random.hpp"
#include "gpp_optimization.hpp"
#include "gpp_optimizer_parameters.hpp"
#include "gpp_test_utils.hpp"

namespace optimal_learning {

namespace {  // tests for pinging log likelihood measures wrt their hyperparameters

/*!\rst
  Supports evaluating log likelihood functions and their gradients wrt hyperparameters.

  The gradient is taken wrt ``hyperparameters[n_hyper]``, so this is the ``input_matrix``, ``X_{d,i}`` with ``i`` unused.
  The other inputs to log marginal are not differentiated against, so they are taken as input and stored by the constructor.
\endrst*/
template <typename LogLikelihoodEvaluator, typename CovarianceClass>
class PingLogLikelihood final : public PingableMatrixInputVectorOutputInterface {
 public:
  using CovarianceType = CovarianceClass;

  PingLogLikelihood(const CovarianceClass& covariance, double const * restrict points_sampled, double const * restrict points_sampled_value, double const * restrict noise_variance, int dim, int num_sampled) OL_NONNULL_POINTERS
      : num_hyperparameters_(covariance.GetNumberOfHyperparameters()),
        gradients_already_computed_(false),
        log_likelihood_eval_(points_sampled, points_sampled_value, noise_variance, dim, num_sampled),
        grad_log_marginal_likelihood_(num_hyperparameters_) {
  }

  virtual void GetInputSizes(int * num_rows, int * num_cols) const noexcept override OL_NONNULL_POINTERS {
    *num_rows = num_hyperparameters_;
    *num_cols = 1;
  }

  virtual int GetGradientsSize() const noexcept override OL_WARN_UNUSED_RESULT {
    return num_hyperparameters_*GetOutputSize();
  }

  virtual int GetOutputSize() const noexcept override OL_WARN_UNUSED_RESULT {
    return 1;
  }

  virtual void EvaluateAndStoreAnalyticGradient(double const * restrict hyperparameters, double * restrict gradients) noexcept override OL_NONNULL_POINTERS_LIST(2) {
    if (gradients_already_computed_ == true) {
      OL_WARNING_PRINTF("WARNING: grad_covariance data pointer NOT nullptr.  Attempting to free...\n");
    }
    gradients_already_computed_ = true;

    CovarianceClass covariance_local(log_likelihood_eval_.dim(), hyperparameters[0], hyperparameters + 1);
    typename LogLikelihoodEvaluator::StateType log_likelihood_state(log_likelihood_eval_, covariance_local);
    log_likelihood_eval_.ComputeGradLogLikelihood(&log_likelihood_state, grad_log_marginal_likelihood_.data());

    if (gradients != nullptr) {
      std::copy(grad_log_marginal_likelihood_.begin(), grad_log_marginal_likelihood_.end(), gradients);
    }
  }

  virtual double GetAnalyticGradient(int row_index, int OL_UNUSED(column_index), int OL_UNUSED(output_index)) const override OL_WARN_UNUSED_RESULT {
    if (gradients_already_computed_ == false) {
      OL_THROW_EXCEPTION(OptimalLearningException, "PingLogLikelihood::GetAnalyticGradient() called BEFORE EvaluateAndStoreAnalyticGradient. NO DATA!");
    }

    return grad_log_marginal_likelihood_[row_index];
  }

  virtual void EvaluateFunction(double const * restrict hyperparameters, double * restrict function_values) const noexcept override OL_NONNULL_POINTERS {
    CovarianceClass covariance_local(log_likelihood_eval_.dim(), hyperparameters[0], hyperparameters + 1);
    typename LogLikelihoodEvaluator::StateType log_likelihood_state(log_likelihood_eval_, covariance_local);

    *function_values = log_likelihood_eval_.ComputeLogLikelihood(log_likelihood_state);
  }

 private:
  //! number of hyperparameters of the underlying covariance function
  int num_hyperparameters_;
  //! whether gradients been computed and stored--whether this class is ready for use
  bool gradients_already_computed_;

  //! log likelihood evaluator that is being tested (e.g., LogMarginalLikelihood, LeaveOneOutLogLikelihood)
  LogLikelihoodEvaluator log_likelihood_eval_;

  //! the gradient of the log marginal measure wrt hyperparameters of covariance
  std::vector<double> grad_log_marginal_likelihood_;

  OL_DISALLOW_DEFAULT_AND_COPY_AND_ASSIGN(PingLogLikelihood);
};

/*!\rst
  Supports evaluating log likelihood functions and their hessians wrt hyperparameters.

  The hessian is taken wrt ``hyperparameters[n_hyper]``, so this is the ``input_matrix``, ``X_{d,i}`` with ``i`` unused.
  The other inputs to log marginal are not differentiated against, so they are taken as input and stored by the constructor.

  Hessians are tested against gradients, so ``GetOutputSize`` returns ``num_hyperparameters``.
\endrst*/
template <typename LogLikelihoodEvaluator, typename CovarianceClass>
class PingHessianLogLikelihood final : public PingableMatrixInputVectorOutputInterface {
 public:
  using CovarianceType = CovarianceClass;

  PingHessianLogLikelihood(const CovarianceClass& covariance, double const * restrict points_sampled, double const * restrict points_sampled_value, double const * restrict noise_variance, int dim, int num_sampled) OL_NONNULL_POINTERS
      : num_hyperparameters_(covariance.GetNumberOfHyperparameters()),
        gradients_already_computed_(false),
        log_likelihood_eval_(points_sampled, points_sampled_value, noise_variance, dim, num_sampled),
        hessian_log_marginal_likelihood_(Square(num_hyperparameters_)) {
  }

  virtual void GetInputSizes(int * num_rows, int * num_cols) const noexcept override OL_NONNULL_POINTERS {
    *num_rows = num_hyperparameters_;
    *num_cols = 1;
  }

  virtual int GetGradientsSize() const noexcept override OL_WARN_UNUSED_RESULT {
    return num_hyperparameters_*GetOutputSize();
  }

  virtual int GetOutputSize() const noexcept override OL_WARN_UNUSED_RESULT {
    return num_hyperparameters_;
  }

  virtual void EvaluateAndStoreAnalyticGradient(double const * restrict hyperparameters, double * restrict gradients) noexcept override OL_NONNULL_POINTERS_LIST(2) {
    if (gradients_already_computed_ == true) {
      OL_WARNING_PRINTF("WARNING: grad_covariance data pointer NOT nullptr.  Attempting to free...\n");
    }
    gradients_already_computed_ = true;

    CovarianceClass covariance_local(log_likelihood_eval_.dim(), hyperparameters[0], hyperparameters + 1);
    typename LogLikelihoodEvaluator::StateType log_likelihood_state(log_likelihood_eval_, covariance_local);
    log_likelihood_eval_.ComputeHessianLogLikelihood(&log_likelihood_state, hessian_log_marginal_likelihood_.data());

    if (gradients != nullptr) {
      std::copy(hessian_log_marginal_likelihood_.begin(), hessian_log_marginal_likelihood_.end(), gradients);
    }
  }

  virtual double GetAnalyticGradient(int row_index, int OL_UNUSED(column_index), int output_index) const override OL_WARN_UNUSED_RESULT {
    if (gradients_already_computed_ == false) {
      OL_THROW_EXCEPTION(OptimalLearningException, "PingHessianLogLikelihood::GetAnalyticGradient() called BEFORE EvaluateAndStoreAnalyticGradient. NO DATA!");
    }

    return hessian_log_marginal_likelihood_[row_index*num_hyperparameters_ + output_index];
  }

  virtual void EvaluateFunction(double const * restrict hyperparameters, double * restrict function_values) const noexcept override OL_NONNULL_POINTERS {
    CovarianceClass covariance_local(log_likelihood_eval_.dim(), hyperparameters[0], hyperparameters + 1);
    typename LogLikelihoodEvaluator::StateType log_likelihood_state(log_likelihood_eval_, covariance_local);

    log_likelihood_eval_.ComputeGradLogLikelihood(&log_likelihood_state, function_values);
  }

 private:
  //! number of hyperparameters of the underlying covariance function
  int num_hyperparameters_;
  //! whether gradients been computed and stored--whether this class is ready for use
  bool gradients_already_computed_;

  //! log likelihood evaluator that is being tested (e.g., LogMarginalLikelihood, LeaveOneOutLogLikelihood)
  LogLikelihoodEvaluator log_likelihood_eval_;

  //! the hessian of the log marginal measure wrt hyperparameters of covariance
  std::vector<double> hessian_log_marginal_likelihood_;

  OL_DISALLOW_DEFAULT_AND_COPY_AND_ASSIGN(PingHessianLogLikelihood);
};

/*!\rst
  Pings the gradients (hyperparameters) of the likelihood functions 50 times with randomly generated test cases
  Covariance fcn to check against is a template parameter of PingLogLikelihood.

  \param
    :class_name: name of the log likelihood being tested (for logging)
    :num_hyperparameters: number of hyperparameters of the underlying covariance function
    :epsilon: coarse, fine ``h`` sizes to use in finite difference computation
    :tolerance_fine: desired amount of deviation from the exact rate
    :tolerance_coarse: maximum allowable abmount of deviation from the exact rate
    :input_output_ratio: for ``||analytic_gradient||/||input|| < input_output_ratio``, ping testing is not performed, see PingDerivative()
  \return
    number of ping test failures
\endrst*/
template <typename PingLogLikelihood>
OL_WARN_UNUSED_RESULT int PingLogLikelihoodTest(char const * class_name, int num_hyperparameters, double epsilon[2], double tolerance_fine, double tolerance_coarse, double input_output_ratio) {
  int total_errors = 0;
  int errors_this_iteration;
  const int dim = 3;

  int num_being_sampled = 0;
  int num_to_sample = 1;
  int num_sampled = 7;

  std::vector<double> hyperparameters(num_hyperparameters);

  MockExpectedImprovementEnvironment EI_environment;
  std::vector<double> noise_variance(num_sampled, 0.0);

  UniformRandomGenerator uniform_generator(3141);
  boost::uniform_real<double> uniform_double(3.0, 5.0);

  for (int i = 0; i < 50; ++i) {
    EI_environment.Initialize(dim, num_to_sample, num_being_sampled, num_sampled);

    for (int j = 0; j < num_hyperparameters; ++j) {
      hyperparameters[j] = uniform_double(uniform_generator.engine);
    }

    typename PingLogLikelihood::CovarianceType sqexp(EI_environment.dim, 1.0, 1.0);
    PingLogLikelihood log_likelihood_evaluator(sqexp, EI_environment.points_sampled(), EI_environment.points_sampled_value(), noise_variance.data(), EI_environment.dim, EI_environment.num_sampled);
    log_likelihood_evaluator.EvaluateAndStoreAnalyticGradient(hyperparameters.data(), nullptr);
    errors_this_iteration = PingDerivative(log_likelihood_evaluator, hyperparameters.data(), epsilon, tolerance_fine, tolerance_coarse, input_output_ratio);

    if (errors_this_iteration != 0) {
      OL_PARTIAL_FAILURE_PRINTF("on iteration %d\n", i);
    }
    total_errors += errors_this_iteration;
  }

  if (total_errors != 0) {
    OL_PARTIAL_FAILURE_PRINTF("%s hyperparameter gradient pings failed with %d errors\n", class_name, total_errors);
  } else {
    OL_PARTIAL_SUCCESS_PRINTF("%s hyperparameter gradient pings passed\n", class_name);
  }

  return total_errors;
}

}  // end unnamed namespace

int RunLogLikelihoodPingTests() {
  int total_errors = 0;
  int current_errors = 0;

  {
    double epsilon_log_marginal[2] = {1.0e-2, 2.0e-3};
    current_errors = PingLogLikelihoodTest<PingLogLikelihood<LogMarginalLikelihoodEvaluator, SquareExponentialSingleLength> >("Log Marginal Likelihood sqexp single", 2, epsilon_log_marginal, 5.0e-4, 1.0e-3, 1.0e-18);
    if (current_errors != 0) {
      OL_PARTIAL_FAILURE_PRINTF("pinging log likelihood failed with %d errors\n", current_errors);
    }
    total_errors += current_errors;
  }

  {
    double epsilon_log_marginal[2] = {1.0e-2, 2.0e-3};
    current_errors = PingLogLikelihoodTest<PingLogLikelihood<LogMarginalLikelihoodEvaluator, SquareExponential> >("Log Marginal Likelihood sqexp", 4, epsilon_log_marginal, 6.0e-3, 1.0e-3, 1.0e-18);
    if (current_errors != 0) {
      OL_PARTIAL_FAILURE_PRINTF("pinging log likelihood failed with %d errors\n", current_errors);
    }
    total_errors += current_errors;
  }

  {
    double epsilon_leave_one_out[2] = {1.0e-2, 2.0e-3};
    current_errors = PingLogLikelihoodTest<PingLogLikelihood<LeaveOneOutLogLikelihoodEvaluator, SquareExponentialSingleLength> >("Leave One Out Log Likelihood sqexp single", 2, epsilon_leave_one_out, 3.0e-4, 3.0e-4, 1.0e-18);
    if (current_errors != 0) {
      OL_PARTIAL_FAILURE_PRINTF("pinging leave one out failed with %d errors\n", current_errors);
    }
    total_errors += current_errors;
  }

  {
    double epsilon_leave_one_out[2] = {1.0e-2, 2.0e-3};
    current_errors = PingLogLikelihoodTest<PingLogLikelihood<LeaveOneOutLogLikelihoodEvaluator, SquareExponential> >("Leave One Out Log Likelihood sqexp", 4, epsilon_leave_one_out, 4.0e-3, 8.0e-4, 1.0e-18);
    if (current_errors != 0) {
      OL_PARTIAL_FAILURE_PRINTF("pinging leave one out failed with %d errors\n", current_errors);
    }
    total_errors += current_errors;
  }

  {
    double epsilon_log_marginal[2] = {1.0e-2, 2.0e-3};
    current_errors = PingLogLikelihoodTest<PingHessianLogLikelihood<LogMarginalLikelihoodEvaluator, SquareExponential> >("Log Marginal Likelihood Hessian sqexp", 4, epsilon_log_marginal, 3.0e-2, 3.0e-2, 1.0e-18);
    if (current_errors != 0) {
      OL_PARTIAL_FAILURE_PRINTF("pinging log marginal hessian failed with %d errors\n", current_errors);
    }
    total_errors += current_errors;
  }

  if (total_errors != 0) {
    OL_PARTIAL_FAILURE_PRINTF("Pinging GP functions failed with %d errors\n\n", total_errors);
  } else {
    OL_PARTIAL_SUCCESS_PRINTF("Pinging GP functions passed\n");
  }

  return total_errors;
}

namespace {  // tests for gradient descent and newton optimization of log likelihood measures wrt hyperparameters

/*!\rst
  Tests hyperparameter optimization.  Basic code flow:

  **SETUP**

    0. Pick a domain and specify hyperparameters (random) + covariance type (hyperparameters, CovarianceClass)
    1. Generate N random points in the domain
    2. Build a GP on the specified hyperparameters, incrementally generating function values for each of the N points

  **OPTIMIZE**

    3. Specify new, random, and different (~1 order of mag larger) hyperparameters than those used to generate the data
       (hyperparameters_wrong, covariance_wrong)
    4. Starting with covariance_wrong, optimize hyperparameters.

  **CHECK**

    5. Rerun optimization once more starting from the optimized values: VERIFY that no change occurs (within tolerance)
    6. VERIFY that the log marginal likelihood after optimization is better than the likelihood before optimization
    7. (REMOVED, FLAKY) CHECK that optimized hyperparameters are close to the hyperparameters used to generate the data
\endrst*/
template <typename LogLikelihoodEvaluator, typename CovarianceClass>
OL_WARN_UNUSED_RESULT int HyperparameterLikelihoodOptimizationTestCore(LogLikelihoodTypes objective_mode) {
  using DomainType = TensorProductDomain;
  using HyperparameterDomainType = TensorProductDomain;
  const int dim = 2;
  const int num_sampled = 40;

  double initial_likelihood;
  double final_likelihood;

  // gradient descent parameters
  const double gamma = 0.5;
  const double pre_mult = 0.5;
  const double max_relative_change = 0.02;
  double tolerance = 1.0e-10;
  if (objective_mode == LogLikelihoodTypes::kLeaveOneOutLogLikelihood) {
    tolerance = 1.0e-7;  // less accurate b/c current implementation uses matrix inverse for speed
  }
  const int max_gradient_descent_steps = 600;
  const int max_num_restarts = 5;
  const int num_steps_averaged = 0;
  GradientDescentParameters gd_parameters(1, max_gradient_descent_steps, max_num_restarts, num_steps_averaged, gamma, pre_mult, max_relative_change, tolerance);

  int total_errors = 0;
  int current_errors = 0;

  // covariance object that will be set with the wrong hyperparameters; used as an initial guess for optimization
  CovarianceClass covariance_wrong(dim, 1.0, 1.0);
  int num_hyperparameters = covariance_wrong.GetNumberOfHyperparameters();

  std::vector<double> hyperparameters_optimized(num_hyperparameters);  // optimized hyperparameters
  std::vector<double> hyperparameters_temp(num_hyperparameters);  // temp hyperparameters
  std::vector<double> hyperparameters_wrong(num_hyperparameters);  // wrong hyperparameters to start gradient descent

  // seed randoms
  UniformRandomGenerator uniform_generator(3141);
  boost::uniform_real<double> uniform_double_hyperparameter(1.0, 2.5);
  boost::uniform_real<double> uniform_double_lower_bound(-2.0, 0.5);
  boost::uniform_real<double> uniform_double_upper_bound(1.5, 2.7);

  ClosedInterval wrong_hyperparameter_range;
  if (objective_mode == LogLikelihoodTypes::kLogMarginalLikelihood) {
    wrong_hyperparameter_range = {2.5, 5.0};
  } else {
    wrong_hyperparameter_range = {0.3, 3.0};
  }
  boost::uniform_real<double> uniform_wrong_hyperparameters(wrong_hyperparameter_range.min, wrong_hyperparameter_range.max);
  FillRandomCovarianceHyperparameters(uniform_wrong_hyperparameters, &uniform_generator, &hyperparameters_wrong, &covariance_wrong);

  std::vector<ClosedInterval> hyperparameter_domain_bounds(num_hyperparameters, {1.0e-10, 1.0e10});
  HyperparameterDomainType hyperparameter_domain(hyperparameter_domain_bounds.data(), num_hyperparameters);

  std::vector<double> noise_variance(num_sampled, 0.1);
  MockGaussianProcessPriorData<DomainType> mock_gp_data(covariance_wrong, noise_variance, dim, num_sampled,
                                                        uniform_double_lower_bound, uniform_double_upper_bound,
                                                        uniform_double_hyperparameter, &uniform_generator);

  LogLikelihoodEvaluator log_likelihood_eval(mock_gp_data.gaussian_process_ptr->points_sampled().data(),
                                             mock_gp_data.gaussian_process_ptr->points_sampled_value().data(),
                                             mock_gp_data.gaussian_process_ptr->noise_variance().data(),
                                             dim, num_sampled);
  typename LogLikelihoodEvaluator::StateType log_likelihood_state(log_likelihood_eval, covariance_wrong);

  initial_likelihood = log_likelihood_eval.ComputeLogLikelihood(log_likelihood_state);
  OL_VERBOSE_PRINTF("initial likelihood: %.18E\n", initial_likelihood);

  RestartedGradientDescentHyperparameterOptimization(log_likelihood_eval, covariance_wrong, gd_parameters, hyperparameter_domain, hyperparameters_optimized.data());
  log_likelihood_state.SetHyperparameters(log_likelihood_eval, hyperparameters_optimized.data());
  final_likelihood = log_likelihood_eval.ComputeLogLikelihood(log_likelihood_state);

  // verify that convergence occurred
  covariance_wrong.SetHyperparameters(hyperparameters_optimized.data());
  gd_parameters.pre_mult *= 0.1;
  RestartedGradientDescentHyperparameterOptimization(log_likelihood_eval, covariance_wrong, gd_parameters, hyperparameter_domain, hyperparameters_temp.data());

  double norm_delta_hyperparameter;
  for (IdentifyType<decltype(hyperparameters_temp)>::type::size_type i = 0, size = hyperparameters_temp.size(); i < size; ++i) {
    hyperparameters_temp[i] -= hyperparameters_optimized[i];
  }
  norm_delta_hyperparameter = VectorNorm(hyperparameters_temp.data(), hyperparameters_temp.size());
  current_errors = 0;
  if (!CheckDoubleWithin(norm_delta_hyperparameter, 0.0, 1.0e-6)) {
    ++current_errors;
  }
  if (current_errors != 0) {
    OL_PARTIAL_FAILURE_PRINTF("gradient descent did not full converge: hyperparameters still changed by (RMS): %.18E\n", norm_delta_hyperparameter);
  }
  total_errors += current_errors;

  current_errors = 0;
  if (final_likelihood <= initial_likelihood) {
    ++current_errors;  // expect improvement from optimization
  }
  if (final_likelihood >= 0.0) {
    ++current_errors;  // must be negative
  }
  if (current_errors != 0) {
    OL_PARTIAL_FAILURE_PRINTF("final likelihood = %.18E is worse than initial likelihood = %.18E\n", final_likelihood, initial_likelihood);
  }
  total_errors += current_errors;

  // check that hyperparameter gradients are small
  std::vector<double> grad_log_marginal(num_hyperparameters);
  log_likelihood_eval.ComputeGradLogLikelihood(&log_likelihood_state, grad_log_marginal.data());

#ifdef OL_VERBOSE_PRINT
  OL_VERBOSE_PRINTF("grad log marginal: ");
  PrintMatrix(grad_log_marginal.data(), 1, grad_log_marginal.size());
#endif

  current_errors = 0;
  for (const auto& entry : grad_log_marginal) {
    if (objective_mode == LogLikelihoodTypes::kLogMarginalLikelihood) {
      if (!CheckDoubleWithinRelative(entry, 0.0, 5.0e-11)) {
        ++current_errors;
      }
    } else {
      if (!CheckDoubleWithinRelative(entry, 0.0, 5.0e-8)) {
        ++current_errors;
      }
    }
  }
  total_errors += current_errors;

  // would check that optimal hyperparameters are close to their initial values, but that does not seem to be a well grounded test.
  // may revisit getting better testing for this function in the future.
  // const double hyperparameter_tolerance = 0.5;
  // current_errors = 0;
  // for (IdentifyType<decltype(hyperparameters)>::type::size_type i = 0; i < hyperparameters.size(); ++i) {
  //   current_errors += CheckDoubleWithinRelative(hyperparameters_optimized[i], hyperparameters[i], hyperparameter_tolerance) == false;
  // }
  // if (current_errors != 0) {
  //   OL_PARTIAL_FAILURE_PRINTF("optimized hyperparameters are NOT close to true values!\n");
  // }
  // total_errors += current_errors;

  return total_errors;
}

/*!\rst
  Tests Newton hyperparameter optimization.  Basic code flow:

  **SETUP**

    0. Pick a domain and specify hyperparameters (random) + covariance type (hyperparameters, CovarianceClass)
    1. Generate N random points in the domain
    2. Build a GP on the specified hyperparameters, incrementally generating function values for each of the N points

  **OPTIMIZE**

    3. Specify new, random, and different (~1 order of mag larger) hyperparameters than those used to generate the data
       (hyperparameters_wrong, covariance_wrong)
    4. Starting with covariance_wrong, optimize hyperparameters.

  **CHECK**

    5. Rerun optimization once more starting from the optimized values: VERIFY that no change occurs (within tolerance).  Set very large time_factor for this check
    6. Verify that the log marginal likelihood after optimization is better than the likelihood before optimization
    7. Verify that the gradient is below newton tolerance.
    8. TODO(GH-121): Verify that the eigenvalues of the Hessian are all negative (=> maxima)
\endrst*/
template <typename LogLikelihoodEvaluator, typename CovarianceClass>
OL_WARN_UNUSED_RESULT int HyperparameterLikelihoodNewtonOptimizationTestCore(LogLikelihoodTypes OL_UNUSED(objective_mode)) {
  using DomainType = TensorProductDomain;
  using HyperparameterDomainType = TensorProductDomain;
  const int num_sampled = 45;
  const int dim = 2;

  double initial_likelihood;
  double final_likelihood;

  // gradient descent parameters
  const double gamma = 1.1;
  const double pre_mult = 1.0e-1;
  const double max_relative_change = 1.0;
  const double tolerance = 1.0e-13;
  const int max_newton_steps = 1000;
  NewtonParameters newton_parameters(1, max_newton_steps, gamma, pre_mult, max_relative_change, tolerance);

  int total_errors = 0;
  int current_errors = 0;

  // covariance object that will be set with the wrong hyperparameters; used as an initial guess for optimization
  CovarianceClass covariance_wrong(dim, 1.0, 1.0);
  int num_hyperparameters = covariance_wrong.GetNumberOfHyperparameters();

  std::vector<double> hyperparameters_optimized(num_hyperparameters);  // optimized hyperparameters
  std::vector<double> hyperparameters_temp(num_hyperparameters);  // temp hyperparameters
  std::vector<double> hyperparameters_wrong(num_hyperparameters);  // wrong hyperparameters to start gradient descent

  // seed randoms
  UniformRandomGenerator uniform_generator(5762);
  boost::uniform_real<double> uniform_double_hyperparameter(1.0, 2.5);
  boost::uniform_real<double> uniform_double_lower_bound(-2.0, 0.5);
  boost::uniform_real<double> uniform_double_upper_bound(2.0, 3.5);

  boost::uniform_real<double> uniform_double_for_wrong_hyperparameter(10.0, 30.0);
  FillRandomCovarianceHyperparameters(uniform_double_for_wrong_hyperparameter, &uniform_generator, &hyperparameters_wrong, &covariance_wrong);
  std::vector<ClosedInterval> hyperparameter_domain_bounds(num_hyperparameters, {1.0e-10, 1.0e10});
  HyperparameterDomainType hyperparameter_domain(hyperparameter_domain_bounds.data(), num_hyperparameters);

  std::vector<double> noise_variance(num_sampled, 0.1);
  MockGaussianProcessPriorData<DomainType> mock_gp_data(covariance_wrong, noise_variance, dim, num_sampled,
                                                        uniform_double_lower_bound, uniform_double_upper_bound,
                                                        uniform_double_hyperparameter, &uniform_generator);

  LogLikelihoodEvaluator log_likelihood_eval(mock_gp_data.gaussian_process_ptr->points_sampled().data(),
                                             mock_gp_data.gaussian_process_ptr->points_sampled_value().data(),
                                             mock_gp_data.gaussian_process_ptr->noise_variance().data(),
                                             dim, num_sampled);
  typename LogLikelihoodEvaluator::StateType log_likelihood_state(log_likelihood_eval, covariance_wrong);

  initial_likelihood = log_likelihood_eval.ComputeLogLikelihood(log_likelihood_state);
  OL_VERBOSE_PRINTF("initial likelihood: %.18E\n", initial_likelihood);

  total_errors += NewtonHyperparameterOptimization(log_likelihood_eval, covariance_wrong, newton_parameters, hyperparameter_domain, hyperparameters_optimized.data());
  covariance_wrong.SetHyperparameters(hyperparameters_optimized.data());
  log_likelihood_state.SetHyperparameters(log_likelihood_eval, hyperparameters_optimized.data());
  final_likelihood = log_likelihood_eval.ComputeLogLikelihood(log_likelihood_state);
#ifdef OL_VERBOSE_PRINT
  OL_VERBOSE_PRINTF("final likelihood: %.18E\n", final_likelihood);
  PrintMatrix(hyperparameters_optimized.data(), 1, num_hyperparameters);
#endif

  // check that hyperparameter gradients are small
  std::vector<double> grad_log_marginal(num_hyperparameters);
  log_likelihood_eval.ComputeGradLogLikelihood(&log_likelihood_state, grad_log_marginal.data());

#ifdef OL_VERBOSE_PRINT
  OL_VERBOSE_PRINTF("grad log marginal: ");
  PrintMatrix(grad_log_marginal.data(), 1, grad_log_marginal.size());
#endif

  current_errors = 0;
  for (const auto& entry : grad_log_marginal) {
    if (!CheckDoubleWithinRelative(entry, 0.0, 1.0e-13)) {
      ++current_errors;
    }
  }
  total_errors += current_errors;

  // verify that convergence occurred
  // set very aggressive time_factor
  newton_parameters.time_factor = 1.0e30;
  newton_parameters.gamma = 10.0;
  newton_parameters.max_num_steps = 10;
  total_errors += NewtonHyperparameterOptimization(log_likelihood_eval, covariance_wrong, newton_parameters, hyperparameter_domain, hyperparameters_temp.data());
#ifdef OL_VERBOSE_PRINT
  PrintMatrix(hyperparameters_optimized.data(), 1, num_hyperparameters);
#endif

  double norm_delta_hyperparameter;
  current_errors = 0;
  for (IdentifyType<decltype(hyperparameters_temp)>::type::size_type i = 0, size = hyperparameters_temp.size(); i < size; ++i) {
    hyperparameters_temp[i] -= hyperparameters_optimized[i];
  }
  norm_delta_hyperparameter = VectorNorm(hyperparameters_temp.data(), hyperparameters_temp.size());
  if (!CheckDoubleWithin(norm_delta_hyperparameter, 0.0, 1.0e-12)) {
    ++current_errors;
  }
  if (current_errors != 0) {
    OL_PARTIAL_FAILURE_PRINTF("newton did not full converge: hyperparameters still changed by (RMS): %.18E\n", norm_delta_hyperparameter);
  }
  total_errors += current_errors;

  current_errors = 0;
  if (final_likelihood <= initial_likelihood) {
    ++current_errors;  // expect improvement from optimization
  }
  if (final_likelihood >= 0.0) {
    ++current_errors;  // must be negative
  }
  if (current_errors != 0) {
    OL_PARTIAL_FAILURE_PRINTF("final likelihood = %.18E is worse than initial likelihood = %.18E\n", final_likelihood, initial_likelihood);
  }
  total_errors += current_errors;

  return total_errors;
}

/*!\rst
  Tests multistarted Newton optimization for hyperparameters.
  Compares result to newton optimization called from an initial guess very near the optimal solution.

  Tests Newton hyperparameter optimization.  Basic code flow:

  **SETUP**

    0. Pick a domain and specify hyperparameters (random) + covariance type (hyperparameters, CovarianceClass)
    1. Generate N random points in the domain
    2. Build a GP on the specified hyperparameters, incrementally generating function values for each of the N points

  **OPTIMIZE**

    3. Multistarted Newton optimization with multithreading enabled

  **CHECK**

    4. Rerun optimization once more starting from the values in step 0; this should be very near the real solution.
    5. Verify that these hyperparameters and the results of multistart Newton are the same
    6. Verify that the log marginal likelihood after multistart optimization is better than the likelihood at the
       hyperparameters from step 0 (since these are not optimal)
\endrst*/
template <typename LogLikelihoodEvaluator, typename CovarianceClass>
OL_WARN_UNUSED_RESULT int MultistartHyperparameterLikelihoodNewtonOptimizationTestCore(LogLikelihoodTypes OL_UNUSED(objective_mode)) {
  using DomainType = TensorProductDomain;
  using HyperparameterDomainType = TensorProductDomain;
  const int num_sampled = 42;
  const int dim = 2;

  double initial_likelihood;
  double final_likelihood;

  // gradient descent parameters
  const double gamma = 1.1;
  const double pre_mult = 1.0e-1;
  const double max_relative_change = 1.0;
  const double tolerance = 1.0e-14;
  const int max_newton_steps = 100;
  const int num_multistarts = 16;
  NewtonParameters newton_parameters(num_multistarts, max_newton_steps, gamma, pre_mult, max_relative_change, tolerance);

  const int max_num_threads = 4;
  ThreadSchedule thread_schedule(max_num_threads, omp_sched_dynamic);

  int total_errors = 0;
  int current_errors = 0;

  // seed randoms
  UniformRandomGenerator uniform_generator(5762);
  boost::uniform_real<double> uniform_double_hyperparameter(1.0, 2.5);
  boost::uniform_real<double> uniform_double_lower_bound(-2.0, 0.5);
  boost::uniform_real<double> uniform_double_upper_bound(2.0, 3.5);

  std::vector<double> noise_variance(num_sampled, 0.1);
  MockGaussianProcessPriorData<DomainType> mock_gp_data(CovarianceClass(dim, 1.0, 1.0), noise_variance, dim,
                                                        num_sampled, uniform_double_lower_bound,
                                                        uniform_double_upper_bound,
                                                        uniform_double_hyperparameter, &uniform_generator);
  int num_hyperparameters = mock_gp_data.covariance_ptr->GetNumberOfHyperparameters();

  std::vector<double> hyperparameters_truth(num_hyperparameters);  // truth hyperparameters
  std::vector<double> hyperparameters_optimized(num_hyperparameters);  // optimized hyperparameters
  std::vector<double> hyperparameters_temp(num_hyperparameters);  // temp hyperparameters

  // set up domain; allows initial guesses to range over [0.01, 10]
  std::vector<ClosedInterval> hyperparameter_log_domain_bounds(num_hyperparameters, {-2.0, 1.0});
  std::vector<ClosedInterval> hyperparameter_domain_bounds(hyperparameter_log_domain_bounds);
  for (auto& interval : hyperparameter_domain_bounds) {
    interval = {std::pow(10.0, interval.min), std::pow(10.0, interval.max)};
  }
  HyperparameterDomainType hyperparameter_domain(hyperparameter_domain_bounds.data(), num_hyperparameters);

  LogLikelihoodEvaluator log_likelihood_eval(mock_gp_data.gaussian_process_ptr->points_sampled().data(),
                                             mock_gp_data.gaussian_process_ptr->points_sampled_value().data(),
                                             mock_gp_data.gaussian_process_ptr->noise_variance().data(),
                                             dim, num_sampled);
  typename LogLikelihoodEvaluator::StateType log_likelihood_state(log_likelihood_eval, *mock_gp_data.covariance_ptr);

  initial_likelihood = log_likelihood_eval.ComputeLogLikelihood(log_likelihood_state);
  OL_VERBOSE_PRINTF("initial likelihood: %.18E\n", initial_likelihood);

  bool found_flag = false;
  MultistartNewtonHyperparameterOptimization(log_likelihood_eval, *mock_gp_data.covariance_ptr,
                                             newton_parameters, hyperparameter_log_domain_bounds.data(),
                                             thread_schedule, &found_flag, &uniform_generator,
                                             hyperparameters_optimized.data());
  if (!found_flag) {
    ++total_errors;
  }

  log_likelihood_state.SetHyperparameters(log_likelihood_eval, hyperparameters_optimized.data());
  final_likelihood = log_likelihood_eval.ComputeLogLikelihood(log_likelihood_state);
#ifdef OL_VERBOSE_PRINT
  OL_VERBOSE_PRINTF("final likelihood: %.18E\n", final_likelihood);
  PrintMatrix(hyperparameters_optimized.data(), 1, num_hyperparameters);
#endif

  // check that hyperparameter gradients are small
  std::vector<double> grad_log_marginal(num_hyperparameters);
  log_likelihood_eval.ComputeGradLogLikelihood(&log_likelihood_state, grad_log_marginal.data());

#ifdef OL_VERBOSE_PRINT
  OL_VERBOSE_PRINTF("grad log marginal: ");
  PrintMatrix(grad_log_marginal.data(), 1, grad_log_marginal.size());
#endif

  current_errors = 0;
  for (const auto& entry : grad_log_marginal) {
    if (!CheckDoubleWithinRelative(entry, 0.0, 1.0e-13)) {
      ++current_errors;
    }
  }
  total_errors += current_errors;

  // verify that convergence occurred, start from the hyperparameters used to generate data (real solution should be nearby)
  total_errors += NewtonHyperparameterOptimization(log_likelihood_eval, *mock_gp_data.covariance_ptr, newton_parameters, hyperparameter_domain, hyperparameters_truth.data());
#ifdef OL_VERBOSE_PRINT
  PrintMatrix(hyperparameters_truth.data(), 1, num_hyperparameters);
#endif

  double norm_delta_hyperparameter;
  current_errors = 0;
  for (IdentifyType<decltype(hyperparameters_temp)>::type::size_type i = 0, size = hyperparameters_truth.size(); i < size; ++i) {
    hyperparameters_temp[i] = hyperparameters_truth[i] - hyperparameters_optimized[i];
  }
  norm_delta_hyperparameter = VectorNorm(hyperparameters_temp.data(), hyperparameters_temp.size());
  if (!CheckDoubleWithin(norm_delta_hyperparameter, 0.0, 1.0e-12)) {
    ++current_errors;
  }
  if (current_errors != 0) {
    OL_PARTIAL_FAILURE_PRINTF("multistart newton did not find the optimal solution: hyperparameters differ by (RMS): %.18E\n", norm_delta_hyperparameter);
  }
  total_errors += current_errors;

  current_errors = 0;
  if (final_likelihood <= initial_likelihood) {
    ++current_errors;  // expect improvement from optimization
  }
  if (final_likelihood >= 0.0) {
    ++current_errors;  // must be negative
  }
  if (current_errors != 0) {
    OL_PARTIAL_FAILURE_PRINTF("final likelihood = %.18E is worse than initial likelihood = %.18E\n", final_likelihood, initial_likelihood);
  }
  total_errors += current_errors;

  // now check that if we call Newton again with the optimal solution, it will fail to find a new solution
  // and found_flag will be FALSE
  {
    newton_parameters.num_multistarts = 8;
    std::vector<double> initial_guesses(num_hyperparameters*newton_parameters.num_multistarts);
    hyperparameter_domain.GenerateUniformPointsInDomain(newton_parameters.num_multistarts,
                                                        &uniform_generator, initial_guesses.data());

    // insert optimal solution into initial_guesses
    std::copy(hyperparameters_optimized.begin(), hyperparameters_optimized.end(), initial_guesses.begin());

    // build state vector
    std::vector<typename LogLikelihoodEvaluator::StateType> log_likelihood_state_vector;
    SetupLogLikelihoodState(log_likelihood_eval, *mock_gp_data.covariance_ptr,
                            thread_schedule.max_num_threads, &log_likelihood_state_vector);

    OptimizationIOContainer io_container(log_likelihood_state_vector[0].GetProblemSize());
    InitializeBestKnownPoint(log_likelihood_eval, initial_guesses.data(), num_hyperparameters,
                             newton_parameters.num_multistarts,
                             log_likelihood_state_vector.data(), &io_container);

    io_container.found_flag = true;  // want to see that this flag is flipped to false

    NewtonOptimizer<LogLikelihoodEvaluator, TensorProductDomain> newton_opt;
    MultistartOptimizer<NewtonOptimizer<LogLikelihoodEvaluator, TensorProductDomain> > multistart_optimizer;
    multistart_optimizer.MultistartOptimize(newton_opt, log_likelihood_eval, newton_parameters,
                                            hyperparameter_domain, thread_schedule,
                                            initial_guesses.data(), newton_parameters.num_multistarts,
                                            log_likelihood_state_vector.data(),
                                            nullptr, &io_container);

    found_flag = io_container.found_flag;
    std::copy(io_container.best_point.begin(), io_container.best_point.end(), hyperparameters_temp.begin());

    // found_flag should be false b/c optimal solution is in the starting guesses
    if (found_flag) {
      ++total_errors;
    }

    // verify that the solution was not altered
    for (IdentifyType<decltype(hyperparameters_temp)>::type::size_type i = 0, size = hyperparameters_truth.size(); i < size; ++i) {
      if (!CheckDoubleWithin(hyperparameters_temp[i], hyperparameters_optimized[i], 0.0)) {
        ++total_errors;
      }
    }
  }

  return total_errors;
}

}  // end unnamed namespace

int HyperparameterLikelihoodOptimizationTest(OptimizerTypes optimizer_type, LogLikelihoodTypes objective_mode) {
  switch (optimizer_type) {
    case OptimizerTypes::kGradientDescent: {
      switch (objective_mode) {
        case LogLikelihoodTypes::kLogMarginalLikelihood: {
          return HyperparameterLikelihoodOptimizationTestCore<LogMarginalLikelihoodEvaluator, SquareExponential>(objective_mode);
        }
        case LogLikelihoodTypes::kLeaveOneOutLogLikelihood: {
          return HyperparameterLikelihoodOptimizationTestCore<LeaveOneOutLogLikelihoodEvaluator, SquareExponential>(objective_mode);
        }
        default: {
          OL_ERROR_PRINTF("%s: INVALID objective_mode choice: %d\n", OL_CURRENT_FUNCTION_NAME, objective_mode);
          return 1;
        }
      }  // end switch over objective_mode
    }  // end case kGradientDescent
    case OptimizerTypes::kNewton: {
      switch (objective_mode) {
        case LogLikelihoodTypes::kLogMarginalLikelihood: {
          // check base newton optimization
          int current_errors = 0;
          int total_errors = 0;
          current_errors = HyperparameterLikelihoodNewtonOptimizationTestCore<LogMarginalLikelihoodEvaluator, SquareExponential>(objective_mode);
          total_errors += current_errors;

          // check multistarted newton
          current_errors = MultistartHyperparameterLikelihoodNewtonOptimizationTestCore<LogMarginalLikelihoodEvaluator, SquareExponential>(objective_mode);
          total_errors += current_errors;
          return total_errors;
        }
        default: {
          OL_ERROR_PRINTF("%s: INVALID objective_mode choice: %d\n", OL_CURRENT_FUNCTION_NAME, objective_mode);
          return 1;
        }
      }  // end switch over objective_mode
    }  // end case kNewton
    default: {
      OL_ERROR_PRINTF("%s: INVALID optimizer_type choice: %d\n", OL_CURRENT_FUNCTION_NAME, optimizer_type);
      return 1;
    }
  }  // end switch over optimizer_type
}

int EvaluateLogLikelihoodAtPointListTest() {
  using DomainType = TensorProductDomain;
  using HyperparameterDomainType = TensorProductDomain;
  const int dim = 3;

  int total_errors = 0;

  // grid search parameters
  int num_grid_search_points = 100000;

  // random number generators
  UniformRandomGenerator uniform_generator(314);
  boost::uniform_real<double> uniform_double_hyperparameter(0.4, 1.3);
  boost::uniform_real<double> uniform_double_lower_bound(-2.0, 0.5);
  boost::uniform_real<double> uniform_double_upper_bound(2.0, 3.5);

  static const int kMaxNumThreads = 4;
  ThreadSchedule thread_schedule(kMaxNumThreads, omp_sched_guided);

  int num_sampled = 11;  // arbitrary
  std::vector<double> noise_variance(num_sampled, 0.002);
  MockGaussianProcessPriorData<DomainType> mock_gp_data(SquareExponential(dim, 1.0, 1.0), noise_variance,
                                                        dim, num_sampled, uniform_double_lower_bound,
                                                        uniform_double_upper_bound, uniform_double_hyperparameter,
                                                        &uniform_generator);

  using LogLikelihoodEvaluator = LogMarginalLikelihoodEvaluator;
  LogLikelihoodEvaluator log_marginal_eval(mock_gp_data.gaussian_process_ptr->points_sampled().data(),
                                           mock_gp_data.gaussian_process_ptr->points_sampled_value().data(),
                                           mock_gp_data.gaussian_process_ptr->noise_variance().data(),
                                           dim, num_sampled);
  int num_hyperparameters = mock_gp_data.covariance_ptr->GetNumberOfHyperparameters();
  std::vector<ClosedInterval> hyperparameter_log_domain_bounds(num_hyperparameters, {-2.0, 1.0});
  HyperparameterDomainType hyperparameter_log_domain(hyperparameter_log_domain_bounds.data(), num_hyperparameters);

  std::vector<double> grid_search_best_point(num_hyperparameters);
  std::vector<double> function_values(num_grid_search_points);
  std::vector<double> initial_guesses(num_hyperparameters*num_grid_search_points);
  num_grid_search_points = hyperparameter_log_domain.GenerateUniformPointsInDomain(num_grid_search_points,
                                                                                   &uniform_generator,
                                                                                   initial_guesses.data());
  for (auto& point : initial_guesses) {
    point = std::pow(10.0, point);
  }

  // domain in linear-space
  std::vector<ClosedInterval> hyperparameter_domain_linearspace_bounds(hyperparameter_log_domain_bounds);
  for (auto& interval : hyperparameter_domain_linearspace_bounds) {
    interval = {std::pow(10.0, interval.min), std::pow(10.0, interval.max)};
  }
  HyperparameterDomainType hyperparameter_domain_linearspace(hyperparameter_domain_linearspace_bounds.data(), num_hyperparameters);

  bool found_flag = false;
  EvaluateLogLikelihoodAtPointList(log_marginal_eval, *mock_gp_data.covariance_ptr,
                                   hyperparameter_domain_linearspace, thread_schedule,
                                   initial_guesses.data(), num_grid_search_points, &found_flag,
                                   function_values.data(), grid_search_best_point.data());

  if (!found_flag) {
    ++total_errors;
  }

  // find the max function_value and the index at which it occurs
  auto max_value_ptr = std::max_element(function_values.begin(), function_values.end());
  auto max_index = std::distance(function_values.begin(), max_value_ptr);

  // check that EvaluateLogLikelihoodAtPointList found the right point
  for (int i = 0; i < num_hyperparameters; ++i) {
    if (!CheckDoubleWithin(grid_search_best_point[i], initial_guesses[max_index*num_hyperparameters + i], 0.0)) {
      ++total_errors;
    }
  }

  // now check multi-threaded & single threaded give the same result
  {
    std::vector<double> grid_search_best_point_single_thread(num_hyperparameters);
    std::vector<double> function_values_single_thread(num_grid_search_points);
    ThreadSchedule single_thread_schedule(1, omp_sched_static);
    found_flag = false;
    EvaluateLogLikelihoodAtPointList(log_marginal_eval, *mock_gp_data.covariance_ptr,
                                     hyperparameter_domain_linearspace, single_thread_schedule,
                                     initial_guesses.data(), num_grid_search_points,
                                     &found_flag, function_values_single_thread.data(),
                                     grid_search_best_point_single_thread.data());

    if (!found_flag) {
      ++total_errors;
    }

    // check against multi-threaded result matches single
    for (int i = 0; i < num_hyperparameters; ++i) {
      if (!CheckDoubleWithin(grid_search_best_point[i], grid_search_best_point_single_thread[i], 0.0)) {
        ++total_errors;
      }
    }

    // check all function values match too
    for (int i = 0; i < num_grid_search_points; ++i) {
      if (!CheckDoubleWithin(function_values[i], function_values_single_thread[i], 0.0)) {
        ++total_errors;
      }
    }
  }

  return total_errors;
}

}  // end namespace optimal_learning
