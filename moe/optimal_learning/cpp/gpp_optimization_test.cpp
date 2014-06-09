/*!
  \file gpp_optimization_test.cpp
  \rst
  Unit tests for the optimization algorithms in gpp_optimization.hpp.  Currently we have tests for:

  1. restarted gradient descent (which uses gradient descent)
  2. newton

  And each optimizer is tested against:

  1. a quadratic objective function

  performing:

  1. Unconstrained optimization: optimal point is set away from domain boundaries
  2. Constrained optimization: optimal point is set outside the boundaries

  on:

  1. tensor product domains

  Generally the tests verify that:

  1. The expected optimum is actually an optimum (gradients are small).
  2. When started from the optimum value, the optimizer does not move away from it
  3. When started away from the optimum value, the optimizer finds it.

  Where "finds it" is checked by seeing whether the point returned by the optimizer is within tolerance of the true
  optimum and whether the gradients are within tolerance of 0.

  This is a little tricky for the constrained case b/c we are currently assuming we can compute the location of the
  true optimum directly... which may not always be possible.

  TODO(eliu): (GH-146) we should add some more complex objective functions (higher order
  polynomials and/or transcendentals) and simplex domains.
  We also need a more general way of testing constrained optimization since it is not
  always possible to directly compute the location of the optima.

  TODO(eliu): (GH-146) we have quite a bit of code duplication here.  For the most part, the
  only difference is how we set up the optimizers in the beginning of each test function.
  This duplication could be reduced by encapsulating the optimizers in classes and then
  templating the testing functions on the optimizer type.

\endrst*/

// #define OL_VERBOSE_PRINT

#include "gpp_optimization_test.hpp"

#include <cmath>

#include <algorithm>
#include <vector>

#include "gpp_common.hpp"
#include "gpp_domain.hpp"
#include "gpp_logging.hpp"
#include "gpp_mock_optimization_objective_functions.hpp"
#include "gpp_optimization.hpp"
#include "gpp_optimization_parameters.hpp"
#include "gpp_test_utils.hpp"

namespace optimal_learning {

namespace {  // mock objective functions and tests for various optimizers

using optimal_learning::PolynomialEvaluator;
using optimal_learning::Square;

/*!\rst
  Class to evaluate the function ``f(x_1,...,x_{dim}) = -\sum_i (x_i - s_i)^2, i = 1..dim``.
  This is a simple quadratic form with maxima at ``(s_1, ..., s_{dim})``.
\endrst*/
class SimpleQuadraticEvaluator final : public PolynomialEvaluator {
 public:
  SimpleQuadraticEvaluator(double const * restrict maxima_point, int dim_in) : dim_(dim_in), maxima_point_(maxima_point, maxima_point + dim_) {
  }

  virtual int dim() const noexcept override OL_PURE_FUNCTION OL_WARN_UNUSED_RESULT {
    return dim_;
  }

  virtual double GetOptimumValue() const noexcept OL_PURE_FUNCTION OL_WARN_UNUSED_RESULT {
    return 0.0;
  }

  virtual void GetOptimumPoint(double * restrict point) const noexcept OL_NONNULL_POINTERS {
    std::copy(maxima_point_.begin(), maxima_point_.end(), point);
  }

  /*!\rst
    Computes the quadratic objective function (see class description).

    \param
      :quadratic_dummy_state: properly configured state oboject (set up with the point at which to evaluate the objective)
    \return
      objective function value
  \endrst*/
  virtual double ComputeObjectiveFunction(StateType * quadratic_dummy_state) const noexcept override OL_NONNULL_POINTERS OL_WARN_UNUSED_RESULT {
    double sum = 0.0;
    for (int i = 0; i < dim_; ++i) {
      sum -= Square(quadratic_dummy_state->current_point[i] - maxima_point_[i]);
    }
    return sum;
  }

  /*!\rst
    Computes partial derivatives of the objective function wrt the point x (contained in quadratic_dummy_state; see class description).

    \param
      :quadratic_dummy_state[1]: properly configured state oboject (set up with the point at which to evaluate the objective)
    \output
      :grad_objective[dim]: gradient of objective function wrt each dimension of the point contained in dummy_state.
  \endrst*/
  virtual void ComputeGradObjectiveFunction(StateType * quadratic_dummy_state, double * restrict grad_objective) const noexcept override OL_NONNULL_POINTERS {
    for (int i = 0; i < dim_; ++i) {
      grad_objective[i] = -2.0*(quadratic_dummy_state->current_point[i] - maxima_point_[i]);
    }
  }

  /*!\rst
    Computes Hessian matrix of the objective function wrt the point x (contained in quadratic_dummy_state; see class description).
    This matrix is symmetric.  It is also negative definite near maxima of the log marginal.

    \param
      :quadratic_dummy_state[1]: properly configured state oboject (set up with the point at which to evaluate the objective)
    \output
      :hessian_objective[dim][dim]: (i,j)th entry is \mixpderiv{f(x_i)}{x_i}{x_j}, for our quadratic objective, f
  \endrst*/
  virtual void ComputeHessianObjectiveFunction(StateType * OL_UNUSED(quadratic_dummy_state), double * restrict hessian_objective) const OL_NONNULL_POINTERS {
    std::fill(hessian_objective, hessian_objective + dim_*dim_, 0.0);
    for (int i = 0; i < dim_; ++i) {
      hessian_objective[i] = -2.0;
      hessian_objective += dim_;
    }
  }

  OL_DISALLOW_DEFAULT_AND_COPY_AND_ASSIGN(SimpleQuadraticEvaluator);

 private:
  int dim_;
  std::vector<double> maxima_point_;
};

/*!\rst
  Test gradient descent's ability to optimize the function represented by MockEvaluator in an unconstrained setting.

  \return
    number of test failures (invalid results, non-convergence, etc.)
\endrst*/
template <typename MockEvaluator>
OL_WARN_UNUSED_RESULT int MockObjectiveGradientDescentOptimizationTestCore() {
  using DomainType = TensorProductDomain;
  const int dim = 3;

  double initial_objective;
  double final_objective;

  // gradient descent parameters
  const double gamma = 0.9;
  const double pre_mult = 1.0;
  const double max_relative_change = 0.8;
  const double tolerance = 1.0e-12;
  const int max_gradient_descent_steps = 1000;
  const int max_num_restarts = 10;
  GradientDescentParameters gd_parameters(1, max_gradient_descent_steps, max_num_restarts, gamma, pre_mult, max_relative_change, tolerance);

  int total_errors = 0;
  int current_errors = 0;

  std::vector<ClosedInterval> domain_bounds(dim, {-1.0, 1.0});
  DomainType domain(domain_bounds.data(), dim);

  std::vector<double> maxima_point_input(dim, 0.5);

  std::vector<double> wrong_point(dim, 0.2);

  std::vector<double> point_optimized(dim);
  std::vector<double> temp_point(dim);

  MockEvaluator objective_eval(maxima_point_input.data(), dim);

  // get optima data
  objective_eval.GetOptimumPoint(temp_point.data());
  const std::vector<double> maxima_point(temp_point);
  const double maxima_value = objective_eval.GetOptimumValue();

  // build state, setting initial point to maxima_point
  typename MockEvaluator::StateType objective_state(objective_eval, maxima_point.data());

  // verify that claimed optima is actually an optima
  final_objective = objective_eval.ComputeObjectiveFunction(&objective_state);
  if (!CheckDoubleWithinRelative(final_objective, maxima_value, tolerance)) {
    ++total_errors;
  }

  // check that objective function gradients are small
  std::vector<double> grad_objective(dim);
  objective_eval.ComputeGradObjectiveFunction(&objective_state, grad_objective.data());

  OL_VERBOSE_PRINTF("grad objective function: ");
#ifdef OL_VERBOSE_PRINT
  PrintMatrix(grad_objective.data(), 1, grad_objective.size());
#endif

  current_errors = 0;
  for (const auto& entry : grad_objective) {
    if (!CheckDoubleWithinRelative(entry, 0.0, tolerance)) {
      ++current_errors;
    }
  }
  total_errors += current_errors;

  // create gradient descent optimizer
  GradientDescentOptimizer<MockEvaluator, DomainType> gd_opt;

  // verify that gradient descent does not move from the optima if we start it there
  objective_state.UpdateCurrentPoint(objective_eval, maxima_point.data());
  gd_opt.Optimize(objective_eval, gd_parameters, domain, &objective_state);
  objective_state.GetCurrentPoint(point_optimized.data());
#ifdef OL_VERBOSE_PRINT
  PrintMatrix(point_optimized.data(), 1, dim);
#endif
  for (int i = 0; i < dim; ++i) {
    if (!CheckDoubleWithinRelative(point_optimized[i], maxima_point[i], 0.0)) {
      ++total_errors;
    }
  }

  // store initial objective function
  objective_state.UpdateCurrentPoint(objective_eval, wrong_point.data());
  initial_objective = objective_eval.ComputeObjectiveFunction(&objective_state);

  // verify that gradient descent can find the optima
  gd_opt.Optimize(objective_eval, gd_parameters, domain, &objective_state);
  objective_state.GetCurrentPoint(point_optimized.data());

  for (int i = 0; i < dim; ++i) {
    if (!CheckDoubleWithinRelative(point_optimized[i], maxima_point[i], tolerance)) {
      ++total_errors;
    }
  }
  final_objective = objective_eval.ComputeObjectiveFunction(&objective_state);
  // objective function value at optima should be correct
  if (!CheckDoubleWithinRelative(final_objective, maxima_value, tolerance)) {
    ++total_errors;
  }
  // objective function cannot get worse
  if (final_objective < initial_objective) {
    ++total_errors;
  }

  // check that objective function gradients are small
  objective_eval.ComputeGradObjectiveFunction(&objective_state, grad_objective.data());

  OL_VERBOSE_PRINTF("grad objective function: ");
#ifdef OL_VERBOSE_PRINT
  PrintMatrix(grad_objective.data(), 1, grad_objective.size());
#endif

  current_errors = 0;
  for (const auto& entry : grad_objective) {
    if (!CheckDoubleWithinRelative(entry, 0.0, tolerance)) {
      ++current_errors;
    }
  }
  total_errors += current_errors;

  return total_errors;
}

/*!\rst
  Test gradient descent's ability to optimize the function represented by MockEvaluator in a constrained setting.

  \return
    number of test failures (invalid results, non-convergence, etc.)
\endrst*/
template <typename MockEvaluator>
OL_WARN_UNUSED_RESULT int MockObjectiveGradientDescentConstrainedOptimizationTestCore() {
  using DomainType = TensorProductDomain;
  const int dim = 3;

  double initial_objective;
  double final_objective;

  // gradient descent parameters
  const double gamma = 0.9;
  const double pre_mult = 1.0;
  const double max_relative_change = 0.8;
  const double tolerance = 1.0e-12;
  const int max_gradient_descent_steps = 1000;
  const int max_num_restarts = 10;
  GradientDescentParameters gd_parameters(1, max_gradient_descent_steps, max_num_restarts, gamma, pre_mult, max_relative_change, tolerance);

  int total_errors = 0;
  int current_errors = 0;

  std::vector<ClosedInterval> domain_bounds = {
    {0.05, 0.32},
    {0.05, 0.6},
    {0.05, 0.32}};
  DomainType domain(domain_bounds.data(), dim);

  std::vector<double> maxima_point_input(dim, 0.5);

  std::vector<double> wrong_point(dim, 0.2);

  std::vector<double> point_optimized(dim);
  std::vector<double> temp_point(dim);

  MockEvaluator objective_eval(maxima_point_input.data(), dim);

  // get optima data
  objective_eval.GetOptimumPoint(temp_point.data());
  const std::vector<double> maxima_point(temp_point);
  const double maxima_value = objective_eval.GetOptimumValue();

  // work out what the maxima point would be given the domain constraints
  std::vector<double> best_in_domain_point(maxima_point);
  for (int i = 0; i < dim; ++i) {
    if (best_in_domain_point[i] > domain_bounds[i].max) {
      best_in_domain_point[i] = domain_bounds[i].max;
    } else if (best_in_domain_point[i] < domain_bounds[i].min) {
      best_in_domain_point[i] = domain_bounds[i].min;
    }
  }

  // build state, setting point initially to maxima_point
  typename MockEvaluator::StateType objective_state(objective_eval, maxima_point.data());

  // verify that claimed optima is actually an optima
  final_objective = objective_eval.ComputeObjectiveFunction(&objective_state);
  if (!CheckDoubleWithinRelative(final_objective, maxima_value, tolerance)) {
    ++total_errors;
  }

  // check that objective function gradients are small
  std::vector<double> grad_objective(dim);
  objective_eval.ComputeGradObjectiveFunction(&objective_state, grad_objective.data());

  OL_VERBOSE_PRINTF("grad objective function: ");
#ifdef OL_VERBOSE_PRINT
  PrintMatrix(grad_objective.data(), 1, grad_objective.size());
#endif

  current_errors = 0;
  for (const auto& entry : grad_objective) {
    if (!CheckDoubleWithinRelative(entry, 0.0, tolerance)) {
      ++current_errors;
    }
  }
  total_errors += current_errors;

  // create gradient descent optimizer
  GradientDescentOptimizer<MockEvaluator, DomainType> gd_opt;

  // verify that gradient descent does not move from the optima if we start it there
  objective_state.UpdateCurrentPoint(objective_eval, best_in_domain_point.data());
  gd_opt.Optimize(objective_eval, gd_parameters, domain, &objective_state);
  objective_state.GetCurrentPoint(point_optimized.data());
#ifdef OL_VERBOSE_PRINT
  PrintMatrix(point_optimized.data(), 1, dim);
#endif
  for (int i = 0; i < dim; ++i) {
    if (!CheckDoubleWithinRelative(point_optimized[i], best_in_domain_point[i], 0.0)) {
      ++total_errors;
    }
  }

  // store initial objective function
  objective_state.UpdateCurrentPoint(objective_eval, wrong_point.data());
  initial_objective = objective_eval.ComputeObjectiveFunction(&objective_state);

  // verify that gradient descent can find the optima
  objective_state.UpdateCurrentPoint(objective_eval, wrong_point.data());
  gd_opt.Optimize(objective_eval, gd_parameters, domain, &objective_state);
  objective_state.GetCurrentPoint(point_optimized.data());

  for (int i = 0; i < dim; ++i) {
    if (!CheckDoubleWithinRelative(point_optimized[i], best_in_domain_point[i], tolerance)) {
      ++total_errors;
    }
  }
  final_objective = objective_eval.ComputeObjectiveFunction(&objective_state);
  // objective function cannot get worse
  if (final_objective < initial_objective) {
    ++total_errors;
  }

  // check that objective function gradients are small
  objective_eval.ComputeGradObjectiveFunction(&objective_state, grad_objective.data());

  OL_VERBOSE_PRINTF("grad objective function: ");
#ifdef OL_VERBOSE_PRINT
  PrintMatrix(grad_objective.data(), 1, grad_objective.size());
#endif

  current_errors = 0;
  for (IdentifyType<decltype(grad_objective)>::type::size_type i = 0, size = grad_objective.size(); i < size; ++i) {
    // only get 0 gradients if the true optima lies inside the domain (in a particular dimension)
    if (domain_bounds[i].IsInside(maxima_point[i])) {
      if (!CheckDoubleWithinRelative(grad_objective[i], 0.0, tolerance)) {
        ++current_errors;
      }
    }
  }
  total_errors += current_errors;

  return total_errors;
}

/*!\rst
  Test newton's ability to optimize the function represented by MockEvaluator in an unconstrained setting.

  \return
    number of test failures (invalid results, non-convergence, etc.)
\endrst*/
template <typename MockEvaluator>
OL_WARN_UNUSED_RESULT int MockObjectiveNewtonOptimizationTestCore() {
  using DomainType = TensorProductDomain;
  const int dim = 3;

  double initial_objective;
  double final_objective;

  // gradient descent parameters
  const double gamma = 10.0;
  const double pre_mult = 1.0e2;
  const double max_relative_change = 1.0;
  const double tolerance = 1.0e-13;
  const int max_newton_steps = 100;
  NewtonParameters newton_parameters(1, max_newton_steps, gamma, pre_mult, max_relative_change, tolerance);

  int total_errors = 0;
  int current_errors = 0;

  std::vector<ClosedInterval> domain_bounds(dim, {-1.0, 1.0});
  DomainType domain(domain_bounds.data(), dim);

  std::vector<double> maxima_point_input(dim, 0.5);

  std::vector<double> wrong_point(dim, 0.2);

  std::vector<double> point_optimized(dim);
  std::vector<double> temp_point(dim);

  MockEvaluator objective_eval(maxima_point_input.data(), dim);

  // get optima data
  objective_eval.GetOptimumPoint(temp_point.data());
  const std::vector<double> maxima_point(temp_point);
  const double maxima_value = objective_eval.GetOptimumValue();

  // build state, setting initial point to maxima_point
  typename MockEvaluator::StateType objective_state(objective_eval, maxima_point.data());

  // verify that claimed optima is actually an optima
  final_objective = objective_eval.ComputeObjectiveFunction(&objective_state);
  if (!CheckDoubleWithinRelative(final_objective, maxima_value, tolerance)) {
    ++total_errors;
  }

  // check that objective function gradients are small
  std::vector<double> grad_objective(dim);
  objective_eval.ComputeGradObjectiveFunction(&objective_state, grad_objective.data());

  OL_VERBOSE_PRINTF("grad objective function: ");
#ifdef OL_VERBOSE_PRINT
  PrintMatrix(grad_objective.data(), 1, grad_objective.size());
#endif

  current_errors = 0;
  for (const auto& entry : grad_objective) {
    if (!CheckDoubleWithinRelative(entry, 0.0, tolerance)) {
      ++current_errors;
    }
  }
  total_errors += current_errors;

  // create newton optimizer
  NewtonOptimizer<MockEvaluator, DomainType> newton_opt;

  // verify that newton does not move from the optima if we start it there
  objective_state.UpdateCurrentPoint(objective_eval, maxima_point.data());
  total_errors += newton_opt.Optimize(objective_eval, newton_parameters, domain, &objective_state);
  objective_state.GetCurrentPoint(point_optimized.data());
#ifdef OL_VERBOSE_PRINT
  PrintMatrix(point_optimized.data(), 1, dim);
#endif
  for (int i = 0; i < dim; ++i) {
    if (!CheckDoubleWithinRelative(point_optimized[i], maxima_point[i], 0.0)) {
      ++total_errors;
    }
  }

  // store initial objective function
  objective_state.UpdateCurrentPoint(objective_eval, wrong_point.data());
  initial_objective = objective_eval.ComputeObjectiveFunction(&objective_state);

  // verify that newton can find the optima
  total_errors += newton_opt.Optimize(objective_eval, newton_parameters, domain, &objective_state);
  objective_state.GetCurrentPoint(point_optimized.data());

  for (int i = 0; i < dim; ++i) {
    if (!CheckDoubleWithinRelative(point_optimized[i], maxima_point[i], tolerance)) {
      ++total_errors;
    }
  }
  final_objective = objective_eval.ComputeObjectiveFunction(&objective_state);
  // objective function value at optima should be correct
  if (!CheckDoubleWithinRelative(final_objective, maxima_value, tolerance)) {
    ++total_errors;
  }
  // objective function cannot get worse
  if (final_objective < initial_objective) {
    ++total_errors;
  }

  // check that objective function gradients are small
  objective_eval.ComputeGradObjectiveFunction(&objective_state, grad_objective.data());

  OL_VERBOSE_PRINTF("grad objective function: ");
#ifdef OL_VERBOSE_PRINT
  PrintMatrix(grad_objective.data(), 1, grad_objective.size());
#endif

  current_errors = 0;
  for (const auto& entry : grad_objective) {
    if (!CheckDoubleWithinRelative(entry, 0.0, tolerance)) {
      ++current_errors;
    }
  }
  total_errors += current_errors;

  return total_errors;
}

/*!\rst
  Test newton's ability to optimize the function represented by MockEvaluator in a constrained setting.

  \return
    number of test failures (invalid results, non-convergence, etc.)
\endrst*/
template <typename MockEvaluator>
OL_WARN_UNUSED_RESULT int MockObjectiveNewtonConstrainedOptimizationTestCore() {
  using DomainType = TensorProductDomain;
  const int dim = 3;

  double initial_objective;
  double final_objective;

  // newton descent parameters
  const double gamma = 1.1;
  const double pre_mult = 1.0e-1;
  const double max_relative_change = 1.0;
  const double tolerance = 1.0e-13;
  const int max_newton_steps = 100;
  NewtonParameters newton_parameters(1, max_newton_steps, gamma, pre_mult, max_relative_change, tolerance);

  int total_errors = 0;
  int current_errors = 0;

  std::vector<double> maxima_point_input(dim, 0.5);

  std::vector<double> wrong_point(dim, 0.2);

  std::vector<double> point_optimized(dim);
  std::vector<double> temp_point(dim);

  std::vector<ClosedInterval> domain_bounds = {
    {0.05, 0.32},
    {0.05, 0.6},
    {0.05, 0.32}};
  DomainType domain(domain_bounds.data(), dim);

  MockEvaluator objective_eval(maxima_point_input.data(), dim);

  // get optima data
  objective_eval.GetOptimumPoint(temp_point.data());
  const std::vector<double> maxima_point(temp_point);
  const double maxima_value = objective_eval.GetOptimumValue();

  // work out what the maxima point would be given the domain constraints
  std::vector<double> best_in_domain_point(maxima_point);
  for (int i = 0; i < dim; ++i) {
    if (best_in_domain_point[i] > domain_bounds[i].max) {
      best_in_domain_point[i] = domain_bounds[i].max;
    } else if (best_in_domain_point[i] < domain_bounds[i].min) {
      best_in_domain_point[i] = domain_bounds[i].min;
    }
  }

  // build state, setting point initially to maxima_point
  typename MockEvaluator::StateType objective_state(objective_eval, maxima_point.data());

  // verify that claimed optima is actually an optima
  final_objective = objective_eval.ComputeObjectiveFunction(&objective_state);
  if (!CheckDoubleWithinRelative(final_objective, maxima_value, tolerance)) {
    ++total_errors;
  }

  // check that objective function gradients are small
  std::vector<double> grad_objective(dim);
  objective_eval.ComputeGradObjectiveFunction(&objective_state, grad_objective.data());

  OL_VERBOSE_PRINTF("grad objective function: ");
#ifdef OL_VERBOSE_PRINT
  PrintMatrix(grad_objective.data(), 1, grad_objective.size());
#endif

  current_errors = 0;
  for (const auto& entry : grad_objective) {
    if (!CheckDoubleWithinRelative(entry, 0.0, tolerance)) {
      ++current_errors;
    }
  }
  total_errors += current_errors;

  // create newton optimizer
  NewtonOptimizer<MockEvaluator, DomainType> newton_opt;

  // verify that newton does not move from the optima if we start it there
  objective_state.UpdateCurrentPoint(objective_eval, best_in_domain_point.data());
  total_errors += newton_opt.Optimize(objective_eval, newton_parameters, domain, &objective_state);
  objective_state.GetCurrentPoint(point_optimized.data());
#ifdef OL_VERBOSE_PRINT
  PrintMatrix(point_optimized.data(), 1, dim);
#endif
  for (int i = 0; i < dim; ++i) {
    if (!CheckDoubleWithinRelative(point_optimized[i], best_in_domain_point[i], 0.0)) {
      ++total_errors;
    }
  }

  // store initial objective function
  objective_state.UpdateCurrentPoint(objective_eval, wrong_point.data());
  initial_objective = objective_eval.ComputeObjectiveFunction(&objective_state);

  // verify that newton can find the optima
  objective_state.UpdateCurrentPoint(objective_eval, wrong_point.data());
  total_errors += newton_opt.Optimize(objective_eval, newton_parameters, domain, &objective_state);
  objective_state.GetCurrentPoint(point_optimized.data());
#ifdef OL_VERBOSE_PRINT
  PrintMatrix(point_optimized.data(), 1, dim);
#endif

  for (int i = 0; i < dim; ++i) {
    if (!CheckDoubleWithinRelative(point_optimized[i], best_in_domain_point[i], tolerance)) {
      ++total_errors;
    }
  }
  final_objective = objective_eval.ComputeObjectiveFunction(&objective_state);
  // objective function cannot get worse
  if (final_objective < initial_objective) {
    ++total_errors;
  }

  // check that objective function gradients are small
  objective_eval.ComputeGradObjectiveFunction(&objective_state, grad_objective.data());

  OL_VERBOSE_PRINTF("grad objective function: ");
#ifdef OL_VERBOSE_PRINT
  PrintMatrix(grad_objective.data(), 1, grad_objective.size());
#endif

  current_errors = 0;
  for (IdentifyType<decltype(grad_objective)>::type::size_type i = 0, size = grad_objective.size(); i < size; ++i) {
    // only get 0 gradients if the true optima lies inside the domain (in a particular dimension dimension)
    if (domain_bounds[i].IsInside(maxima_point[i])) {
      if (!CheckDoubleWithinRelative(grad_objective[i], 0.0, tolerance)) {
        ++current_errors;
      }
    }
  }
  total_errors += current_errors;

  return total_errors;
}

}  // end unnamed namespace

int RunOptimizationTests(OptimizerTypes optimizer_type) {
  switch (optimizer_type) {
    case OptimizerTypes::kGradientDescent: {  // gradient descent tests
      int errors = 0;
      errors += MockObjectiveGradientDescentOptimizationTestCore<SimpleQuadraticEvaluator>();
      errors += MockObjectiveGradientDescentConstrainedOptimizationTestCore<SimpleQuadraticEvaluator>();
      return errors;
    }
    case OptimizerTypes::kNewton: {  // newton tests
      int errors = 0;
      errors += MockObjectiveNewtonOptimizationTestCore<SimpleQuadraticEvaluator>();
      errors += MockObjectiveNewtonConstrainedOptimizationTestCore<SimpleQuadraticEvaluator>();
      return errors;
    }
    default: {
      OL_ERROR_PRINTF("%s: INVALID optimizer_type choice: %d\n", OL_CURRENT_FUNCTION_NAME, optimizer_type);
      return 1;
    }
  }
}


}  // end namespace optimal_learning
