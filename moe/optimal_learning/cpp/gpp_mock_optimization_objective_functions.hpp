/*!
  \file gpp_mock_optimization_objective_functions.hpp
  \rst
  This file contains mock objective functions *interfaces* for use with optimization routines.  This is solely for unit testing.
  Instead of testing gradient descent against log marginal likelihood with some random set of data (which is an
  integration test), we would instead like to be able to test gradient descent on something easier to understand,
  e.g., ``z = -x^2 - y^2``.  These simpler functions have analytic optima which makes testing optimizers
  (e.g., gradient descent, newton) much easier.

  See the header comments in gpp_optimization.hpp, Section 3a), for further details on what it means for an
  Evaluator class to be "optimizable."

  Since perfomance is irrelevant for these test functions, we will define pure abstract Evaluator classes that can
  be used to test optimizers in gpp_optimization.hpp.  Generally the usage should be to subclass say SimpleObjectiveFunctionEvaluator
  and override the pure virtuals.  Then we only end up compiling one version of the [templated] optimization code
  for running tests.  See gpp_optimization_test.cpp for examples.

  Following the style laid out in gpp_common.hpp (file comments, item 5), we currently define:

  * ``class SimpleObjectiveFunctionEvaluator;``
  * ``struct SimpleObjectiveFunctionState;``

  SimpleObjectiveFunctionEvaluator defines a pure abstract base class with interface consistent with the interface that all
  .\*Evaluator classes must provide (e.g., ExpectedImprovementEvaluator, LogMarginalLikelihoodEvaluator).

  SimpleObjectiveFunctionState is simple: it's just a container class that holds a point at which to evaluate the polynomial.
\endrst*/

#ifndef MOE_OPTIMAL_LEARNING_CPP_GPP_MOCK_OPTIMIZATION_OBJECTIVE_FUNCTIONS_HPP_
#define MOE_OPTIMAL_LEARNING_CPP_GPP_MOCK_OPTIMIZATION_OBJECTIVE_FUNCTIONS_HPP_

#include <algorithm>
#include <vector>

#include "gpp_common.hpp"

namespace optimal_learning {

struct SimpleObjectiveFunctionState;

/*!\rst
  Class to evaluate the function ``f(x_1,...,x_{dim}) = -\sum_i (x_i - s_i)^2, i = 1..dim``.
  This is a simple quadratic form with maxima at ``(s_1, ..., s_{dim})``.
\endrst*/
class SimpleObjectiveFunctionEvaluator {
 public:
  using StateType = SimpleObjectiveFunctionState;

  virtual ~SimpleObjectiveFunctionEvaluator() = default;

  virtual int dim() const noexcept OL_PURE_FUNCTION OL_WARN_UNUSED_RESULT = 0;

  /*!\rst
    Helpful for testing so we know what the optimum is.  This value should be the result of::

      GetOptimumPoint(point);
      state.SetCurrentPoint(point);
      optimum_value = ComputeObjectiveFunction(state);

    Then ``optimum_value == GetOptimumValue()``.

    \return
      the optimum value of the polynomial.
  \endrst*/
  virtual double GetOptimumValue() const noexcept OL_PURE_FUNCTION OL_WARN_UNUSED_RESULT = 0;

  /*!\rst
    Helpful for testing so we know where the optimum (value returned by GetOptimumValue) occurs.

    .. NOTE:: if the optimal point is not unique, this function may return one arbitrarily.

    \input
      :point[dim]: space to write the output
    OUTPUTS:
      :point[dim]: the point at which the polynomial obtains its optimum value
  \endrst*/
  virtual void GetOptimumPoint(double * restrict point) const noexcept OL_NONNULL_POINTERS = 0;

  /*!\rst
    Compute the quadratic objective function: ``f(x_1,...,x_{dim}) = -\sum_i (x_i - s_i)^2``.

    \param
      quadratic_dummy_state[1]: ptr to a FULLY CONFIGURED StateType (e.g., SimpleObjectiveFunctionState)
    \output
      quadratic_dummy_state[1]: ptr to a FULLY CONFIGURED StateType; only temporary state may be mutated
    \return
      the value of the objective at ``quadratic_dummy_state.GetCurrentPoint()``
  \endrst*/
  virtual double ComputeObjectiveFunction(StateType * quadratic_dummy_state) const OL_NONNULL_POINTERS OL_WARN_UNUSED_RESULT = 0;

  /*!\rst
    Compute the gradient of the objective function: ``f'(x_1,...,x_{dim})_i = -2 * (x_i - s_i)``.

    \param
      quadratic_dummy_state[1]: ptr to a FULLY CONFIGURED StateType (e.g., SimpleObjectiveFunctionState)
    \output
      quadratic_dummy_state[1]: ptr to a FULLY CONFIGURED StateType; only temporary state may be mutated
      grad_polynomial[dim]: gradient of the objective
  \endrst*/
  virtual void ComputeGradObjectiveFunction(StateType * quadratic_dummy_state, double * restrict grad_polynomial) const OL_NONNULL_POINTERS = 0;

  /*!\rst
    Compute the gradient of the objective function: ``f''(x_1,...,x_{dim})_{i,j} = -2 * \delta_{i,j}``.

    \param
      quadratic_dummy_state[1]: ptr to a FULLY CONFIGURED StateType (e.g., SimpleObjectiveFunctionState)
    \output
      quadratic_dummy_state[1]: ptr to a FULLY CONFIGURED StateType; only temporary state may be mutated
      hessian_polynomial[dim][dim]: hessian of the objective
  \endrst*/
  virtual void ComputeHessianObjectiveFunction(StateType * quadratic_dummy_state, double * restrict hessian_polynomial) const OL_NONNULL_POINTERS = 0;

  OL_DISALLOW_COPY_AND_ASSIGN(SimpleObjectiveFunctionEvaluator);

 protected:
  SimpleObjectiveFunctionEvaluator() = default;
};

struct SimpleObjectiveFunctionState final {
  using EvaluatorType = SimpleObjectiveFunctionEvaluator;

  SimpleObjectiveFunctionState(const EvaluatorType& quadratic_eval, double const * restrict current_point_in)
      : dim(quadratic_eval.dim()),
        current_point(current_point_in, current_point_in + dim) {
  }

  SimpleObjectiveFunctionState(SimpleObjectiveFunctionState&& OL_UNUSED(other)) = default;

  int GetProblemSize() const noexcept OL_PURE_FUNCTION OL_WARN_UNUSED_RESULT {
    return dim;
  }

  void GetCurrentPoint(double * restrict current_point_in) const noexcept OL_NONNULL_POINTERS {
    std::copy(current_point.begin(), current_point.end(), current_point_in);
  }

  void SetCurrentPoint(const EvaluatorType& OL_UNUSED(ei_eval), double const * restrict current_point_in) noexcept OL_NONNULL_POINTERS {
    std::copy(current_point_in, current_point_in + dim, current_point.begin());
  }

  //! spatial dimension (e.g., entries per point of points_sampled)
  const int dim;

  // state variables
  //! the point at which to evaluate the associated objective
  std::vector<double> current_point;

  OL_DISALLOW_DEFAULT_AND_COPY_AND_ASSIGN(SimpleObjectiveFunctionState);
};

}  // end namespace optimal_learning

#endif  // MOE_OPTIMAL_LEARNING_CPP_GPP_MOCK_OPTIMIZATION_OBJECTIVE_FUNCTIONS_HPP_
