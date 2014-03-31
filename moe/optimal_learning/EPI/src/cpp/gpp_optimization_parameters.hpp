// gpp_optimization_parameters.hpp
/*
  This file specifies OptimizerParameters structs (e.g., GradientDescent, Newton) for holding values that control the behavior
  of the optimizers in gpp_optimization.hpp.  For example, max step sizes, number of iterations, step size control, etc. are all
  specified through these structs.

  These structs also specify multistart behavior pertaining to the multistart optimization code in gpp_math and
  gpp_model_selection_and_hyperparameter_optimization.
*/

#ifndef OPTIMAL_LEARNING_EPI_SRC_CPP_GPP_OPTIMIZATION_PARAMETERS_HPP_
#define OPTIMAL_LEARNING_EPI_SRC_CPP_GPP_OPTIMIZATION_PARAMETERS_HPP_

#include "gpp_common.hpp"

namespace optimal_learning {

/*
  Enum for the various optimizer types. Convenient for specifying which optimizer to use
  in testing and also used by the Python interface to specify the optimizer (e.g., for EI
  and hyperparameter optimization).
*/
enum class OptimizerTypes {
  kNull = 0,  // NullOptimizer<>, used for evaluating objective at points
  kGradientDescent = 1,  // GradientDescentOptimizer<>
  kNewton = 2,  // NewtonOptimizer<>
};

// TODO(eliu): (#58807) do one of two things:
// 1) ADD num_multistarts to NullParameters (since we multistart the "null" optimizer!)
// 2) REMOVE num_multistarts from ALL OptimizerParameter structs. num_multistarts doesn't configure the
//    individual optimizer, it just controls how many times we call that optimizer.
//    Then num_multistarts can live in a struct for multistart optimization (or something else entirely?)

/*
  Empty container for optimizers that do not require any parameters (e.g., the null optimizer).
*/
struct NullParameters {
};

/*
  Container to hold parameters that specify the behavior of Gradient Descent
*/
struct GradientDescentParameters {
  // Users must set parameters explicitly.
  GradientDescentParameters() = delete;

  /*
    Construct a GradientDescentParameters object.  Default, copy, and assignment constructor are disallowed.

    INPUTS:
    See member declarations below for a description of each parameter.
  */
  GradientDescentParameters(int num_multistarts_in, int max_num_steps_in, int max_num_restarts_in, double gamma_in, double pre_mult_in, double max_relative_change_in, double tolerance_in)
      : num_multistarts(num_multistarts_in),
        max_num_steps(max_num_steps_in),
        max_num_restarts(max_num_restarts_in),
        gamma(gamma_in),
        pre_mult(pre_mult_in),
        max_relative_change(max_relative_change_in),
        tolerance(tolerance_in) {
  }

  GradientDescentParameters(GradientDescentParameters&& OL_UNUSED(other)) = default;

  // iteration control
  int num_multistarts;  // number of initial guesses to try in multistarted gradient descent (suggest: a few hundred)
  int max_num_steps;  // maximum number of gradient descent iterations per restart (suggest: 200-1000)
  int max_num_restarts;  // maximum number of gradient descent restarts, the we are allowed to call gradient descent.  Should be >= 2 as a minimum (suggest: 10-20)
  // the total number of gradient descent steps is at most num_multistarts * max_num_steps * max_num_restarts
  // Generally, allowing more iterations leads to a better solution but costs more time.

  // learning rate control
  // GD may be implemented using a learning rate: pre_mult * (i+1)^{-\gamma}, where i is the current iteration
  double gamma;  // exponent controlling rate of step size decrease (see RestartedGradientDescentEIOptimization()) (suggest: 0.5-0.9)
  double pre_mult;  // scaling factor for step size (see RestartedGradientDescentEIOptimization()) (suggest: 0.1-1.0)
  // larger gamma causes the GD step size to (artificially) scale down faster
  // smaller pre_mult (artificially) shrinks the GD step size

  // Generally, taking a very large number of small steps leads to the most robustness; but it is very slow.

  // tolerance control
  double max_relative_change;  // max change allowed per GD iteration (as a relative fraction of current distance to wall) (suggest: 0.5-1.0 for less sensitive problems like EI; 0.02 for more sensitive problems like hyperparameter opt)
  double tolerance;  // when the magnitude of the gradient falls below this value OR we will not move farther than tolerance
                     // (e.g., at a boundary), stop.  (suggest: 1.0e-7)
  // Larger relative changes are potentially less robust but lead to faster convergence.

  // Large tolerances run faster but may lead to high errors or false convergence (e.g., if the tolerance is 1.0e-3 and the learning
  // rate control forces steps to fall below 1.0e-3 quickly, then GD will quit "successfully" without genuinely converging.)
};

/*
  Container to hold parameters that specify the behavior of Newton
*/
struct NewtonParameters {
  // Users must set parameters explicitly.
  NewtonParameters() = delete;

  /*
    Construct a NewtonParameters object.  Default, copy, and assignment constructor are disallowed.

    INPUTS:
    num_multistarts: number of initial guesses to try in multistarted newton
    max_num_steps: maximum number of newton iterations
    gamma: exponent controlling rate of time_factor growth (see NewtonHyperparameterOptimization)
    time_factor: initial amount of additive diagonal dominance (see NewtonHyperparameterOptimization())
    max_relative_change: max relative change allowed per iteration of newton (UNUSED)
    tolerance: when the magnitude of the gradient falls below this value, stop
  */
  NewtonParameters(int num_multistarts_in, int max_num_steps_in, double gamma_in, double time_factor_in, double max_relative_change_in, double tolerance_in)
      : num_multistarts(num_multistarts_in),
        max_num_steps(max_num_steps_in),
        gamma(gamma_in),
        time_factor(time_factor_in),
        max_relative_change(max_relative_change_in),
        tolerance(tolerance_in) {
  }

  NewtonParameters(NewtonParameters&& OL_UNUSED(other)) = default;

  // iteration control
  int num_multistarts;  // number of initial guesses for multistarting
  int max_num_steps;  // maximum number of newton iterations (per initial guess)
  const int max_num_restarts = 1;  // maximum number of newton restarts

  // diagonal dominance control
  // on i-th newton iteration, we add 1/(time_factor*gamma^(i+1)) * I to the Hessian to improve robustness

  // Choosing a small gamma (e.g., 1.0 < gamma <= 1.01) and time_factor (e.g., 0 < time_factor <= 1.0e-3)
  // leads to more consistent/stable convergence at the cost of slower performance (and in fact
  // for gamma or time_factor too small, gradient descent is preferred).  Conversely, choosing more
  // aggressive values may lead to very fast convergence at the cost of more cases failing to
  // converge.

  // gamma = 1.01, time_factor = 1.0e-3 should lead to good robustness at reasonable speed.  This should be a fairly safe default.
  // gamma = 1.05, time_factor = 1.0e-1 will be several times faster but not as robust.
  // for "easy" problems, these can be much more aggressive, e.g., gamma = 2.0, time_factor = 1.0e1 or more
  double gamma;
  double time_factor;

  // tolerance control
  double max_relative_change;  // max change allowed per update (as a relative fraction of current distance to wall)
  double tolerance;  // when the magnitude of the gradient falls below this value, stop
};

}  // end namespace optimal_learning

#endif  // OPTIMAL_LEARNING_EPI_SRC_CPP_GPP_OPTIMIZATION_PARAMETERS_HPP_
