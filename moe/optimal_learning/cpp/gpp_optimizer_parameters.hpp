/*!
  \file gpp_optimizer_parameters.hpp
  \rst
  This file specifies OptimizerParameters structs (e.g., GradientDescent, Newton) for holding values that control the behavior
  of the optimizers in gpp_optimization.hpp.  For example, max step sizes, number of iterations, step size control, etc. are all
  specified through these structs.

  These structs also specify multistart behavior pertaining to the multistart optimization code in gpp_math and
  gpp_model_selection.
\endrst*/

#ifndef MOE_OPTIMAL_LEARNING_CPP_GPP_OPTIMIZER_PARAMETERS_HPP_
#define MOE_OPTIMAL_LEARNING_CPP_GPP_OPTIMIZER_PARAMETERS_HPP_

#include "gpp_common.hpp"

namespace optimal_learning {

/*!\rst
  Enum for the various optimizer types. Convenient for specifying which optimizer to use
  in testing and also used by the Python interface to specify the optimizer (e.g., for EI
  and hyperparameter optimization).
\endrst*/
enum class OptimizerTypes {
  //! NullOptimizer<>, used for evaluating objective at points
  kNull = 0,
  //! GradientDescentOptimizer<>
  kGradientDescent = 1,
  //! NewtonOptimizer<>
  kNewton = 2,
};

// TODO(GH-167): Remove num_multistarts from ALL OptimizerParameter structs. num_multistarts doesn't
// configure the individual optimizer, it just controls how many times we call that optimizer.
// Then num_multistarts can live in a parameter struct specifically for multistart optimization.

/*!\rst
  Empty container for optimizers that do not require any parameters (e.g., the null optimizer).
\endrst*/
struct NullParameters {
};

/*!\rst
  Container to hold parameters that specify the behavior of Gradient Descent.

  .. Note:: these comments are copied in build_gradient_descent_parameters() in cpp_wrappers/optimizer_parameters.py.
     That function wraps this struct's ctor.

  **Iterations**

  The total number of gradient descent steps is at most ``num_multistarts * max_num_steps * max_num_restarts``
  Generally, allowing more iterations leads to a better solution but costs more time.

  **Averaging (TODO(GH-390): NOT IMPLEMTED YET)**

  When optimizing stochastic objective functions, it can often be beneficial to average some number of gradient descent
  steps to obtain the final result (vs just returning the last step).
  Polyak-Ruppert averaging: postprocessing step where we replace ``x_n`` with:
  ``\overbar{x} = \frac{1}{n - n_0} \sum_{t=n_0 + 1}^n x_t``
  ``n_0 = 0`` averages all steps; ``n_0 = n - 1`` is equivalent to returning ``x_n`` directly.
  Here, num_steps_averaged is ``n - n_0``.

  * ``num_steps_averaged`` < 0: averages all steps
  * ``num_steps_averaged`` == 0: do not average
  * ``num_steps_averaged`` > 0 and <= ``max_num_steps``: average the specified number of steps
  * ``max_steps_averaged`` > ``max_num_steps``: average all steps

  **Learning Rate**

  GD may be implemented using a learning rate: ``pre_mult * (i+1)^{-\gamma}``, where i is the current iteration
  Larger gamma causes the GD step size to (artificially) scale down faster.
  Smaller pre_mult (artificially) shrinks the GD step size.
  Generally, taking a very large number of small steps leads to the most robustness; but it is very slow.

  **Tolerances**

  Larger relative changes are potentially less robust but lead to faster convergence.
  Large tolerances run faster but may lead to high errors or false convergence (e.g., if the tolerance is 1.0e-3 and the learning
  rate control forces steps to fall below 1.0e-3 quickly, then GD will quit "successfully" without genuinely converging.)
\endrst*/
struct GradientDescentParameters {
  // Users must set parameters explicitly.
  GradientDescentParameters() = delete;

  /*!\rst
    Construct a GradientDescentParameters object.  Default, copy, and assignment constructor are disallowed.

    INPUTS:
    See member declarations below for a description of each parameter.
  \endrst*/
  GradientDescentParameters(int num_multistarts_in, int max_num_steps_in,
                            int max_num_restarts_in, int num_steps_averaged_in,
                            double gamma_in, double pre_mult_in,
                            double max_relative_change_in, double tolerance_in)
      : num_multistarts(num_multistarts_in),
        max_num_steps(max_num_steps_in),
        max_num_restarts(max_num_restarts_in),
        num_steps_averaged(num_steps_averaged_in),
        gamma(gamma_in),
        pre_mult(pre_mult_in),
        max_relative_change(max_relative_change_in),
        tolerance(tolerance_in) {
  }

  GradientDescentParameters(GradientDescentParameters&& OL_UNUSED(other)) = default;

  // iteration control
  //! number of initial guesses to try in multistarted gradient descent (suggest: a few hundred)
  int num_multistarts;
  //! maximum number of gradient descent iterations per restart (suggest: 200-1000)
  int max_num_steps;
  //! maximum number of gradient descent restarts, the we are allowed to call gradient descent.  Should be >= 2 as a minimum (suggest: 4-20)
  int max_num_restarts;

  // polyak-ruppert averaging control
  //! number of steps to use in polyak-ruppert averaging (see above)
  //! (suggest: 10-50% of max_num_steps for stochastic problems, 0 otherwise)
  int num_steps_averaged;

  // learning rate control
  //! exponent controlling rate of step size decrease (see struct docs or GradientDescentOptimizer) (suggest: 0.5-0.9)
  double gamma;
  //! scaling factor for step size (see struct docs or GradientDescentOptimizer) (suggest: 0.1-1.0)
  double pre_mult;

  // tolerance control
  //! max change allowed per GD iteration (as a relative fraction of current distance to wall)
  //! (suggest: 0.5-1.0 for less sensitive problems like EI; 0.02 for more sensitive problems like hyperparameter opt)
  double max_relative_change;
  //! when the magnitude of the gradient falls below this value OR we will not move farther than tolerance
  //! (e.g., at a boundary), stop.  (suggest: 1.0e-7)
  double tolerance;
};

/*!\rst
  Container to hold parameters that specify the behavior of Newton.

  .. Note:: these comments are copied in build_newton_parameters() in cpp_wrappers/optimizer_parameters.py.
     That function wraps this struct's ctor.

  **Diagonal dominance control: ``gamma`` and ``time_factor``**
  On i-th newton iteration, we add ``1/(time_factor*gamma^{i+1}) * I`` to the Hessian to improve robustness

  Choosing a small gamma (e.g., ``1.0 < gamma <= 1.01``) and time_factor (e.g., ``0 < time_factor <= 1.0e-3``)
  leads to more consistent/stable convergence at the cost of slower performance (and in fact
  for gamma or time_factor too small, gradient descent is preferred).  Conversely, choosing more
  aggressive values may lead to very fast convergence at the cost of more cases failing to
  converge.

  ``gamma = 1.01``, ``time_factor = 1.0e-3`` should lead to good robustness at reasonable speed.  This should be a fairly safe default.
  ``gamma = 1.05, time_factor = 1.0e-1`` will be several times faster but not as robust.
  for "easy" problems, these can be much more aggressive, e.g., ``gamma = 2.0``, ``time_factor = 1.0e1`` or more.
\endrst*/
struct NewtonParameters {
  // Users must set parameters explicitly.
  NewtonParameters() = delete;

  /*!\rst
    Construct a NewtonParameters object.  Default, copy, and assignment constructor are disallowed.

    INPUTS:
    See member declarations below for a description of each parameter.
  \endrst*/
  NewtonParameters(int num_multistarts_in, int max_num_steps_in, double gamma_in,
                   double time_factor_in, double max_relative_change_in,
                   double tolerance_in)
      : num_multistarts(num_multistarts_in),
        max_num_steps(max_num_steps_in),
        gamma(gamma_in),
        time_factor(time_factor_in),
        max_relative_change(max_relative_change_in),
        tolerance(tolerance_in) {
  }

  NewtonParameters(NewtonParameters&& OL_UNUSED(other)) = default;

  // iteration control
  //! number of initial guesses for multistarting (suggest: a few hundred)
  int num_multistarts;
  //! maximum number of newton iterations (per initial guess) (suggest: 100)
  int max_num_steps;
  //! maximum number of newton restarts (fixed; not used by newton)
  const int max_num_restarts = 1;

  // diagonal domaince control
  //! exponent controlling rate of time_factor growth (see class docs and NewtonOptimizer) (suggest: 1.01-1.1)
  double gamma;
  //! initial amount of additive diagonal dominance (see class docs and NewtonOptimizer) (suggest: 1.0e-3-1.0e-1)
  double time_factor;

  // tolerance control
  //! max change allowed per update (as a relative fraction of current distance to wall) (Newton may ignore this) (suggest: 1.0)
  double max_relative_change;
  //! when the magnitude of the gradient falls below this value, stop (suggest: 1.0e-10)
  double tolerance;
};

}  // end namespace optimal_learning

#endif  // MOE_OPTIMAL_LEARNING_CPP_GPP_OPTIMIZER_PARAMETERS_HPP_
