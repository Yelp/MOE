# -*- coding: utf-8 -*-
r"""Classes for various optimizers (maximizers) and multistarting said optimizers.

.. Note:: comments in this module are copied from the header comments in gpp_optimization.hpp.

Table of Contents:

1. FILE OVERVIEW
2. OPTIMIZATION OF OBJECTIVE FUNCTIONS

   a. GRADIENT DESCENT

      i. OVERVIEW
      ii. IMPLEMENTATION DETAILS

   b. NEWTON'S METHOD

      i. OVERVIEW
      ii. IMPLEMENTATION DETAILS

   c. MULTISTART OPTIMIZATION

Read the "OVERVIEW" sections for header-style comments that describe the file contents at a high level.
Read the "IMPLEMENTATION" comments for cpp-style comments that talk more about the specifics.  Both types
are included together here since this file contains template class declarations and template function definitions.
For further implementation details, see comment blocks before each individual class/function.

**1. FILE OVERVIEW**

First, the functions in this file are all MAXIMIZERS.  We also use the term "optima," and unless we specifically
state otherwise, "optima" and "optimization" refer to "maxima" and "maximization," respectively.  (Note that
minimizing ``g(x)`` is equivalent to maximizing ``f(x) = -1 * g(x)``.)

This file contains templates for some common optimization techniques: gradient descent (GD) and Newton's method.
We provide constrained implementations (constraint via heuristics like restricting updates to 50% of the distance
to the nearest wall) of these optimizers.  For unconstrained, just set the domain to be huge: ``[-DBL_MAX, DBL_MAX]``.

We provide \*Optimizer template classes (e.g., NewtonOptimizer) as main endpoints for doing local optimization
(i.e., run the optimization method from a single initial guess).  We also provide a MultistartOptimizer class
for global optimization (i.e., start optimizers from each of a set of initial guesses).  These are all discussed
further below. These templated classes are general and can optimize any OptimizableInterface subclass.

In this way, we can make the local and global optimizers completely agonistic to the function they are optimizing.

**2. OPTIMIZATION OF OBJECTIVE FUNCTIONS**

**2a. GRADIENT DESCENT (GD)**

.. Note:: Below there is some discussion of "restarted" Gradient Descent; this is not yet implemented in Python.
    See :mod:`moe.optimal_learning.python.cpp_wrappers.optimization` if you want to use this feature.

**2a, i. OVERVIEW**

We use first derivative information to walk the path of steepest ascent, hopefully toward a (local) maxima of the
chosen log likelihood measure.  This is implemented in: GradientDescentOptimization().
This method ensures that the result lies within a specified domain.

We additionally restart gradient-descent in practice; i.e., we repeatedly take the output of a GD run and start a
new GD run from that point.  This lives in: GradientDescentOptimizer::Optimize().

Even with restarts, gradient descent (GD) cannot start "too far" from the solution and still
successfully find it.  Thus users should typically start it from multiple initial guesses and take the best one
(see gpp_math and gpp_model_selection for examples).  The MultistartOptimizer template class in this file
provides generic multistart functionality.

Finally, we optionally apply Polyak-Ruppert averaging. This is described in more detail in the docstring for
GradientDescentParameters. For functions where we only have access to gradient + noise, this averaging can lead
to better answers than straight gradient descent. It amounts to averaging over the final ``N_{avg}`` steps.

Gradient descent is implemented in: GradientDescentOptimizer::Optimize() (which calls GradientDescentOptimization())

**2a, ii. IMPLEMENTATION DETAILS**

GD's update is: ``\theta_{i+1} = \theta_i + \gamma * \nabla f(\theta_i)``
where ``\gamma`` controls the step-size and is chosen heuristically, often varying by problem.

The previous update leads to unconstrained optimization.  To ensure that our results always stay within the
specified domain, we additionally limit updates if they would move us outside the domain.  For example,
we could imagine only moving half the distance to the nearest boundary.

With gradient descent (GD), it is hard to know what step sizes to take.  Unfortunately, far enough away from an
optima, the objective could increase (but very slowly).  If gradient descent takes too large of a step in a
bad direction, it can easily "get lost."  At the same time, taking very small steps leads to slow performance.
To help, we take the standard approach of scaling down step size with iteration number. We also allow the user
to specify a maximum relative change to limit the aggressiveness of GD steps.  Finally, we wrap GD in a restart
loop, where we fire off another GD run from the current location unless convergence was reached.

**2b. NEWTON'S METHOD:**

.. Note:: Newton's method is not yet implemented in Python. See :mod:`moe.optimal_learning.python.cpp_wrappers.optimization` if you want to use this feature.

**2b, i. OVERVIEW**

Newton's Method (for optimization) uses second derivative information in addition to the first derivatives used by
gradient descent (GD). In higher dimensions, first derivatives => gradients and second derivatives => Hessian matrix.
At each iteration, gradient descent computes the derivative and blindly takes a step (of some
heuristically determined size) in that direction.  Care must be taken in the step size choice to balance robustness
and speed while ensuring that convergence is possible.  By using second derivative (the Hessian matrix in higher
dimensions), which is interpretable as information about local curvature, Newton makes better\* choices about
step size and direction to ensure rapid\*\* convergence.

\*, \*\* See "IMPLEMENTATION DETAILS" comments section for details.

Recall that Newton indiscriminately finds solutions where ``f'(x) = 0``; the eigenvalues of the Hessian classify these
``x`` as optima, saddle points, or indeterminate. We multistart Newton (e.g., gpp_model_selection)
but just take the best objective value without classifying solutions.
The MultistartOptimizer template class in this file provides generic multistart functionality.

Newton is implemented here: NewtonOptimizer::Optimize() (which calls NewtonOptimization())

**2b, ii. IMPLEMENTATION DETAILS**

Let's address the footnotes from the previous section (Section 2b, i paragraph 1):

\* Within its region of attraction, Newton's steps are optimal (when we have only second derivative information).  Outside
of this region, Newton can make very poor decisions and diverge.  In general, Newton is more sensitive to its initial
conditions than gradient descent, but it has the potential to be much, much faster.

\*\* By quadratic convergence, we mean that once Newton is near enough to the solution, the log of the error will roughly
halve each iteration.  Numerically, we would see the "number of digits" double each iteration.  Again, this only happens
once Newton is "close enough."

Newton's Method is a root-finding technique at its base.  To find a root of ``g(x)``, Newton requires an
initial guess, ``x_0``, and the ability to compute ``g(x)`` and ``g'(x)``.  Then the idea is that you compute
root of the line tangent to ``g(x_0)``; call this ``x_1``.  And repeat.  But the core idea is to make repeated
linear approximations to ``g(x)`` and proceed in a fixed-point like fashion.

As an optimization method, we are looking for roots of the gradient, ``f'(x_{opt}) = 0``.  So we require an initial guess
x_0 and the ability to evaluate ``f'(x)`` and ``f''(x)`` (in higher dimensions, the gradient and Hessian of f).  Thus Newton
makes repeated linear approximations to ``f'(x)`` or equivalently, it locally approximates ``f(x)`` with a *quadratic* function,
continuing iteration from the optima of that quadratic.
In particular, Newton would solve the optimization problem of a quadratic program in one iteration.

Mathematically, the update formulas for gradient descent (GD) and Newton are:
GD:     ``\theta_{i+1} = \theta_i +     \gamma       * \nabla f(\theta_i)``
Newton: ``\theta_{i+1} = \theta_i - H_f^-1(\theta_i) * \nabla f(\theta_i)``
Note: the sign of the udpate is flipped because H is *negative* definite near a maxima.
These update schemes are similar.  In GD, ``\gamma`` is chosen heuristically.  There are many ways to proceed but only
so much that can be done with just gradient information; moreover the standard algorithm always proceeds in the direction
of the gradient.  Newton takes a much more general appraoch.  Instead of a scalar ``\gamma``, the Newton update applies
``H^-1`` to the gradient, changing both the direction and magnitude of the step.

Unfortunately, Newton indiscriminately finds solutions where ``f'(x) = 0``.  This is not necesarily an optima!  In one dimension,
we can have ``f'(x) = 0`` and ``f''(x) = 0``, in which case the solution need not be an optima (e.g., ``y = x^3`` at ``x = 0``).
In higher dimensions, a saddle point can also result (e.g., ``z = x^2 - y^2`` at ``x,y = 0``).  More generally, we have an
optima if the Hessian is strictly negative or positive definite; a saddle if the Hessian has both positive and negative
eigenvalues, and an indeterminate case if the Hessian is singular.

**2c. MULTISTART OPTIMIZATION**

Above, we mentioned that gradient descent (GD), Newton, etc. have a difficult time converging if they are started "too far"
from an optima.  Even if convergence occurs, it will typically be very slow unless the problem is simple.  Worse,
in a problem with multiple optima, the methods may converge to the wrong one!

Multistarting the optimizers is one way of mitigating\* this issue.  Multistart involves starting a run of the
specified optimizer (e.g., Newton) from each of a set of initial guesses.  Then the best result is reported as
the result of the whole procedure.  By trying a large number of initial guesses, we potentially reduce the need
for good guesses; i.e., hopefully at least one guess will be "near enough" to the global optimum.  This
functionality is provided in MultistartOptimizer::MultistartOptimize(...).

\* As noted below in the MultistartOptimizer::MultistartOptimize() function docs, mitigate is intentional here.
Multistarting is NOT GUARANTEED to find global optima.  But it can increase the chances of success.

Currently we let the user specify the initial guesses.  In practice, this typically means a random sampling of points.
We do not (yet) make any effort to say sample more heavily from regions where "more stuff is happening" or any
other heuristics.

TODO(GH-165): Improve multistart heuristics.

Finally, MultistartOptimizer::MultistartOptimize() is also used to provide 'dumb' search functionality (optimization
by just evaluating the objective at numerous points).  For sufficiently complex problems, gradient descent, Newton, etc.
can have exceptionally poor convergence characteristics or run too slowly.  In cases where these more advanced techniques
fail, we commonly fall back to 'dumb' search.

"""
from abc import abstractmethod

import collections

import numpy

import scipy.optimize

from moe.optimal_learning.python.interfaces.optimization_interface import OptimizerInterface


def multistart_optimize(optimizer, starting_points=None, num_multistarts=None):
    """Multistart the specified optimizer randomly or from the specified list of initial guesses.

    If ``starting_points`` is specified, this will always multistart from those points.
    If ``starting_points`` is not specified and ``num_multistarts`` is specified, we start from a random set of points.
    If ``starting_points`` is not specified and ``num_multistarts`` is not specified, an exception is raised.

    This is a simple wrapper around MultistartOptimizer.

    :param optimizer: object that will perform the optimization
    :type optimizer: interfaces.optimization_interface.OptimizerInterface subclass
    :param starting_points: points at which to initialize optimization runs
    :type starting_points: array of float64 with shape (num_points, evaluator.problem_size)
    :return: (best point found, objective function values at the end of each optimization run)
    :rtype: tuple: (array of float64 with shape (optimizer.dim), array of float64 with shape (starting_points.shape[0]) or (num_multistarts))
    :raises: ValueError: if both ``starting_points`` and ``num_multistarts`` are None

    """
    if starting_points is None and num_multistarts is None:
        raise ValueError('MUST specify starting points OR num_multistarts.')

    multistart_optimizer = MultistartOptimizer(optimizer, num_multistarts)
    return multistart_optimizer.optimize(random_starts=starting_points)


class NullParameters(collections.namedtuple('NullParameters', [])):

    """Empty container for optimizers that do not require any parameters (e.g., the null optimizer)."""

    __slots__ = ()


# See GradientDescentParameters (below) for docstring.
_BaseNewtonParameters = collections.namedtuple('_BaseNewtonParameters', [
    'max_num_steps',
    'gamma',
    'time_factor',
    'max_relative_change',
    'tolerance',
])


class NewtonParameters(_BaseNewtonParameters):

    """See docstring at :class:`moe.optimal_learning.python.cpp_wrappers.optimization.NewtonParameters`."""

    __slots__ = ()

# See GradientDescentParameters (below) for docstring.
_BaseGradientDescentParameters = collections.namedtuple('_BaseGradientDescentParameters', [
    'max_num_steps',
    'max_num_restarts',
    'num_steps_averaged',
    'gamma',
    'pre_mult',
    'max_relative_change',
    'tolerance',
])


class GradientDescentParameters(_BaseGradientDescentParameters):

    r"""Container to hold parameters that specify the behavior of Gradient Descent.

    .. Note:: the following comments are copied from cpp_wrappers.optimization.GradientDescentParameters

    **Iterations**

    The total number of gradient descent steps is at most ``num_multistarts * max_num_steps * max_num_restarts``
    Generally, allowing more iterations leads to a better solution but costs more time.

    **Averaging**

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

    :ivar max_num_steps: (*int > 0*) maximum number of gradient descent iterations per restart (suggest: 200-1000)
    :ivar max_num_restarts: (*int > 0*) maximum number of gradient descent restarts, the we are allowed to call gradient descent.  Should be >= 2 as a minimum (suggest: 10-20)
    :ivar num_steps_averaged: (*int*) number of steps to use in polyak-ruppert averaging (see above) (suggest: 10-50% of max_num_steps for stochastic problems, 0 otherwise)
    :ivar gamma: (*float64 > 1.0*) exponent controlling rate of step size decrease (see struct docs or GradientDescentOptimizer) (suggest: 0.5-0.9)
    :ivar pre_mult: (*float64 > 1.0*) scaling factor for step size (see struct docs or GradientDescentOptimizer) (suggest: 0.1-1.0)
    :ivar max_relative_change: (*float64 in [0, 1]*) max change allowed per GD iteration (as a relative fraction of current distance to wall)
           (suggest: 0.5-1.0 for less sensitive problems like EI; 0.02 for more sensitive problems like hyperparameter opt)
    :ivar tolerance: (*float 64 >= 0.0*) when the magnitude of the gradient falls below this value OR we will not move farther than tolerance
           (e.g., at a boundary), stop.  (suggest: 1.0e-7)

    """

    __slots__ = ()


# See LBFGSBParameters (below) for docstring.
_BaseLBFGSBParameters = collections.namedtuple('_BaseLBFGSBParameters', [
    'approx_grad',
    'max_func_evals',
    'max_metric_correc',
    'factr',
    'pgtol',
    'epsilon',
])


class LBFGSBParameters(_BaseLBFGSBParameters):

    r"""Container to hold parameters that specify the behavior of L-BFGS-B.

    Suggested values come from scipy documentation for ``scipy.optimize.fmin_l_bfgs_b``:
    http://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.optimize.fmin_l_bfgs_b.html

    :ivar approx_grad: (*bool*) if true, BFGS will approximate the gradient
    :ivar max_func_evals: (*int > 0*) maximum number of objective function calls to make (suggest: 15000)
    :ivar max_metric_correc: (*int > 0*) maximum number of variable metric corrections used to define the limited memorty matrix (suggest: 10)
    :ivar factr: (*float64 > 1.0*) 1e12 for low accuracy, 1e7 for moderate accuracy, and 10 for extremely high accuracy (suggest: 1000.0)
    :ivar pgtol: (*float64 > 0.0*) cutoff for highest component of gradient to be considered a critical point (suggest: 1.0e-5)
    :ivar epsilon: (*float64 > 0.0*) step size for approximating the gradient (suggest: 1.0e-8)

    """

    __slots__ = ()

    def scipy_kwargs(self):
        """Return a dict that can be unpacked as kwargs to ``scipy.optimize.fmin_l_bfgs_b``.

        :return: kwargs for controlling the behavior of fmin_l_bfgs_b
        :rtype: dict

        """
        out_dict = dict(self._asdict())
        out_dict['m'] = out_dict.pop('max_metric_correc')
        out_dict['maxfun'] = out_dict.pop('max_func_evals')
        return out_dict


# See COBYLAParameters (below) for docstring.
_BaseCOBYLAParameters = collections.namedtuple('_BaseCOBYLAParameters', [
    'rhobeg',
    'rhoend',
    'maxfun',
    'catol',
])


class COBYLAParameters(_BaseCOBYLAParameters):

    r"""Container to hold parameters that specify the behavior of COBYLA.

    Suggested values come from scipy documentation for scipy.optimize.fmin_cobyla:
    http://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.optimize.fmin_cobyla.html

    :ivar rhobeg: (*float64 > 0.0*) reasonable initial changes to the variables (suggest: 1.0)
    :ivar rhoend: (*float64 > 0.0*) final accuracy in the optimization (not precisely guaranteed), which is a lower bound on the size of the trust region (suggest: 1.0e-4)
    :ivar maxfun: (*int > 0*) maximum number of objective function calls to make (suggest: 1000)
    :ivar catol: (*float64 > 0.0*) absolute tolerance for constraint violations (suggest: 2.0e-4)

    """

    __slots__ = ()

    # Return a dict that can be unpacked as kwargs to ``scipy.optimize.fmin_cobyla``.
    scipy_kwargs = _BaseCOBYLAParameters._asdict


class NullOptimizer(OptimizerInterface):

    """A "null" or identity optimizer: this does nothing. It is used to perform "dumb" search with MultistartOptimizer."""

    def __init__(self, domain, optimizable, *args, **kwargs):
        """Construct a NullOptimizer.

        :param domain: the domain that this optimizer operates over
        :type domain: interfaces.domain_interface.DomainInterface subclass
        :param optimizable: object representing the objective function being optimized
        :type optimizable: interfaces.optimization_interface.OptimizableInterface subclass

        """
        self.domain = domain
        self.objective_function = optimizable

    def optimize(self, *args, **kwargs):
        """Do nothing; arguments are unused."""
        pass


class GradientDescentOptimizer(OptimizerInterface):

    """Optimizes an objective function over the specified domain with the gradient descent method.

    .. Note:: See optimize() docstring for more details.

    """

    def __init__(self, domain, optimizable, optimizer_parameters, num_random_samples=None):
        """Construct a GradientDescentOptimizer.

        :param domain: the domain that this optimizer operates over
        :type domain: interfaces.domain_interface.DomainInterface subclass
        :param optimizable: object representing the objective function being optimized
        :type optimizable: interfaces.optimization_interface.OptimizableInterface subclass
        :param optimizer_parameters: parameters describing how to perform optimization (tolerances, iterations, etc.)
        :type optimizer_parameters: python_version.optimization.GradientDescentParameters object

        """
        self.domain = domain
        self.objective_function = optimizable
        self.optimizer_parameters = optimizer_parameters

    @staticmethod
    def _get_averaging_range(num_steps_averaged, num_steps_total):
        """Return a valid (non-empty, not out-of-bounds) to average over in suppport of Polyak-Ruppert averaging.

        Yields a range according to the following:

        * ``num_steps_averaged`` < 0: averages all steps
        * ``num_steps_averaged`` == 0: do not average
        * ``num_steps_averaged`` > 0 and <= ``max_num_steps``: average the specified number of steps
        * ``max_steps_averaged`` > ``max_num_steps``: average all steps

        Assumes that the averaging will be done from a 0-indexed array where the first index (position 0)
        is skipped. (We skip this point b/c it would bias the average toward the initial guess.)

        :param num_steps_averaged: desired number of steps to average
        :type num_steps_averaged: int
        :param num_steps_total: total number of steps taken
        :type num_steps_total: int > 1

        """
        if num_steps_averaged < 0 or num_steps_averaged > num_steps_total:
            num_steps_averaged = num_steps_total  # average all steps
        elif num_steps_averaged == 0:
            num_steps_averaged = 1                # average only the last step aka do not average

        start = num_steps_total - num_steps_averaged + 1
        end = num_steps_total + 1
        return start, end

    def optimize(self, **kwargs):
        """Apply gradient-descrent to to find a locally optimal (maximal here) value of the specified objective function.

        .. Note:: comments in this method are copied from the function comments of GradientDescentOptimization() in cpp/gpp_optimization.hpp.

        .. Note:: Additional high-level discussion is provided in section 2a) in the header docs of this file.

        Basic gradient descent (GD) to optimize objective function ``f(x)``::

          input: initial_guess

          next_point = initial_guess
          i = 0;
          while (not converged) {
            direction = derivative of f(x) at next_point
            step_scale = compute step_size scaling: pre_mult * (i+1)^(-gamma)

            next_point += step_scale * direction
            ++i
          }
          if (averaging) {
            next_point = average(previous_points, average_range_start, average_range_end)
          }

        See GradientDescentParameters docstring or the GD code for more information on averaging.

        So it marches along the direction of largest gradient (so the steepest descent) for some distance.  The distance
        is a combination of the size of the gradient and the step_scale factor.  Here, we use an exponentially decreasing
        scale to request progressively smaller step sizes: ``(i+1)^(-gamma)``, where ``i`` is the iteration number

        We do not allow the step to take next_point out of the domain; if this happens, the update is limited.
        Thus the solution is guaranteed to lie within the region specified by "domain"; note that this may not be a
        true optima (i.e., the gradient may be substantially nonzero).

        We may also limit very large updates (controlled via max_relative_change).  Decreasing this value
        makes gradient descent (GD) more stable but also slower.  For very sensitive problems like hyperparameter
        optimization, max_relative_change = 0.02 is suggested; for less sensitive problems
        (e.g., EI, especially analytic), you can use 1.0 (or near).

        The constraint implementation (no stepping outside the domain) and the large update limiting are not "pure" gradient
        descent approaches.  They are all heuristics meant to improve Newton's robustness.  The constraint implementation
        in particular may lead to non-convergence and it also may not find constrained optima that lie exactly on a boundary.  This would
        require a more general handling where we search in an ``d-1`` dimensional subspace (i.e., only on the boundary).

        Note that we are using an absolute tolerance here, based on the size of the most recent step.
        The suggested value is 1.0e-7, although this may need to be loosened for problems with 'difficult' optima (e.g., the shape
        is not locally very peaked).  Setting too high of a tolerance can cause wrong answers--e.g., we stop at a point
        that is not an optima but simply an region with small gradient.  Setting the tolerance too low may make convergence impossible;
        GD could get stuck (bouncing between the same few points) or numerical effects could make it impossible to satisfy tolerance.

        Finally, GD terminates if updates are very small.

        """
        # TODO(GH-59): Implement restarts like in the C++ code.
        initial_guess = self.objective_function.current_point
        x_hat = initial_guess
        x_path = numpy.empty((self.optimizer_parameters.max_num_steps + 1, ) + initial_guess.shape)
        x_path[0, ...] = initial_guess

        step_counter = 1
        while step_counter <= self.optimizer_parameters.max_num_steps:
            alpha_n = self.optimizer_parameters.pre_mult * numpy.power(float(step_counter), -self.optimizer_parameters.gamma)

            self.objective_function.current_point = x_path[step_counter - 1, ...]
            orig_step = self.objective_function.compute_grad_objective_function(**kwargs)

            orig_step *= alpha_n
            fixed_step = self.domain.compute_update_restricted_to_domain(
                self.optimizer_parameters.max_relative_change,
                x_path[step_counter - 1, ...],
                orig_step,
            )

            x_path[step_counter, ...] = fixed_step + x_path[step_counter - 1, ...]

            step_counter += 1
            # TODO(GH-59): tolerance control: if step is too small, stop. This goes at the loop's end, AFTER incrementing step_counter!

        # Polyak-Ruppert averaging: postprocessing step where we replace x_n with:
        # \overbar{x} = \frac{1}{n - n_0} \sum_{t=n_0 + 1}^n x_t
        # n_0 = 0 averages all steps; n_0 = n - 1 is equivalent to returning x_n directly.
        start, end = self._get_averaging_range(self.optimizer_parameters.num_steps_averaged, step_counter - 1)
        x_hat = numpy.mean(x_path[start:end, ...], axis=0)
        self.objective_function.current_point = x_hat


class MultistartOptimizer(OptimizerInterface):

    r"""A general class for multistarting any class that implements interfaces.optimization_interface.OptimizerInterface (except itself).

    .. Note:: comments copied from MultistartOptimizer in gpp_optimization.hpp.

    The use with GradientDescentOptimizer, NewtonOptimizer, etc. are standard practice in nonlinear optimization.  In particular,
    without special properties like convexity, single-start optimizers can converge to local optima.  In general, a nonlinear
    function can have many local optima, so the only way to improve\* your chances of finding the global optimum is to start
    from many different locations.

    \* Improve is intentional here.  In the general case, you are not *guaranteed* (in finite time) to find the global optimum.

    Use with NullOptimizer requires special mention here as it might seem silly. This case reduces to evaluating the
    objective function at every point of initial_guesses.  Through function_values, you can get the objective value at each
    of point of initial_guesses too (e.g., for plotting).  So use MultistartOptimize with NullOptimzer to perform a
    'dumb' search (e.g., initial_guesses can be obtained from a grid, random sampling, etc.).  NullOptimizer allows 'dumb' search
    to use the same code as multistart optimization.  'Dumb' search is inaccurate but it never fails, so we often use it as a
    fall-back when more advanced (e.g., gradient descent) techniques fail.

    """

    def __init__(self, optimizer, num_multistarts):
        """Construct a MultistartOptimizer for multistarting any implementation of OptimizerInterface.

        :param optimizer: object representing the optimization method to be multistarted
        :type optimizer: interfaces.optimization_interface.OptimizableInterface subclass (except itself)
        :param optimizer_parameters:
        :type optimizer_parameters:

        """
        self.optimizer = optimizer
        self.num_multistarts = num_multistarts

    def optimize(self, random_starts=None, **kwargs):
        """Perform multistart optimization with self.optimizer.

        .. Note:: comments copied from MultistartOptimizer::MultistartOptimize in gpp_optimization.hpp.

        Performs multistart optimization with the specified Optimizer (instance variable) to optimize the specified
        OptimizableInterface (objective function) over the specified DomainInterface. Optimizer behavior is controlled
        by the specified ParameterStruct. See class docs and header docs of this file, section 2c and 3b, iii),
        for more information.

        The method allows you to specify what the current best is, so that if optimization cannot beat it, no improvement will be
        reported.  It will otherwise report the overall best improvement (through io_container) as well as the result of every
        individual multistart run if desired (through function_values).

        :param random_starts: points from which to multistart ``self.optimizer``; if None, points are chosen randomly
        :type random_starts: array of float64 with shape (num_points, dim) or None
        :return: (best point found, objective function values at the end of each optimization run)
        :rtype: tuple: (array of float64 with shape (self.optimizer.dim), array of float64 with shape (self.num_multistarts))

        """
        # TODO(GH-59): Pass the best point, fcn value, etc. in thru an IOContainer-like structure.
        if random_starts is None:
            random_starts = self.optimizer.domain.generate_uniform_random_points_in_domain(self.num_multistarts, None)

        best_function_value = -numpy.inf
        best_point = random_starts[0, ...]  # any point will do
        function_value_list = numpy.empty(random_starts.shape[0])

        for i, point in enumerate(random_starts):
            self.optimizer.objective_function.current_point = point
            self.optimizer.optimize(**kwargs)
            function_value = self.optimizer.objective_function.compute_objective_function(**kwargs)
            function_value_list[i] = function_value

            if function_value > best_function_value:
                best_function_value = function_value
                best_point = self.optimizer.objective_function.current_point

        return best_point, function_value_list


class _ScipyOptimizerWrapper(OptimizerInterface):

    """Wrapper class to construct an optimizer from scipy optimization methods.

    Requires the implementation of the :func:`~moe.optimal_learning.python.python_interface.optimization._ScipyOptimizerWrapper._optimize_core` method.

    """

    # Type of the optimization_parameters object, specified in subclass
    optimization_parameters_type = None

    def __init__(self, domain, optimizable, optimization_parameters):
        """Construct the optimizer.

        :param domain: the domain that this optimizer operates over
        :type domain: :class:`~moe.optimal_learning.python.interfaces.domain_interface.DomainInterface` subclass.
        :param optimizable: object representing the objective function being optimized
        :type optimizable: :class:`~moe.optimal_learning.python.interfaces.optimization_interface.OptimizableInterface` subclass
        :param optimization_parameters: parameters describing how to perform optimization (tolerances, iterations, etc.)
        :type optimization_parameters: ``python_version.optimization.*Parameters`` object, matching optimization_parameters_type

        """
        self.domain = domain
        self.objective_function = optimizable

        if not isinstance(optimization_parameters, self.optimization_parameters_type):
            raise TypeError('optimization_paramters is of type: {0:s}, expected {1:s}'.format(optimization_parameters.__class__, self.optimization_parameters_type))
        else:
            self.optimization_parameters = optimization_parameters

        self._num_points = 1
        if hasattr(self.domain, 'num_repeats'):
            self._num_points = self.domain.num_repeats

    def _scipy_decorator(self, func, **kwargs):
        """Wrapper function for expected improvement calculation to feed into the optimizer function.

        func should be of the form ``compute_*`` in :class:`moe.optimal_learning.python.interfaces.optimization_interface.OptimizableInterface`.

        """
        def decorated(point):
            """Decorator for compute_* functions in interfaces.optimization_interface.OptimizableInterface.

            Converts the point to proper format (array with dim (self._num_points, self.domain.dim) instead of flat array)
            and sets the current point before calling the compute function.

            :param point: the point on which to do the calculation
            :type point: array of float64 with shape (self._num_points * self.domain.dim, )

            """
            shaped_point = point.reshape(self._num_points, self.domain.dim)
            self.objective_function.current_point = shaped_point
            value = -func(**kwargs)
            if isinstance(value, (numpy.ndarray)):
                return value.flatten()
            else:
                return value

        return decorated

    def optimize(self, **kwargs):
        """Perform optimization with :func:`~moe.optimal_learning.python.interfaces.optimization_interface._ScipyOptimizerWrapper._optimize_core` and shape the output point.

        Calls the :func:`~moe.optimal_learning.python.interfaces.optimization_interface._ScipyOptimizerWrapper._optimize_core` method.
        ``objective_function.current_point`` will be set to the optimal point found.

        """
        unshaped_point = self._optimize_core(**kwargs)

        if self._num_points == 1:
            shaped_point = unshaped_point
        else:
            shaped_point = unshaped_point.reshape(self._num_points, self.domain.dim)
        self.objective_function.current_point = shaped_point

    @abstractmethod
    def _optimize_core(self, **kwargs):
        """Should return an unshaped point corresponding to the output of the optimizer function from scipy.

        See :func:`~moe.optimal_learning.python.python_version.optimization.LBFGSOptimizer._optimize_core` or
        :func:`~moe.optimal_learning.python.python_version.optimization.COBYLAOptimizer._optimize_core` function for examples.

        :return: The unshaped optimal point from calling the scipy optimization method.
        :rtype: array of float64 with shape (self._num_points * self.domain.dim, )

        """
        pass


class LBFGSBOptimizer(_ScipyOptimizerWrapper):

    r"""Optimizes an objective function over the specified domain with the L-BFGS-B method.

    The BFGS (Broyden-Fletcher-Goldfarb-Shanno) algorithm is a quasi-Newton algorithm for optimization. It can
    be used for DFO (Derivative-Free Optimization) when the gradient is not available, such as is the case for
    the analytic qEI algorithm.

    L-BFGS is a memory efficient version of BFGS, and BFGS-B is a variant that handles simple box constraints.
    We use L-BFGS-B, which is a combination of the two, and is often the optimization algorithm of choice for
    these types of problems.

    For more information, visit the scipy docs and the wikipedia page on BFGS:
    http://en.wikipedia.org/wiki/Limited-memory_BFGS
    http://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_l_bfgs_b.html

    """

    optimization_parameters_type = LBFGSBParameters

    def __init__(self, domain, optimizable, optimization_parameters, num_random_samples=None):
        """Construct a LBFGSBOptimizer.

        :param domain: the domain that this optimizer operates over
        :type domain: :class:`~moe.optimal_learning.python.interfaces.domain_interface.DomainInterface` subclass. Only supports TensorProductDomain.
        :param optimizable: object representing the objective function being optimized
        :type optimizable: :class:`~moe.optimal_learning.python.interfaces.optimization_interface.OptimizableInterface` subclass
        :param optimization_parameters: parameters describing how to perform optimization (tolerances, iterations, etc.)
        :type optimization_parameters: :class:`~moe.optimal_learning.python.python_version.optimization.LBFGSBParameters` object

        """
        super(LBFGSBOptimizer, self).__init__(domain, optimizable, optimization_parameters)

    def _optimize_core(self, **kwargs):
        """Perform an L-BFGS-B optimization given the parameters in ``self.optimization_parameters``."""
        domain_bounding_box = self.domain.get_bounding_box()
        domain_list = [(interval.min, interval.max) for interval in domain_bounding_box]
        domain_numpy = numpy.array(domain_list * self._num_points)

        # Parameters defined above in :class:`~moe.optimal_learning.python.python_version.optimization.LBFGSBParameters` class.
        return scipy.optimize.fmin_l_bfgs_b(
            func=self._scipy_decorator(self.objective_function.compute_objective_function, **kwargs),
            x0=self.objective_function.current_point.flatten(),
            bounds=domain_numpy,
            fprime=self._scipy_decorator(self.objective_function.compute_grad_objective_function, **kwargs),
            **self.optimization_parameters.scipy_kwargs()
        )[0]


class COBYLAOptimizer(_ScipyOptimizerWrapper):

    r"""Optimizes an objective function over the specified contraints with the COBYLA method.

    For more information, visit the scipy docs page and the original paper by Powell:
    http://docs.scipy.org/doc/scipy-0.13.0/reference/generated/scipy.optimize.fmin_cobyla.html
    http://www.damtp.cam.ac.uk/user/na/NA_papers/NA2007_03.pdf

    """

    optimization_parameters_type = COBYLAParameters

    def __init__(self, domain, optimizable, optimization_parameters):
        """Construct a COBYLAOptimizer.

        :param domain: the domain that this optimizer operates over
        :type domain: :class:`~moe.optimal_learning.python.interfaces.domain_interface.DomainInterface` subclass. Only supports TensorProductDomain for now.
        :param optimizable: object representing the objective function being optimized
        :type optimizable: :class:`~moe.optimal_learning.python.interfaces.optimization_interface.OptimizableInterface` subclass
        :param optimization_parameters: parameters describing how to perform optimization (tolerances, iterations, etc.)
        :type optimization_parameters: :class:`~moe.optimal_learning.python.python_version.optimization.COBYLAParameters` object

        """
        super(COBYLAOptimizer, self).__init__(domain, optimizable, optimization_parameters)

    def _optimize_core(self, **kwargs):
        """Perform a COBYLA optimization given the parameters in ``self.optimization_parameters``."""
        # Parameters defined above in :class:`~moe.optimal_learning.python.python_version.optimization.COBYLAParameters` class.
        return scipy.optimize.fmin_cobyla(
            func=self._scipy_decorator(self.objective_function.compute_objective_function, **kwargs),
            x0=self.objective_function.current_point.flatten(),
            cons=self.domain.get_constraint_list(),
            disp=0,  # Suppresses output from the routine.
            **self.optimization_parameters.scipy_kwargs()
        )
