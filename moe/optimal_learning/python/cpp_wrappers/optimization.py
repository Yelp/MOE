# -*- coding: utf-8 -*-
r"""Thin optimization-related containers that can be passed to ``cpp_wrappers.*`` functions/classes that perform optimization.

See cpp/gpp_optimization.hpp for more details on optimization techniques.

C++ optimization tools are templated, so it doesn't make much sense to expose their members to Python (unless we built a
C++ optimizable object that could call Python). So the classes in this file track the data necessary
for C++ calls to construct the matching C++ optimization object and the appropriate optimizer parameters.

C++ expects input objects to have a certain format; the classes in this file make it convenient to put data into the expected
format. Generally the C++ optimizers want to know the objective function (what), optimization method (how), domain (where, etc.
along with paramters like number of iterations, tolerances, etc.

These Python classes/functions wrap the C++ structs in: gpp_optimizer_parameters.hpp.

The \*OptimizerParameters structs contain the high level details--what to optimize, how to do it, etc. explicitly. And the hold
a reference to a C++ struct containing parameters for the specific optimizer. The build_*_parameters() helper functions provide
wrappers around these C++ objects' constructors.

.. Note:: the following comments in this module are copied from the header comments in gpp_optimization.hpp.

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

**1 FILE OVERVIEW**

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

**2 OPTIMIZATION OF OBJECTIVE FUNCTIONS**

**2a GRADIENT DESCENT (GD)**

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

**2b. NEWTON'S METHOD**

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
from builtins import object
import collections

import moe.build.GPP as C_GP
from moe.optimal_learning.python.comparison import EqualityComparisonMixin
from moe.optimal_learning.python.interfaces.optimization_interface import OptimizerInterface


# TODO(GH-167): Kill 'num_multistarts' when you reoganize num_multistarts for C++.
class NullParameters(collections.namedtuple('NullParameters', ['num_multistarts'])):

    """Empty container for optimizers that do not require any parameters (e.g., the null optimizer)."""

    __slots__ = ()


class NewtonParameters(C_GP.NewtonParameters, EqualityComparisonMixin):

    """Container to hold parameters that specify the behavior of Newton in a C++-readable form.

    See :func:`~moe.optimal_learning.python.cpp_wrappers.optimization.NewtonParameters.__init__` docstring for more information.

    """

    __slots__ = ()

    def __init__(self, *args, **kwargs):
        r"""Build a NewtonParameters (C++ object) via its ctor; this object specifies multistarted Newton behavior and is required by C++ Newton optimization.

        .. Note:: See gpp_optimizer_parameters.hpp for more details.
            The following comments are copied from NewtonParameters struct in gpp_optimizer_parameters.hpp.

        **Diagonal dominance control: ``gamma`` and ``time_factor``**

        On i-th newton iteration, we add ``1/(time_factor*gamma^(i+1)) * I`` to the Hessian to improve robustness

        Choosing a small gamma (e.g., ``1.0 < gamma <= 1.01``) and time_factor (e.g., ``0 < time_factor <= 1.0e-3``)
        leads to more consistent/stable convergence at the cost of slower performance (and in fact
        for gamma or time_factor too small, gradient descent is preferred).  Conversely, choosing more
        aggressive values may lead to very fast convergence at the cost of more cases failing to
        converge.

        ``gamma = 1.01``, ``time_factor = 1.0e-3`` should lead to good robustness at reasonable speed.  This should be a fairly safe default.
        ``gamma = 1.05, time_factor = 1.0e-1`` will be several times faster but not as robust.
        for "easy" problems, these can be much more aggressive, e.g., ``gamma = 2.0``, ``time_factor = 1.0e1`` or more

        :param num_multistarts: number of initial guesses to try in multistarted newton (suggest: a few hundred)
        :type num_multistarts: int > 0
        :param max_num_steps: maximum number of newton iterations (per initial guess) (suggest: 100)
        :type max_num_steps: int > 0
        :param gamma: exponent controlling rate of time_factor growth (see function comments) (suggest: 1.01-1.1)
        :type gamma: float64 > 1.0
        :param time_factor: initial amount of additive diagonal dominance (see function comments) (suggest: 1.0e-3-1.0e-1)
        :type time_factor: float64 > 0.0
        :param max_relative_change: max change allowed per update (as a relative fraction of current distance to wall) (suggest: 1.0)
        :type max_relative_change: float64 in [0, 1]
        :param tolerance: when the magnitude of the gradient falls below this value, stop (suggest: 1.0e-10)
        :type tolerance: float64 >= 0.0

        """
        super(NewtonParameters, self).__init__(*args, **kwargs)


class GradientDescentParameters(C_GP.GradientDescentParameters, EqualityComparisonMixin):

    """Container to hold parameters that specify the behavior of Gradient Descent in a C++-readable form.

    See :func:`~moe.optimal_learning.python.cpp_wrappers.optimization.GradientDescentParameters.__init__` docstring for more information.

    """

    __slots__ = ()

    def __init__(self, *args, **kwargs):
        r"""Build a GradientDescentParameters (C++ object) via its ctor; this object specifies multistarted GD behavior and is required by C++ GD optimization.

        .. Note:: See gpp_optimizer_parameters.hpp for more details.
            The following comments are copied from GradientDescentParameters struct in gpp_optimizer_parameters.hpp.
            And they are copied to python_version.optimization.GradientDescentParameters.

        **Iterations**

        The total number of gradient descent steps is at most ``num_multistarts * max_num_steps * max_num_restarts``
        Generally, allowing more iterations leads to a better solution but costs more time.

        **Averaging (NOT IMPLEMENTED IN C++ YET)**

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

        :param num_multistarts: number of initial guesses to try in multistarted gradient descent (suggest: a few hundred)
        :type num_multistarts: int > 0
        :param max_num_steps: maximum number of gradient descent iterations per restart (suggest: 200-1000)
        :type max_num_steps: int > 0
        :param max_num_restarts: maximum number of gradient descent restarts, the we are allowed to call gradient descent.  Should be >= 2 as a minimum (suggest: 4-20)
        :type max_num_restarts: int > 0
        :param num_steps_averaged: number of steps to use in polyak-ruppert averaging (see above) (suggest: 10-50% of max_num_steps for stochastic problems, 0 otherwise) (UNUSED)
        :type num_steps_averaged: int (range is clamped as described above)
        :param gamma: exponent controlling rate of step size decrease (see struct docs or GradientDescentOptimizer) (suggest: 0.5-0.9)
        :type gamma: float64 > 1.0
        :param pre_mult: scaling factor for step size (see struct docs or GradientDescentOptimizer) (suggest: 0.1-1.0)
        :type pre_mult: float64 > 0.0
        :param max_relative_change: max change allowed per GD iteration (as a relative fraction of current distance to wall)
            (suggest: 0.5-1.0 for less sensitive problems like EI; 0.02 for more sensitive problems like hyperparameter opt)
        :type max_relative_change: float64 in [0, 1]
        :param tolerance: when the magnitude of the gradient falls below this value OR we will not move farther than tolerance
            (e.g., at a boundary), stop.  (suggest: 1.0e-7)
        :type tolerance: float64 >= 0.0

        """
        super(GradientDescentParameters, self).__init__(*args, **kwargs)


class _CppOptimizerParameters(object):

    r"""Container for parameters that specify what & how to optimize in C++.

    This object is *internal*. Build cpp_wrappers.*Optimizer objects instead.

    We use slots to enforce a "type." Typo'ing a member name will error, not add a new field.
    This class is passed to C++, so it is convenient to be strict about its structure.

    :ivar domain_type: (*C_GP.DomainTypes*) type of domain that we are optimizing expected improvement over (e.g., tensor, simplex)
    :ivar objective_type: (*C_GP.LogLikelihoodTypes*) which log likelihood measure to use as the metric of model quality
      e.g., log marginal likelihood, leave one out cross validation log pseudo-likelihood.
      This attr is set via the cpp_wrappers.log_likelihood.LogLikelihood object used with optimization.
      This attr is not used when optimizing EI.
    :ivar optimizer_type: (*C_GP.OptimizerTypes*) which optimizer to use (e.g., dumb search, gradient dsecent, Newton)
    :ivar num_random_samples: (*int >= 0*) number of samples to try if using 'dumb' search
    :ivar optimizer_parameters: (*C_GP.\*Parameters* struct, matching ``optimizer_type``) parameters to control
      derviative-based optimizers, e.g., step size control, number of steps tolerance, etc.

    .. NOTE:: ``optimizer_parameters`` this MUST be a C++ object whose type matches objective_type. e.g., if objective_type
        is kNewton, then this must be built via C_GP.NewtonParameters() (i.e., cpp_wrapers.NewtonParameters.newton_data)

    .. NOTE:: when optimizing EI, need both num_random_samples AND optimizer_parameters if generating > 1 sample
        using gradient descent optimization

    """

    __slots__ = ('domain_type', 'objective_type', 'optimizer_type', 'num_random_samples', 'optimizer_parameters', )

    def __init__(
            self,
            domain_type=None,
            objective_type=None,
            optimizer_type=None,
            num_random_samples=None,
            optimizer_parameters=None,
    ):
        """Construct CppOptimizerParameters that specifies optimization behavior to C++."""
        # see gpp_python_common.cpp for .*_type enum definitions. .*_type variables must be from those enums (NOT integers)
        self.domain_type = domain_type
        self.objective_type = objective_type
        self.optimizer_type = optimizer_type
        self.num_random_samples = num_random_samples  # number of samples to 'dumb' search over
        if optimizer_parameters:
            self.optimizer_parameters = optimizer_parameters  # must match the optimizer_type
        else:
            self.optimizer_parameters = None


class NullOptimizer(OptimizerInterface):

    """A "null" or identity optimizer: this does nothing. It is used to perform "dumb" search with MultistartOptimizer."""

    def __init__(self, domain, optimizable, optimizer_parameters, num_random_samples=None):
        """Construct a NullOptimizer.

        :param domain: the domain that this optimizer operates over
        :type domain: interfaces.domain_interface.DomainInterface subclass
        :param optimizable: object representing the objective function being optimized
        :type optimizable: interfaces.optimization_interface.OptimizableInterface subclass
        :param optimizer_parameters: None
        :type optimizer_parameters: None
        :params num_random_samples: number of random samples to use if performing 'dumb' search
        :type num_random_sampes: int >= 0

        """
        self.domain = domain
        self.objective_function = optimizable
        self.optimizer_type = C_GP.OptimizerTypes.null
        self.optimizer_parameters = _CppOptimizerParameters(
            domain_type=domain._domain_type,
            objective_type=optimizable.objective_type,
            optimizer_type=self.optimizer_type,
            num_random_samples=num_random_samples,
            optimizer_parameters=None,
        )

    def optimize(self, *args, **kwargs):
        """Do nothing; arguments are unused."""
        pass


class GradientDescentOptimizer(OptimizerInterface):

    """Simple container for telling C++ to use Gradient Descent for optimization.

    See this module's docstring for some more information or the comments in gpp_optimization.hpp
    for full details on GD.

    """

    def __init__(self, domain, optimizable, optimizer_parameters, num_random_samples=None):
        """Construct a GradientDescentOptimizer.

        :param domain: the domain that this optimizer operates over
        :type domain: interfaces.domain_interface.DomainInterface subclass from cpp_wrappers
        :param optimizable: object representing the objective function being optimized
        :type optimizable: interfaces.optimization_interface.OptimizableInterface subclass from cpp_wrappers
        :param optimizer_parameters: parameters describing how to perform optimization (tolerances, iterations, etc.)
        :type optimizer_parameters: cpp_wrappers.optimization.GradientDescentParameters object
        :params num_random_samples: number of random samples to use if performing 'dumb' search
        :type num_random_sampes: int >= 0

        """
        self.domain = domain
        self.objective_function = optimizable
        self.optimizer_type = C_GP.OptimizerTypes.gradient_descent
        self.optimizer_parameters = _CppOptimizerParameters(
            domain_type=domain._domain_type,
            objective_type=optimizable.objective_type,
            optimizer_type=self.optimizer_type,
            num_random_samples=num_random_samples,
            optimizer_parameters=optimizer_parameters,
        )

    def optimize(self, **kwargs):
        """C++ does not expose this endpoint."""
        raise NotImplementedError("C++ wrapper currently does not support optimization member functions.")


class NewtonOptimizer(OptimizerInterface):

    """Simple container for telling C++ to use Gradient Descent for optimization.

    See this module's docstring for some more information or the comments in gpp_optimization.hpp
    for full details on Newton.

    """

    def __init__(self, domain, optimizable, optimizer_parameters, num_random_samples=None):
        """Construct a NewtonOptimizer.

        :param domain: the domain that this optimizer operates over
        :type domain: interfaces.domain_interface.DomainInterface subclass from cpp_wrappers
        :param optimizable: object representing the objective function being optimized
        :type optimizable: interfaces.optimization_interface.OptimizableInterface subclass from cpp_wrappers
        :param optimizer_parameters: parameters describing how to perform optimization (tolerances, iterations, etc.)
        :type optimizer_parameters: cpp_wrappers.optimization.NewtonParameters object
        :params num_random_samples: number of random samples to use if performing 'dumb' search
        :type num_random_sampes: int >= 0

        """
        self.domain = domain
        self.objective_function = optimizable
        self.optimizer_type = C_GP.OptimizerTypes.newton
        self.optimizer_parameters = _CppOptimizerParameters(
            domain_type=domain._domain_type,
            objective_type=optimizable.objective_type,
            optimizer_type=self.optimizer_type,
            num_random_samples=num_random_samples,
            optimizer_parameters=optimizer_parameters,
        )

    def optimize(self, **kwargs):
        """C++ does not expose this endpoint."""
        raise NotImplementedError("C++ wrapper currently does not support optimization member functions.")
