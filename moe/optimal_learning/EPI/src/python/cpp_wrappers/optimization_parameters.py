# -*- coding: utf-8 -*-
r"""Classes (and utilities for constructing them) that specify the behavior of C++ optimization routines like Gradient Descent and Newton.

C++ expects input objects to have a certain format; the classes in this file make it convenient to put data into the expected
format. Generally the C++ optimizers want to know the objective function (what), optimization method (how), domain (where, etc.
along with paramters like number of iterations, tolerances, etc.

These Python classes/functions wrap the C++ structs in: gpp_optimization_parameters.hpp.

The \*OptimizationParameters structs contain the high level details--what to optimize, how to do it, etc. explicitly. And the hold
a reference to a C++ struct containing parameters for the specific optimizer. The build_*_parameters() helper functions provide
wrappers around these C++ objects' constructors.

"""
import numpy

import moe.build.GPP as C_GP


def build_newton_parameters(num_multistarts, max_num_steps, gamma, time_factor, max_relative_change, tolerance):
    r"""Build a NewtonParameters (C++ object) via its ctor; this object specifies multistarted Newton behavior and is required by C++ Newton optimization.

    .. Note:: See gpp_optimization_parameters.hpp for more details.
       The following comments are copied from NewtonParameters struct in gpp_optimization_parameters.hpp.

    Diagonal dominance control: gamma and time_factor:
    On i-th newton iteration, we add ``1/(time_factor*gamma^(i+1)) * I`` to the Hessian to improve robustness

    Choosing a small gamma (e.g., ``1.0 < gamma <= 1.01``) and time_factor (e.g., ``0 < time_factor <= 1.0e-3``)
    leads to more consistent/stable convergence at the cost of slower performance (and in fact
    for gamma or time_factor too small, gradient descent is preferred).  Conversely, choosing more
    aggressive values may lead to very fast convergence at the cost of more cases failing to
    converge.

    ``gamma = 1.01``, ``time_factor = 1.0e-3`` should lead to good robustness at reasonable speed.  This should be a fairly safe default.
    ``gamma = 1.05, time_factor = 1.0e-1`` will be several times faster but not as robust.
    for "easy" problems, these can be much more aggressive, e.g., ``gamma = 2.0``, ``time_factor = 1.0e1`` or more

    :param num_multistarts: number of initial guesses to try in multistarted newton
    :type num_multistarts: int > 0
    :param max_num_steps: maximum number of newton iterations (per initial guess)
    :type max_num_steps: int > 0
    :param gamma: exponent controlling rate of time_factor growth (see function comments)
    :type gamma: double > 1.0
    :param time_factor: initial amount of additive diagonal dominance (see function comments)
    :type time_factor: double > 0.0
    :param max_relative_change: max change allowed per update (as a relative fraction of current distance to wall)
    :type max_relative_change: double in [0, 1]
    :param tolerance: when the magnitude of the gradient falls below this value, stop
    :type tolerance: double >= 0.0

    """
    newton_data = C_GP.NewtonParameters(
        num_multistarts,
        max_num_steps,
        numpy.float64(gamma),
        numpy.float64(time_factor),
        numpy.float64(max_relative_change),
        numpy.float64(tolerance),
    )
    return newton_data


def build_gradient_descent_parameters(num_multistarts, max_num_steps, max_num_restarts, gamma, pre_mult, max_relative_change, tolerance):
    r"""Build a GradientDescentParameters (C++ object) via its ctor; this object specifies multistarted GD behavior and is required by C++ GD optimization.

    .. Note:: See gpp_optimization_parameters.hpp for more details.
       The following comments are copied from GradientDescentParameters struct in gpp_optimization_parameters.hpp.

    Iterations:
    The total number of gradient descent steps is at most ``num_multistarts * max_num_steps * max_num_restarts``
    Generally, allowing more iterations leads to a better solution but costs more time.

    Learning Rate:
    GD may be implemented using a learning rate: ``pre_mult * (i+1)^{-\gamma}``, where i is the current iteration
    Larger gamma causes the GD step size to (artificially) scale down faster.
    Smaller pre_mult (artificially) shrinks the GD step size.
    Generally, taking a very large number of small steps leads to the most robustness; but it is very slow.

    Tolerances:
    Larger relative changes are potentially less robust but lead to faster convergence.
    Large tolerances run faster but may lead to high errors or false convergence (e.g., if the tolerance is 1.0e-3 and the learning
    rate control forces steps to fall below 1.0e-3 quickly, then GD will quit "successfully" without genuinely converging.)

    :param num_multistarts: number of initial guesses to try in multistarted gradient descent (suggest: a few hundred)
    :type num_multistarts: int > 0
    :param max_num_steps: maximum number of gradient descent iterations per restart (suggest: 200-1000)
    :type max_num_steps: int > 0
    :param max_num_restarts: maximum number of gradient descent restarts, the we are allowed to call gradient descent.  Should be >= 2 as a minimum (suggest: 10-20)
    :type max_num_restarts: int > 0
    :param gamma: exponent controlling rate of step size decrease (see struct docs or GradientDescentOptimizer) (suggest: 0.5-0.9)
    :type gamma: double > 1.0
    :param time_factor: scaling factor for step size (see struct docs or GradientDescentOptimizer) (suggest: 0.1-1.0)
    :type time_factor: double > 0.0
    :param max_relative_change: max change allowed per GD iteration (as a relative fraction of current distance to wall)
           (suggest: 0.5-1.0 for less sensitive problems like EI; 0.02 for more sensitive problems like hyperparameter opt)
    :type max_relative_change: double in [0, 1]
    :param tolerance: when the magnitude of the gradient falls below this value OR we will not move farther than tolerance
           (e.g., at a boundary), stop.  (suggest: 1.0e-7)
    :type tolerance: double >= 0.0

    """
    gd_data = C_GP.GradientDescentParameters(
        num_multistarts,
        max_num_steps,
        max_num_restarts,
        numpy.float64(gamma),
        numpy.float64(pre_mult),
        numpy.float64(max_relative_change),
        numpy.float64(tolerance),
    )
    return gd_data


class HyperparameterOptimizationParameters(object):

    r"""Container for parameters that specify the behavior of a hyperparameter optimizer.

    We use slots to enforce a "type." Typo'ing a member name will error, not add a new field.
    This class is passed to C++, so it is convenient to be strict about its structure.

    :ivar objective_type: (*C_GP.LogLikelihoodTypes*) which log likelihood measure to use as the metric of model quality
      e.g., log marginal likelihood, leave one out cross validation log pseudo-likelihood.
      This attr is set via the cpp_wrappers.log_likelihood.LogLikelihood object used with optimization.
    :ivar optimizer_type: (*C_GP.OptimizerTypes*) which optimizer to use (e.g., dumb search, gradient dsecent, Newton)
    :ivar num_random_samples: (*int >= 0*) number of samples to try if using 'dumb' search
    :ivar optimizer_parameters: (*C_GP.\*Parameters* struct, matching ``optimizer_type``) parameters to control
      derviative-based optimizers, e.g., step size control, number of steps tolerance, etc.

    .. NOTE:: ``optimizer_parameters`` this MUST be a C++ object whose type matches objective_type. e.g., if objective_type
      is kNewton, then this must be built via C_GP.NewtonParameters() (wrapper: build_newton_parameters())

    """

    __slots__ = ('objective_type', 'optimizer_type', 'num_random_samples', 'optimizer_parameters', )
    def __init__(self, optimizer_type=None, num_random_samples=None, optimizer_parameters=None):
        """Construct HyperparameterOptimizationParameters that specifies hyperparameter optimization behavior."""
        # see gpp_python_common.cpp for .*_type enum definitions. .*_type variables must be from those enums (NOT integers)
        self.objective_type = None # set via the LogLikelihood object passed to optimization
        self.optimizer_type = optimizer_type
        self.num_random_samples = num_random_samples # number of samples to 'dumb' search over
        self.optimizer_parameters = optimizer_parameters # must match the optimizer_type

class ExpectedImprovementOptimizationParameters(object):

    r"""Container for parameters that specify the behavior of a expected improvement optimizer.

    We use slots to enforce a "type." Typo'ing a member name will error, not add a new field.
    This class is passed to C++, so it is convenient to be strict about its structure.

    :ivar domain_type: (*C_GP.DomainTypes*) type of domain that we are optimizing expected improvement over (e.g., tensor, simplex)
    :ivar optimizer_type: (*C_GP.OptimizerTypes*) which optimizer to use (e.g., dumb search, gradient dsecent)
    :ivar num_random_samples: (*int >= 0*) number of samples to try if using 'dumb' search or if generating more
      than one simultaneous sample with dumb search fallback enabled
    :ivar optimizer_parameters: (*C_GP.\*Parameters* struct, matching ``optimizer_type``) parameters to control
      derviative-based optimizers, e.g., step size control, number of steps tolerance, etc.

    .. NOTE:: ``optimizer_parameters`` MUST be a C++ object whose type matches objective_type. e.g., if objective_type
      is kGradientDescent, then this must be built via C_GP.GradientDescentParameters object
      (wrapper: build_gradient_descent_parameters())

    """

    __slots__ = ('domain_type', 'optimizer_type', 'num_random_samples', 'optimizer_parameters', )
    def __init__(self, optimizer_type=None, num_random_samples=None, optimizer_parameters=None):
        """Construct ExpectedImprovementOptimizationParameters that specifies EI optimization behavior."""
        # see gpp_python_common.cpp for .*_type enum definitions. .*_type variables must be from those enums (NOT integers)
        self.domain_type = None # set via the DomainInterface object passed to optimization
        self.optimizer_type = optimizer_type
        self.num_random_samples = num_random_samples # number of samples to 'dumb' search over
        self.optimizer_parameters = optimizer_parameters # must match the optimizer_type
        # NOTE: need both num_random_samples AND optimizer_parameters if generating > 1 sample
        # using gradient descent optimization
