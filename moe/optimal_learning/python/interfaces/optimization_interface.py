# -*- coding: utf-8 -*-
"""Interfaces for optimization (maximization): OptimizableInterface for things that can be optimized and OptimizerInterface to perform the optimization.

First, implementation of these interfaces should be MAXIMIZERS.  We also use the term "optima," and unless we specifically
state otherwise, "optima" and "optimization" refer to "maxima" and "maximization," respectively.  (Note that
minimizing g(x) is equivalent to maximizing f(x) = -1 * g(x).)

See the file comments for gpp_optimization.hpp for further dicussion of optimization as well as the particular techniques available
through the C++ interface.

"""
from builtins import object
from abc import ABCMeta, abstractmethod, abstractproperty
from future.utils import with_metaclass


class OptimizableInterface(with_metaclass(ABCMeta, object)):

    r"""Interface that an object must fulfill to be optimized by an implementation of OptimizerInterface.

    Below, ``f(x)`` is the scalar objective function represented by this object. ``x`` is a vector-valued input
    with ``problem_size`` dimensions. With ``f(x)`` (and/or its derivatives), a OptimizableInterface implementation
    can be hooked up to a OptimizerInterface implementation to find the maximum value of ``f(x)`` and the input
    ``x`` at which this maximum occurs.

    This interface is straightforward--we need the ability to compute the problem size (how many independent parameters to
    optimize) as well as the ability to compute ``f(x)`` and/or its various derivatives. An implementation of ``f(x)`` is
    required; this allows for derivative-free optimization methods. Providing derivatives opens the door to more
    advanced/efficient techniques (e.g., gradient descent, BFGS, Newton).

    This interface is meant to be generic. For example, when optimizing the log marginal likelihood of a GP model
    (wrt hyperparameters of covariance; e.g., python_version.log_likelihood.GaussianProcessLogMarginalLikelihood)
    ``f`` is the log marginal, ``x`` is the vector of hyperparameters, and ``problem_size`` is ``num_hyperparameters``.
    Note that log marginal and covariance both have an associated spatial dimension, and this is NOT ``problem_size``.
    For Expected Improvement (e.g., python_version.expected_improvement.ExpectedImprovement), ``f`` would be the EI,
    ``x`` is the new experiment point (or points) being optimized, and ``problem_size`` is ``dim`` (or ``num_points*dim``).

    TODO(GH-71): getter/setter for current_point.

    """

    @abstractproperty
    def problem_size(self):
        """Return the number of independent parameters to optimize."""
        pass

    def get_current_point(self):
        """Get the current_point (array of float64 with shape (problem_size)) at which this object is evaluating the objective function, ``f(x)``."""
        pass

    def set_current_point(self, current_point):
        """Set current_point to the specified point; ordering must match.

        :param current_point: current_point at which to evaluate the objective function, ``f(x)``
        :type current_point: array of float64 with shape (problem_size)

        """
        pass

    current_point = abstractproperty(get_current_point, set_current_point)

    @abstractmethod
    def compute_objective_function(self, **kwargs):
        r"""Compute ``f(current_point)``.

        :return: value of objective function evaluated at ``current_point``
        :rtype: float64

        """
        pass

    @abstractmethod
    def compute_grad_objective_function(self, **kwargs):
        r"""Compute the gradient of ``f(current_point)`` wrt ``current_point``.

        :return: gradient of the objective, i-th entry is ``\pderiv{f(x)}{x_i}``
        :rtype: array of float64 with shape (problem_size)

        """
        pass

    @abstractmethod
    def compute_hessian_objective_function(self, **kwargs):
        r"""Compute the hessian matrix of ``f(current_point)`` wrt ``current_point``.

        This matrix is symmetric as long as the mixed second derivatives of f(x) are continuous: Clairaut's Theorem.
        http://en.wikipedia.org/wiki/Symmetry_of_second_derivatives

        :return: hessian of the objective, (i,j)th entry is ``\mixpderiv{f(x)}{x_i}{x_j}``
        :rtype: array of float64 with shape (problem_size, problem_size)

        """
        pass


class OptimizerInterface(with_metaclass(ABCMeta, object)):

    r"""Interface to *maximize* any object implementing OptimizableInterface (defined above).

    Implementations are responsible for tracking an OptimizableInterface subclass (the objective being optimized),
    a DomainInterface subclass (the domain that the objective lives in), and any parameters needed for controlling
    optimization behavior\*.

    \* Examples include iteration counts, tolerances, learning rate, etc. It is suggested that implementers define a
       FooParameters container class for their FooOptimizer implementation of this interface.

    """

    @abstractmethod
    def optimize(self, **kwargs):
        r"""Maximize a function f(x), represented by an implementation of OptimizableInterface.

        The initial guess is set through calling the ``set_current_point`` method of this object's
        OptimizableInterface data member.

        In general, kwargs not specifically consumed by the implementation of optimize() should be passed down to
        member functions of the ``optimizable`` input.

        This method is not required to have a return value; implementers may use one for convenience. The
        optimal point (as determined by optimization) should be available through the OptimizableInterface data
        member's ``get_current_point`` method.

        TODO(GH-59): Pass the best point, fcn value, etc. in thru an IOContainer-like structure.

        """
        pass
