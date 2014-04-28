# -*- coding: utf-8 -*-
"""Interfaces for optimization (maximization): OptimizableInterface for things that can be optimized and OptimizerInterface to perform the optimizattion.

First, implementation of these interfaces should be MAXIMIZERS.  We also use the term "optima," and unless we specifically
state otherwise, "optima" and "optimization" refer to "maxima" and "maximization," respectively.  (Note that
minimizing g(x) is equivalent to maximizing f(x) = -1 * g(x).)

See the file comments for gpp_optimization.hpp for further dicussion of optimization as well as the particular techniques available
through the C++ interface.

"""
from abc import ABCMeta, abstractmethod, abstractproperty


class OptimizableInterface(object):

    r"""Interface that an object must fulfill to be optimized by an implementation of OptimizationInterface.

    Below, ``f(x)`` is the scalar objective function represented by this object. ``x`` is a vector-valued input
    with ``problem_size`` dimensions. With ``f(x)`` (and/or its derivatives), a OptimizableInterface implementation
    can be hooked up to a OptimizationInterface implementation to find the maximum value of ``f(x)`` and the input
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

    TODO(eliu): getter/setter for current_point. maybe following this?
    http://google-styleguide.googlecode.com/svn/trunk/pyguide.html#Function_and_Method_Decorators
    How to make it work with ABCs?

    """

    __metaclass__ = ABCMeta

    @abstractproperty
    def problem_size(self):
        """Return the number of independent parameters to optimize."""
        pass

    @abstractmethod
    def get_current_point(self):
        """Get the current_point (array of float64 with shape (problem_size)) at which this object is evaluating the objective function, ``f(x)``."""
        pass

    @abstractmethod
    def set_current_point(self, current_point):
        """Set current_point to the specified point; ordering must match.

        :param current_point: current_point at which to evaluate the objective function, ``f(x)``
        :type current_point: array of float64 with shape (problem_size)

        """
        pass

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


class OptimizerInterface(object):

    r"""Interface to *maximize* any object implementing OptimizableInterface (defined above)."""

    __metaclass__ = ABCMeta

    @abstractmethod
    def optimize(self, optimizable, optimization_parameters, **kwargs):
        r"""Maximize a function f(x), represented by an implementation of OptimizableInterface.

        If an initial guess is required (vs optimizer auto-selects starting point(s)), passing via an ``initial_guess``
        kwarg is suggested.

        In general, kwargs not specifically consumed by the implementation of optimize() should be passed down to
        member functions of the ``optimizable`` input.

        :param optimizable: point at which to compute the objective
        :type optimizable: OptimizableInterface
        :param optimization_parameters: object specifying the desired optimization method (e.g., gradient descent, random search)
          and parameters controlling its behavior (e.g., tolerance, iterations, etc.)
        :type optimization_parameters: implementation-defined
        :return: point at which the objective function is maximized
        :rtype: array of float64 with shape (optimizable.problem_size)

        """
        pass
