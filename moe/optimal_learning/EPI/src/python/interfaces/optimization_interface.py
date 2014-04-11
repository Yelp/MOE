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
    with ``problem_size`` dimensions.

    This interface is straightforward--we need the ability to compute the problem size (how many independent parameters to
    optimize) as well as the ability to compute ``f(x)`` and/or its various derivatives.

    """

    __metaclass__ = ABCMeta

    @abstractproperty
    def problem_size(self):
        """Return the number of independent parameters to optimize."""
        pass

    @abstractmethod
    def compute_objective_function(self, current_point, **kwargs):
        r"""Compute ``f(current_point)``.

        :param current_point: point at which to compute the objective
        :type current_point: 1d array[problem_size] of double
        :return: value of objective function evaluated at ``current_point``
        :rtype: double

        """
        pass

    @abstractmethod
    def compute_grad_objective_function(self, current_point, **kwargs):
        r"""Compute the gradient of ``f(current_point)`` wrt ``current_point``.

        :param current_point: point at which to compute the objective
        :type current_point: 1d array[problem_size] of double
        :return: gradient of the objective, i-th entry is ``\pderiv{f(x)}{x_i}``
        :rtype: 1d array[problem_size] of double

        """
        pass

    @abstractmethod
    def compute_hessian_objective_function(self, current_point, **kwargs):
        r"""Compute the hessian matrix of ``f(current_point)`` wrt ``current_point``.

        This matrix is symmetric as long as the mixed second derivatives of f(x) are continuous: Clairaut's Theorem.
        http://en.wikipedia.org/wiki/Symmetry_of_second_derivatives

        :param current_point: point at which to compute the objective
        :type current_point: 1d array[problem_size] of double
        :return: hessian of the objective, (i,j)th entry is ``\mixpderiv{f(x)}{x_i}{x_j}``
        :rtype: 2d array[problem_size][problem_size] of double

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
        :rtype: 1d array[optimizable.problem_size] of double

        """
        pass

