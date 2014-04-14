# -*- coding: utf-8 -*-
r"""Interface for computation of the Expected Improvement at points sampled from a GaussianProcess.

.. NOTE:: These comments were copied from the file comments in gpp_math.cpp.

See the package docs (interfaces/__init__.py) for the basics of expected improvement and the definition of the q,p-EI problem.

Then the improvement for this single sample is:
``I = { best_known - min(y)   if (best_known - min(y) > 0)      (Equation 5)``
``    {          0               else``
where y is a particular prediction from the underlying Gaussian Process and best_known is the best observed value (min) so far.

And the expected improvement, EI, can be computed by averaging repeated computations of I; i.e., monte-carlo integration.
This is done in ExpectedImprovementInterface.compute_expected_improvement(); we can also compute the gradient. This
computation is needed in the optimization of q,p-EI.

There is also a special, analytic case of EI computation that does not require monte-carlo integration. This special
case can only be used to compute 1,0-EI (and its gradient). Still this can be very useful (e.g., the heuristic
optimization in gpp_heuristic_expected_improvement_optimization.hpp estimates q,0-EI by repeatedly solving
1,0-EI).

From there, since EI is taken from a sum of gaussians, we expect it to be reasonably smooth
and apply multistart, restarted gradient descent to find the optimum.  The use of gradient descent
implies the need for all of the various "grad" functions, e.g., gaussian_process.compute_grad_mean_of_points().
This is handled by coupling an implementation of ExpectedImprovementInterface to an optimizer (optimization_interface.py).

"""
import numpy

from abc import ABCMeta, abstractmethod, abstractproperty

class ExpectedImprovementInterface(object):

    r"""Interface for Expected Improvement computation: EI and its gradient at specified point(s) sampled from a GaussianProcess.

    A class to encapsulate the computation of expected improvement and its spatial gradient using points sampled from an
    associated GaussianProcess. The general EI computation requires monte-carlo integration; it can support q,p-EI optimization.
    It is designed to work with any GaussianProcess.

    See file docs for a description of what EI is and an overview of how it can be computed.

    Implementers are responsible for dealing with PRNG state for any randomness needed in EI computation.
    Implementers are also responsible for storing current_point and points_to_sample:

    :param current_point: point at which to compute EI (i.e., q in q,p-EI)
    :type current_point: 1d array[dim] of double
    :param points_to_sample: array of points which are being sampled concurrently (i.e., p in q,p-EI)
    :type points_to_sample: 2d array[num_to_sample][dim] of double

    """

    __metaclass__ = ABCMeta

    @abstractproperty
    def dim(self):
        """Return the number of spatial dimensions."""
        pass

    @abstractmethod
    def compute_expected_improvement(self, **kwargs):
        r"""Compute the expected improvement at ``current_point``, with ``points_to_sample`` concurrent points being sampled.

        .. NOTE:: These comments were copied from ExpectedImprovementEvaluator::ComputeExpectedImprovement in gpp_math.hpp.
           and duplicated in cpp_wrappers/expected_improvement.py.

        ``current_points`` is the q and points_to_sample is the p in q,p-EI.

        We compute ``EI(Xs) = E_n[[f^*_n(X) - min(f(Xs_1),...,f(Xs_m))]^+]``, where ``Xs`` are potential points
        to sample and ``X`` are already sampled points.  The ``^+`` indicates that the expression in the expectation evaluates to 0
        if it is negative.  ``f^*(X)`` is the MINIMUM over all known function evaluations (``points_sampled_value``), whereas
        ``f(Xs)`` are *GP-predicted* function evaluations.

        The EI is the expected improvement in the current best known objective function value that would result from sampling
        at ``points_to_sample``.

        In general, the EI expression is complex and difficult to evaluate; hence we use Monte-Carlo simulation to approximate it.

        :return: value of EI evaluated at ``current_point``
        :rtype: double

        """
        pass

    @abstractmethod
    def compute_grad_expected_improvement(self, **kwargs):
        r"""Compute the gradient of expected improvement at ``current_point`` wrt ``current_point``, with ``points_to_sample`` concurrent samples.

        .. NOTE:: These comments were copied from ExpectedImprovementEvaluator::ComputeGradExpectedImprovement in gpp_math.hpp
           and duplicated in cpp_wrappers/expected_improvement.py.

        ``current_points`` is the q and points_to_sample is the p in q,p-EI.

        In general, the expressions for gradients of EI are complex and difficult to evaluate; hence we use
        Monte-Carlo simulation to approximate it.


        :return: gradient of EI, i-th entry is ``\pderiv{EI(x)}{x_i}`` where ``x`` is ``current_point``
        :rtype: 1d array[dim] of double

        """
        pass
