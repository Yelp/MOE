# -*- coding: utf-8 -*-
r"""Interface for computation of the Expected Improvement at points sampled from a GaussianProcess.

.. NOTE:: These comments were copied from the file comments in gpp_math.cpp.

See the package docs (:mod:`moe.optimal_learning.python.interfaces`) for the basics of expected improvement and the definition of the q,p-EI problem.

Then the improvement for this single sample is:
``I = { best_known - min(y)   if (best_known - min(y) > 0)      (Equation 5)``
``    {          0               else``
where y is a particular prediction from the underlying Gaussian Process and best_known is the best observed value (min) so far.

And the expected improvement, EI, can be computed by averaging repeated computations of I; i.e., monte-carlo integration.
This is done in :mod:`moe.optimal_learning.python.interfaces.expected_improvement_interface.ExpectedImprovementInterface.compute_expected_improvement`;
we can also compute the gradient. This computation is needed in the optimization of q,p-EI.

There is also a special, analytic case of EI computation that does not require monte-carlo integration. This special
case can only be used to compute 1,0-EI (and its gradient). Still this can be very useful (e.g., the heuristic
optimization in gpp_heuristic_expected_improvement_optimization.hpp estimates q,0-EI by repeatedly solving
1,0-EI).

From there, since EI is taken from a sum of gaussians, we expect it to be reasonably smooth
and apply multistart, restarted gradient descent to find the optimum.  The use of gradient descent
implies the need for all of the various "grad" functions, e.g., gaussian_process.compute_grad_mean_of_points().
This is handled by coupling an implementation of
:class:`moe.optimal_learning.python.interfaces.expected_improvement_interface.ExpectedImprovementInterface`
to an optimizer (:mod:`moe.optimal_learning.python.interfaces.optimization_interface`).

"""
from builtins import object
from abc import ABCMeta, abstractmethod, abstractproperty
from future.utils import with_metaclass


class ExpectedImprovementInterface(with_metaclass(ABCMeta, object)):

    r"""Interface for Expected Improvement computation: EI and its gradient at specified point(s) sampled from a GaussianProcess.

    A class to encapsulate the computation of expected improvement and its spatial gradient using points sampled from an
    associated GaussianProcess. The general EI computation requires monte-carlo integration; it can support q,p-EI optimization.
    It is designed to work with any GaussianProcess.

    See file docs for a description of what EI is and an overview of how it can be computed.

    Implementers are responsible for dealing with PRNG state for any randomness needed in EI computation.
    Implementers are also responsible for storing ``points_to_sample`` and ``points_being_sampled``:

    :param points_to_sample: points at which to evaluate EI and/or its gradient to check their value in future experiments (i.e., "q" in q,p-EI)
    :type points_to_sample: array of float64 with shape (num_to_sample, dim)
    :param points_being_sampled: points being sampled in concurrent experiments (i.e., "p" in q,p-EI)
    :type points_being_sampled: array of float64 with shape (num_being_sampled, dim)

    """

    @abstractproperty
    def dim(self):
        """Return the number of spatial dimensions."""
        pass

    @abstractproperty
    def num_to_sample(self):
        """Number of points at which to compute/optimize EI, aka potential points to sample in future experiments; i.e., the ``q`` in ``q,p-EI``."""
        pass

    @abstractproperty
    def num_being_sampled(self):
        """Number of points being sampled in concurrent experiments; i.e., the ``p`` in ``q,p-EI``."""
        pass

    @abstractmethod
    def compute_expected_improvement(self, **kwargs):
        r"""Compute the expected improvement at ``points_to_sample``, with ``points_being_sampled`` concurrent points being sampled.

        .. NOTE:: These comments were copied from ExpectedImprovementEvaluator::ComputeExpectedImprovement in gpp_math.hpp.
          and duplicated in
          :meth:`moe.optimal_learning.python.cpp_wrappers.expected_improvement.ExpectedImprovement.compute_expected_improvement` and
          :meth:`moe.optimal_learning.python.python_version.expected_improvement.ExpectedImprovement.compute_expected_improvement`

        ``points_to_sample`` is the "q" and ``points_being_sampled`` is the "p" in q,p-EI.

        Computes the expected improvement ``EI(Xs) = E_n[[f^*_n(X) - min(f(Xs_1),...,f(Xs_m))]^+]``, where ``Xs``
        are potential points to sample (union of ``points_to_sample`` and ``points_being_sampled``) and ``X`` are
        already sampled points.  The ``^+`` indicates that the expression in the expectation evaluates to 0 if it
        is negative.  ``f^*(X)`` is the MINIMUM over all known function evaluations (``points_sampled_value``),
        whereas ``f(Xs)`` are *GP-predicted* function evaluations.

        In words, we are computing the expected improvement (over the current ``best_so_far``, best known
        objective function value) that would result from sampling (aka running new experiments) at
        ``points_to_sample`` with ``points_being_sampled`` concurrent/ongoing experiments.

        In general, the EI expression is complex and difficult to evaluate; hence we use Monte-Carlo simulation to approximate it.
        When faster (e.g., analytic) techniques are available, we will prefer them.

        The idea of the MC approach is to repeatedly sample at the union of ``points_to_sample`` and
        ``points_being_sampled``. This is analogous to gaussian_process_interface.sample_point_from_gp,
        but we sample ``num_union`` points at once:
        ``y = \mu + Lw``
        where ``\mu`` is the GP-mean, ``L`` is the ``chol_factor(GP-variance)`` and ``w`` is a vector
        of ``num_union`` draws from N(0, 1). Then:
        ``improvement_per_step = max(max(best_so_far - y), 0.0)``
        Observe that the inner ``max`` means only the smallest component of ``y`` contributes in each iteration.
        We compute the improvement over many random draws and average.

        :return: the expected improvement from sampling ``points_to_sample`` with ``points_being_sampled`` concurrent experiments
        :rtype: float64

        """
        pass

    @abstractmethod
    def compute_grad_expected_improvement(self, **kwargs):
        r"""Compute the gradient of expected improvement at ``points_to_sample`` wrt ``points_to_sample``, with ``points_being_sampled`` concurrent samples.

        .. NOTE:: These comments were copied from ExpectedImprovementEvaluator::ComputeGradExpectedImprovement in gpp_math.hpp
          and duplicated
          :meth:`moe.optimal_learning.python.cpp_wrappers.expected_improvement.ExpectedImprovement.compute_grad_expected_improvement` and
          :meth:`moe.optimal_learning.python.python_version.expected_improvement.ExpectedImprovement.compute_grad_expected_improvement`

        ``points_to_sample`` is the "q" and ``points_being_sampled`` is the "p" in q,p-EI.

        In general, the expressions for gradients of EI are complex and difficult to evaluate; hence we use
        Monte-Carlo simulation to approximate it. When faster (e.g., analytic) techniques are available, we will prefer them.

        The MC computation of grad EI is similar to the computation of EI (decsribed in
        compute_expected_improvement). We differentiate ``y = \mu + Lw`` wrt ``points_to_sample``;
        only terms from the gradient of ``\mu`` and ``L`` contribute. In EI, we computed:
        ``improvement_per_step = max(max(best_so_far - y), 0.0)``
        and noted that only the smallest component of ``y`` may contribute (if it is > 0.0).
        Call this index ``winner``. Thus in computing grad EI, we only add gradient terms
        that are attributable to the ``winner``-th component of ``y``.

        :return: gradient of EI, ``\pderiv{EI(Xq \cup Xp)}{Xq_{i,d}}`` where ``Xq`` is ``points_to_sample``
          and ``Xp`` is ``points_being_sampled`` (grad EI from sampling ``points_to_sample`` with
          ``points_being_sampled`` concurrent experiments wrt each dimension of the points in ``points_to_sample``)
        :rtype: array of float64 with shape (num_to_sample, dim)

        """
        pass
