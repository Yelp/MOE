# -*- coding: utf-8 -*-
r"""Tools to compute log likelihood-like measures of model fit and optimize them (wrt the hyperparameters of covariance) to select the best model for a given set of historical data.

See the file comments in :mod:`moe.optimal_learning.python.interfaces.log_likelihood_interface`
for an overview of log likelihood-like metrics and their role
in model selection. This file provides an implementation of the Log Marginal Likelihood.

.. Note:: This is a copy of the file comments in :mod:`moe.optimal_learning.python.cpp_wrappers.log_likelihood`.

**LOG MARGINAL LIKELIHOOD (LML)**

(Rasmussen & Williams, 5.4.1)
The Log Marginal Likelihood measure comes from the ideas of Bayesian model selection, which use Bayesian inference
to predict distributions over models and their parameters.  The cpp file comments explore this idea in more depth.
For now, we will simply state the relevant result.  We can build up the notion of the "marginal likelihood":
probability(observed data GIVEN sampling points (``X``), model hyperparameters, model class (regression, GP, etc.)),
which is denoted: ``p(y | X, \theta, H_i)`` (see the cpp file comments for more).

So the marginal likelihood deals with computing the probability that the observed data was generated from (another
way: is easily explainable by) the given model.

The marginal likelihood is in part paramaterized by the model's hyperparameters; e.g., as mentioned above.  Thus
we can search for the set of hyperparameters that produces the best marginal likelihood and use them in our model.
Additionally, a nice property of the marginal likelihood optimization is that it automatically trades off between
model complexity and data fit, producing a model that is reasonably simple while still explaining the data reasonably
well.  See the cpp file comments for more discussion of how/why this works.

In general, we do not want a model with perfect fit and high complexity, since this implies overfit to input noise.
We also do not want a model with very low complexity and poor data fit: here we are washing the signal out with
(assumed) noise, so the model is simple but it provides no insight on the data.

This is not magic.  Using GPs as an example, if the covariance function is completely mis-specified, we can blindly
go through with marginal likelihood optimization, obtain an "optimal" set of hyperparameters, and proceed... never
realizing that our fundamental assumptions are wrong.  So care is always needed.

"""
import copy

import numpy

import scipy.linalg

from moe.optimal_learning.python.constant import DEFAULT_MAX_NUM_THREADS
from moe.optimal_learning.python.geometry_utils import ClosedInterval
from moe.optimal_learning.python.interfaces.log_likelihood_interface import GaussianProcessLogLikelihoodInterface
from moe.optimal_learning.python.interfaces.optimization_interface import OptimizableInterface
from moe.optimal_learning.python.python_version import python_utils
from moe.optimal_learning.python.python_version.domain import TensorProductDomain
from moe.optimal_learning.python.python_version.optimization import multistart_optimize, NullOptimizer


def multistart_hyperparameter_optimization(
        hyperparameter_optimizer,
        num_multistarts,
        randomness=None,
        max_num_threads=DEFAULT_MAX_NUM_THREADS,
        status=None,
):
    r"""Select the hyperparameters that maximize the specified log likelihood measure of model fit (over the historical data) within the specified domain.

    .. Note:: The following comments are copied from
      :func:`moe.optimal_learning.python.cpp_wrappers.log_likelihood.multistart_hyperparameter_optimization`.

    See :class:`moe.optimal_learning.python.python_version.log_likelihood.GaussianProcessLogMarginalLikelihood` and
    :class:`moe.optimal_learning.python.python_version.log_likelihood.GaussianProcessLeaveOneOutLogLikelihood`
    for an overview of some example log likelihood-like measures.

    Optimizers are: null ('dumb' search), gradient descent, newton
    Newton is the suggested optimizer, which is not presently available in Python (use the C++ interface). In Python,
    gradient descent is suggested.

    TODO(GH-57): Implement hessians and Newton's method.

    'dumb' search means this will just evaluate the objective log likelihood measure at num_multistarts 'points'
    (hyperparameters) in the domain, uniformly sampled using latin hypercube sampling.

    See gpp_python_common.cpp for C++ enum declarations laying out the options for objective and optimizer types.

    Currently, during optimization, we recommend that the coordinates of the initial guesses not differ from the
    coordinates of the optima by more than about 1 order of magnitude. This is a very (VERY!) rough guideline for
    sizing the domain and gd_parameters.num_multistarts; i.e., be wary of sets of initial guesses that cover the space too sparsely.

    Solution is guaranteed to lie within the region specified by "domain"; note that this may not be a
    true optima (i.e., the gradient may be substantially nonzero).

    .. WARNING:: this function fails if NO improvement can be found!  In that case,
       the output will always be the first randomly chosen point. status will report failure.

    TODO(GH-56): Allow callers to pass in a source of randomness.

    :param hyperparameter_optimizer: object that optimizes (e.g., gradient descent, newton) the desired log_likelihood
        measure over a domain (wrt the hyperparameters of covariance)
    :type hyperparameter_optimizer: interfaces.optimization_interfaces.OptimizerInterface subclass
    :param num_multistarts: number of times to multistart ``hyperparameter_optimizer``
    :type num_multistarts: int > 0
    :param randomness: random source used to generate multistart points (UNUSED)
    :type randomness: (UNUSED)
    :param max_num_threads: maximum number of threads to use, >= 1 (UNUSED)
    :type max_num_threads: int > 0
    :param status: (output) status messages (e.g., reporting on optimizer success, etc.)
    :type status: dict
    :return: hyperparameters that maximize the specified log likelihood measure within the specified domain
    :rtype: array of float64 with shape (log_likelihood_evaluator.num_hyperparameters)

    """
    # Producing the random starts in log10 space improves robustness by clustering some extra points near 0
    domain_bounds_log10 = numpy.log10(hyperparameter_optimizer.domain._domain_bounds)
    domain_log10 = TensorProductDomain(ClosedInterval.build_closed_intervals_from_list(domain_bounds_log10))
    random_starts = domain_log10.generate_uniform_random_points_in_domain(num_points=num_multistarts)
    random_starts = numpy.power(10.0, random_starts)

    best_hyperparameters, _ = multistart_optimize(hyperparameter_optimizer, starting_points=random_starts)

    # TODO(GH-59): Have GD actually indicate whether updates were found, e.g., in an IOContainer-like structure.
    found_flag = True
    if status is not None:
        status["gradient_descent_found_update"] = found_flag

    return best_hyperparameters


def evaluate_log_likelihood_at_hyperparameter_list(
        log_likelihood_evaluator,
        hyperparameters_to_evaluate,
        max_num_threads=DEFAULT_MAX_NUM_THREADS,
        status=None,
):
    """Compute the specified log likelihood measure at each input set of hyperparameters.

    Generally Newton or gradient descent is preferred but when they fail to converge this may be the only "robust" option.
    This function is also useful for plotting or debugging purposes (just to get a bunch of log likelihood values).

    :param log_likelihood_evaluator: object specifying which log likelihood measure to evaluate
    :type log_likelihood_evaluator: interfaces.log_likelihood_interface.LogLikelihoodInterface subclass
    :param hyperparameters_to_evaluate: the hyperparameters at which to compute the specified log likelihood
    :type hyperparameters_to_evaluate: array of float64 with shape (num_to_eval, log_likelihood_evaluator.num_hyperparameters)
    :param max_num_threads: maximum number of threads to use, >= 1 (UNUSED)
    :type max_num_threads: int
    :param status: (output) status messages (e.g., reporting on optimizer success, etc.)
    :type status: dict
    :return: log likelihood value at each specified set of hyperparameters
    :rtype: array of float64 with shape (hyperparameters_to_evaluate.shape[0])

    """
    null_optimizer = NullOptimizer(None, log_likelihood_evaluator)
    _, values = multistart_optimize(null_optimizer, starting_points=hyperparameters_to_evaluate)

    # TODO(GH-59): Have null optimizer actually indicate whether updates were found, e.g., in an IOContainer-like structure.
    found_flag = True
    if status is not None:
        status["evaluate_log_likelihood_at_hyperparameter_list"] = found_flag

    return values


class GaussianProcessLogMarginalLikelihood(GaussianProcessLogLikelihoodInterface, OptimizableInterface):

    r"""Class for computing the Log Marginal Likelihood, ``log(p(y | X, \theta))``.

    That is, the probability of observing the training values, y, given the training points, X,
    and hyperparameters (of the covariance function), ``\theta``.

    This is a measure of how likely it is that the observed values came from our Gaussian Process Prior.

    .. Note:: Comments are copied from
      :class:`moe.optimal_learning.python.cpp_wrappers.log_likelihood.GaussianProcessLogMarginalLikelihood`.

    Given a particular covariance function (including hyperparameters) and
    training data ((point, function value, measurement noise) tuples), the log marginal likelihood is the log probability that
    the data were observed from a Gaussian Process would have generated the observed function values at the given measurement
    points.  So log marginal likelihood tells us "the probability of the observations given the assumptions of the model."
    Log marginal sits well with the Bayesian Inference camp.
    (Rasmussen & Williams p118)

    This quantity primarily deals with the trade-off between model fit and model complexity.  Handling this trade-off is automatic
    in the log marginal likelihood calculation.  See Rasmussen & Williams 5.2 and 5.4.1 for more details.

    We can use the log marginal likelihood to determine how good our model is.  Additionally, we can maximize it by varying
    hyperparameters (or even changing covariance functions) to improve our model quality.  Hence this class provides access
    to functions for computing log marginal likelihood and its hyperparameter gradients.

    .. Note:: Equivalent methods of :class:`moe.optimal_learning.python.interfaces.log_likelihood_interface.GaussianProcessLogLikelihoodInterface` and
      :class:`moe.optimal_learning.python.interfaces.optimization_interface.OptimizableInterface`
      are aliased below (e.g., :class:`~moe.optimal_learning.python.python_version.log_likelihood.GaussianProcessLogLikelihood.problem_size` and
      :class:`~moe.optimal_learning.python.python_version.log_likelihood.GaussianProcessLogLikelihood.num_hyperparameters`,
      :class:`~moe.optimal_learning.python.python_version.log_likelihood.GaussianProcessLogLikelihood.compute_log_likelihood` and
      :class:`~moe.optimal_learning.python.python_version.log_likelihood.GaussianProcessLogLikelihood.compute_objective_function`, etc).

    """

    def __init__(self, covariance_function, historical_data):
        """Construct a LogLikelihood object that knows how to call C++ for evaluation of member functions.

        :param covariance_function: covariance object encoding assumptions about the GP's behavior on our data
        :type covariance_function: :class:`moe.optimal_learning.python.interfaces.covariance_interface.CovarianceInterface` subclass
        :param historical_data: object specifying the already-sampled points, the objective value at those points, and the noise variance associated with each observation
        :type historical_data: :class:`moe.optimal_learning.python.data_containers.HistoricalData` object

        """
        self._covariance = copy.deepcopy(covariance_function)
        self._historical_data = copy.deepcopy(historical_data)

    @property
    def dim(self):
        """Return the number of spatial dimensions."""
        return self._historical_data.dim

    @property
    def num_hyperparameters(self):
        """Return the number of hyperparameters aka the number of independent parameters to optimize."""
        return self._covariance.num_hyperparameters

    problem_size = num_hyperparameters

    def get_hyperparameters(self):
        """Get the hyperparameters (array of float64 with shape (num_hyperparameters)) of this covariance.

        Equivalently, get the current_point at which this object is evaluating the objective function, ``f(x)``

        """
        return self._covariance.hyperparameters

    def set_hyperparameters(self, hyperparameters):
        """Set hyperparameters to the specified hyperparameters; ordering must match.

        :param hyperparameters: hyperparameters at which to evaluate the log likelihood (objective function), ``f(x)``
        :type hyperparameters: array of float64 with shape (num_hyperparameters)

        """
        self._covariance.hyperparameters = hyperparameters

    hyperparameters = property(get_hyperparameters, set_hyperparameters)
    current_point = hyperparameters

    @property
    def _points_sampled(self):
        """Return the coordinates of the already-sampled points; see :class:`moe.optimal_learning.python.data_containers.HistoricalData`."""
        return self._historical_data.points_sampled

    @property
    def _points_sampled_value(self):
        """Return the function values measured at each of points_sampled; see :class:`moe.optimal_learning.python.data_containers.HistoricalData`."""
        return self._historical_data.points_sampled_value

    @property
    def _points_sampled_noise_variance(self):
        """Return the noise variance associated with points_sampled_value; see :class:`moe.optimal_learning.python.data_containers.HistoricalData`."""
        return self._historical_data.points_sampled_noise_variance

    def get_covariance_copy(self):
        """Return a copy of the covariance object specifying the Gaussian Process.

        :return: covariance object encoding assumptions about the GP's behavior on our data
        :rtype: interfaces.covariance_interface.CovarianceInterface subclass

        """
        return copy.deepcopy(self._covariance)

    def get_historical_data_copy(self):
        """Return the data (points, function values, noise) specifying the prior of the Gaussian Process.

        :return: object specifying the already-sampled points, the objective value at those points, and the noise variance associated with each observation
        :rtype: data_containers.HistoricalData

        """
        return copy.deepcopy(self._historical_data)

    def compute_log_likelihood(self):
        r"""Compute the _log_likelihood_type measure at the specified hyperparameters.

        .. NOTE:: These comments are copied from LogMarginalLikelihoodEvaluator::ComputeLogLikelihood in gpp_model_selection.cpp.

        ``log p(y | X, \theta) = -\frac{1}{2} * y^T * K^-1 * y - \frac{1}{2} * \log(det(K)) - \frac{n}{2} * \log(2*pi)``
        where n is ``num_sampled``, ``\theta`` are the hyperparameters, and ``\log`` is the natural logarithm.  In the following,
        ``term1 = -\frac{1}{2} * y^T * K^-1 * y``
        ``term2 = -\frac{1}{2} * \log(det(K))``
        ``term3 = -\frac{n}{2} * \log(2*pi)``

        For an SPD matrix ``K = L * L^T``,
        ``det(K) = \Pi_i L_ii^2``
        We could compute this directly and then take a logarithm.  But we also know:
        ``\log(det(K)) = 2 * \sum_i \log(L_ii)``
        The latter method is (currently) preferred for computing ``\log(det(K))`` due to reduced chance for overflow
        and (possibly) better numerical conditioning.

        :return: value of log_likelihood evaluated at hyperparameters (``LL(y | X, \theta)``)
        :rtype: float64

        """
        covariance_matrix = python_utils.build_covariance_matrix(
            self._covariance,
            self._points_sampled,
            noise_variance=self._points_sampled_noise_variance,
        )
        K_chol = scipy.linalg.cho_factor(covariance_matrix, lower=True, overwrite_a=True)

        log_marginal_term2 = -numpy.log(K_chol[0].diagonal()).sum()

        K_inv_y = scipy.linalg.cho_solve(K_chol, self._points_sampled_value)
        log_marginal_term1 = -0.5 * numpy.inner(self._points_sampled_value, K_inv_y)

        log_marginal_term3 = -0.5 * numpy.float64(self._points_sampled_value.size) * numpy.log(2.0 * numpy.pi)
        return log_marginal_term1 + log_marginal_term2 + log_marginal_term3

    compute_objective_function = compute_log_likelihood

    def compute_grad_log_likelihood(self):
        r"""Compute the gradient (wrt hyperparameters) of the _log_likelihood_type measure at the specified hyperparameters.

        .. NOTE:: These comments are copied from LogMarginalLikelihoodEvaluator::ComputeGradLogLikelihood in gpp_model_selection.cpp.

        Computes ``\pderiv{log(p(y | X, \theta))}{\theta_k} = \frac{1}{2} * y_i * \pderiv{K_{ij}}{\theta_k} * y_j - \frac{1}{2}``
        ``* trace(K^{-1}_{ij}\pderiv{K_{ij}}{\theta_k})``
        Or equivalently, ``= \frac{1}{2} * trace([\alpha_i \alpha_j - K^{-1}_{ij}]*\pderiv{K_{ij}}{\theta_k})``,
        where ``\alpha_i = K^{-1}_{ij} * y_j``

        :return: grad_log_likelihood: i-th entry is ``\pderiv{LL(y | X, \theta)}{\theta_i}``
        :rtype: array of float64 with shape (num_hyperparameters)

        """
        covariance_matrix = python_utils.build_covariance_matrix(
            self._covariance,
            self._points_sampled,
            noise_variance=self._points_sampled_noise_variance,
        )
        K_chol = scipy.linalg.cho_factor(covariance_matrix, lower=True, overwrite_a=True)
        K_inv_y = scipy.linalg.cho_solve(K_chol, self._points_sampled_value)

        grad_hyperparameter_cov_matrix = python_utils.build_hyperparameter_grad_covariance_matrix(
            self._covariance,
            self._points_sampled,
        )
        grad_log_marginal = numpy.empty(self.num_hyperparameters)
        for k in xrange(self.num_hyperparameters):
            grad_cov_block = grad_hyperparameter_cov_matrix[..., k]
            # computing 0.5 * \alpha^T * grad_hyperparameter_cov_matrix * \alpha, where \alpha = K^-1 * y (aka K_inv_y)
            # temp_vec := grad_hyperparameter_cov_matrix * K_inv_y
            temp_vec = numpy.dot(grad_cov_block, K_inv_y)
            # computes 0.5 * K_inv_y^T * temp_vec
            grad_log_marginal[k] = 0.5 * numpy.dot(K_inv_y, temp_vec)

            # compute -0.5 * tr(K^-1 * dK/d\theta)
            temp = scipy.linalg.cho_solve(K_chol, grad_cov_block, overwrite_b=True)
            grad_log_marginal[k] -= 0.5 * temp.trace()
            # TODO(GH-180): this can be much faster if we form K^-1 explicitly (see below), but that is less accurate
            # grad_log_marginal[k] -= 0.5 * numpy.einsum('ij,ji', K_inv, grad_cov_block)

        return grad_log_marginal

    compute_grad_objective_function = compute_grad_log_likelihood

    def compute_hessian_log_likelihood(self):
        """We do not currently support computation of the (hyperparameter) hessian of log likelihood-like metrics."""
        raise NotImplementedError('Currently C++ does not expose Hessian computation of log likelihood-like metrics.')

    compute_hessian_objective_function = compute_hessian_log_likelihood
