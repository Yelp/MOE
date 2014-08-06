# -*- coding: utf-8 -*-
r"""Tools to compute log likelihood-like measures of model fit and optimize them (wrt the hyperparameters of covariance).

See the file comments in :mod:`moe.optimal_learning.python.interfaces.log_likelihood_interface`
for an overview of log likelihood-like metrics and their role
in model selection. This file provides hooks to implementations of two such metrics in C++: Log Marginal Likelihood and
Leave One Out Cross Validation Log Pseudo-Likelihood.

.. Note:: This is a copy of the file comments in gpp_model_selection.hpp.
  These comments are copied in :mod:`moe.optimal_learning.python.python_version.log_likelihood`.
  See this file's comments and interfaces.log_likelihood_interface for more details as well as the hpp and corresponding .cpp file.

**a. LOG MARGINAL LIKELIHOOD (LML)**

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

**b. LEAVE ONE OUT CROSS VALIDATION (LOO-CV)**

(Rasmussen & Williams, Chp 5.4.2)
In cross validation, we split the training data, X, into two sets--a sub-training set and a validation set.  Then we
train a model on the sub-training set and test it on the validation set.  Since the validation set comes from the
original training data, we can compute the error.  In effect we are examining how well the model explains itself.

Leave One Out CV works by considering n different validation sets, one at a time.  Each point of X takes a turn
being the sole member of the validation set.  Then for each validation set, we compute a log pseudo-likelihood, measuring
how probable that validation set is given the remaining training data and model hyperparameters.

Again, we can maximize this quanitity over hyperparameters to help us choose the "right" set for the GP.

"""
import copy

import numpy

import moe.build.GPP as C_GP
from moe.optimal_learning.python.constant import DEFAULT_MAX_NUM_THREADS
from moe.optimal_learning.python.cpp_wrappers import cpp_utils
from moe.optimal_learning.python.interfaces.log_likelihood_interface import GaussianProcessLogLikelihoodInterface
from moe.optimal_learning.python.interfaces.optimization_interface import OptimizableInterface


def multistart_hyperparameter_optimization(
        log_likelihood_optimizer,
        num_multistarts,
        randomness=None,
        max_num_threads=DEFAULT_MAX_NUM_THREADS,
        status=None,
):
    r"""Select the hyperparameters that maximize the specified log likelihood measure of model fit (over the historical data) within the specified domain.

    .. Note:: The following comments are copied to
      :mod:`moe.optimal_learning.python.python_version.log_likelihood.multistart_hyperparameter_optimization`.

    See :class:`moe.optimal_learning.python.cpp_wrappers.log_likelihood.GaussianProcessLogMarginalLikelihood` and
    :class:`moe.optimal_learning.python.cpp_wrappers.log_likelihood.GaussianProcessLeaveOneOutLogLikelihood`
    for an overview of some example log likelihood-like measures.

    Optimizers are: null ('dumb' search), gradient descent, newton
    Newton is the suggested optimizer.

    'dumb' search means this will just evaluate the objective log likelihood measure at num_multistarts 'points'
    (hyperparameters) in the domain, uniformly sampled using latin hypercube sampling.
    The hyperparameter_optimizer_parameters input specifies the desired optimization technique as well as parameters controlling
    its behavior (see :mod:`moe.optimal_learning.python.cpp_wrappers.optimization`).

    See gpp_python_common.cpp for C++ enum declarations laying out the options for objective and optimizer types.

    Currently, during optimization, we recommend that the coordinates of the initial guesses not differ from the
    coordinates of the optima by more than about 1 order of magnitude. This is a very (VERY!) rough guideline for
    sizing the domain and gd_parameters.num_multistarts; i.e., be wary of sets of initial guesses that cover the space too sparsely.

    Note that the domain here must be specified in LOG-10 SPACE!

    Solution is guaranteed to lie within the region specified by "domain"; note that this may not be a
    true optima (i.e., the gradient may be substantially nonzero).

    .. WARNING:: this function fails if NO improvement can be found!  In that case,
       the output will always be the first randomly chosen point. status will report failure.

    :param ei_optimizer: object that optimizes (e.g., gradient descent, newton) log likelihood over a domain
    :type ei_optimizer: cpp_wrappers.optimization.*Optimizer object
    :param num_multistarts: number of times to multistart ``ei_optimizer`` (UNUSED, data is in log_likelihood_optimizer.optimizer_parameters)
    :type num_multistarts: int > 0
    :param randomness: RNGs used by C++ to generate initial guesses
    :type randomness: RandomnessSourceContainer (C++ object; e.g., from C_GP.RandomnessSourceContainer())
    :param max_num_threads: maximum number of threads to use, >= 1
    :type max_num_threads: int > 0
    :param status: (output) status messages from C++ (e.g., reporting on optimizer success, etc.)
    :type status: dict
    :return: hyperparameters that maximize the specified log likelihood measure within the specified domain
    :rtype: array of float64 with shape (log_likelihood_optimizer.objective_function.num_hyperparameters)

    """
    # Create enough randomness sources if none are specified.
    if randomness is None:
        randomness = C_GP.RandomnessSourceContainer(max_num_threads)
        # Set seed based on less repeatable factors (e.g,. time)
        randomness.SetRandomizedUniformGeneratorSeed(0)
        randomness.SetRandomizedNormalRNGSeed(0)

    # status must be an initialized dict for the call to C++.
    if status is None:
        status = {}

    # C++ expects the domain in log10 space and in list form
    domain_bounds_log10 = numpy.log10(log_likelihood_optimizer.domain._domain_bounds)

    hyperparameters_opt = C_GP.multistart_hyperparameter_optimization(
        log_likelihood_optimizer.optimizer_parameters,
        cpp_utils.cppify(domain_bounds_log10),
        cpp_utils.cppify(log_likelihood_optimizer.objective_function._points_sampled),
        cpp_utils.cppify(log_likelihood_optimizer.objective_function._points_sampled_value),
        log_likelihood_optimizer.objective_function.dim,
        log_likelihood_optimizer.objective_function._num_sampled,
        cpp_utils.cppify_hyperparameters(log_likelihood_optimizer.objective_function.hyperparameters),
        cpp_utils.cppify(log_likelihood_optimizer.objective_function._points_sampled_noise_variance),
        max_num_threads,
        randomness,
        status,
    )
    return numpy.array(hyperparameters_opt)


def evaluate_log_likelihood_at_hyperparameter_list(
        log_likelihood_evaluator,
        hyperparameters_to_evaluate,
        max_num_threads=DEFAULT_MAX_NUM_THREADS,
        status=None,
):
    """Compute the specified log likelihood measure at each input set of hyperparameters.

    Generally Newton or gradient descent is preferred but when they fail to converge this may be the only "robust" option.
    This function is also useful for plotting or debugging purposes (just to get a bunch of log likelihood values).

    Calls into evaluate_log_likelihood_at_hyperparameter_list() in cpp/GPP_python_model_selection.cpp.

    :param log_likelihood_evaluator: object specifying which log likelihood measure to evaluate
    :type log_likelihood_evaluator: cpp_wrappers.log_likelihood.LogLikelihood
    :param hyperparameters_to_evaluate: the hyperparameters at which to compute the specified log likelihood
    :type hyperparameters_to_evaluate: array of float64 with shape (num_to_eval, log_likelihood_evaluator.num_hyperparameters)
    :param max_num_threads: maximum number of threads to use, >= 1
    :type max_num_threads: int > 0
    :param status: (output) status messages from C++ (e.g., reporting on optimizer success, etc.)
    :type status: dict
    :return: log likelihood value at each specified set of hyperparameters
    :rtype: array of float64 with shape (hyperparameters_to_evaluate.shape[0])

    """
    # status must be an initialized dict for the call to C++.
    if status is None:
        status = {}

    # We could just call log_likelihood_evaluator.compute_log_likelihood() in a loop, but instead we do
    # the looping in C++ where it can be multithreaded.
    log_likelihood_list = C_GP.evaluate_log_likelihood_at_hyperparameter_list(
        cpp_utils.cppify(hyperparameters_to_evaluate),
        cpp_utils.cppify(log_likelihood_evaluator._points_sampled),
        cpp_utils.cppify(log_likelihood_evaluator._points_sampled_value),
        log_likelihood_evaluator.dim,
        log_likelihood_evaluator._num_sampled,
        log_likelihood_evaluator.objective_type,
        cpp_utils.cppify_hyperparameters(log_likelihood_evaluator.hyperparameters),
        cpp_utils.cppify(log_likelihood_evaluator._points_sampled_noise_variance),
        hyperparameters_to_evaluate.shape[0],
        max_num_threads,
        status,
    )
    return numpy.array(log_likelihood_list)


class GaussianProcessLogLikelihood(GaussianProcessLogLikelihoodInterface, OptimizableInterface):

    r"""Class for computing log likelihood-like measures of model fit via C++ wrappers (currently log marginal and leave one out cross validation).

    See :class:`moe.optimal_learning.python.cpp_wrappers.log_likelihood.GaussianProcessLogMarginalLikelihood` and
    :class:`moe.optimal_learning.python.cpp_wrappers.log_likelihood.GaussianProcessLeaveOneOutLogLikelihood`
    classes below for some more details on these metrics. Users may find it more convenient to
    construct these objects instead of a :class:`~moe.optimal_learning.python.cpp_wrappers.log_likelihood.GaussianProcessLogLikelihood`
    object directly. Since these various metrics are fairly different, the member function docs
    in this class will remain generic.

    .. Note:: Equivalent methods of :class:`moe.optimal_learning.python.interfaces.log_likelihood_interface.GaussianProcessLogLikelihoodInterface` and
      :class:`moe.optimal_learning.python.interfaces.optimization_interface.OptimizableInterface`
      are aliased below (e.g., :class:`~moe.optimal_learning.python.cpp_wrappers.log_likelihood.GaussianProcessLogLikelihood.problem_size` and
      :class:`~moe.optimal_learning.python.cpp_wrappers.log_likelihood.GaussianProcessLogLikelihood.num_hyperparameters`,
      :class:`~moe.optimal_learning.python.cpp_wrappers.log_likelihood.GaussianProcessLogLikelihood.compute_log_likelihood` and
      :class:`~moe.optimal_learning.python.cpp_wrappers.log_likelihood.GaussianProcessLogLikelihood.compute_objective_function`, etc).

    See gpp_model_selection.hpp/cpp for further overview and in-depth discussion, respectively.

    """

    def __init__(self, covariance_function, historical_data, log_likelihood_type=C_GP.LogLikelihoodTypes.log_marginal_likelihood):
        """Construct a LogLikelihood object that knows how to call C++ for evaluation of member functions.

        :param covariance_function: covariance object encoding assumptions about the GP's behavior on our data
        :type covariance_function: :class:`moe.optimal_learning.python.interfaces.covariance_interface.CovarianceInterface` subclass
          (e.g., from :mod:`moe.optimal_learning.python.cpp_wrappers.covariance`).
        :param historical_data: object specifying the already-sampled points, the objective value at those points, and the noise variance associated with each observation
        :type historical_data: :class:`moe.optimal_learning.python.data_containers.HistoricalData` object
        :param log_likelihood_type: enum specifying which log likelihood measure to compute
        :type log_likelihood_type: GPP.LogLikelihoodTypes

        """
        self._covariance = copy.deepcopy(covariance_function)
        self._historical_data = copy.deepcopy(historical_data)

        self.objective_type = log_likelihood_type

    @property
    def dim(self):
        """Return the number of spatial dimensions."""
        return self._historical_data.dim

    @property
    def num_hyperparameters(self):
        """Return the number of hyperparameters aka the number of independent parameters to optimize."""
        return self._covariance.num_hyperparameters

    #: an alias for num_hyperparameters to fulfill :class:`moe.optimal_learning.python.interfaces.optimization_interface.OptimizableInterface`
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
    def _num_sampled(self):
        """Return the number of sampled points."""
        return self._historical_data.num_sampled

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
        r"""Compute the objective_type measure at the specified hyperparameters.

        :return: value of log_likelihood evaluated at hyperparameters (``LL(y | X, \theta)``)
        :rtype: float64

        """
        return C_GP.compute_log_likelihood(
            cpp_utils.cppify(self._points_sampled),
            cpp_utils.cppify(self._points_sampled_value),
            self.dim,
            self._num_sampled,
            self.objective_type,
            cpp_utils.cppify_hyperparameters(self.hyperparameters),
            cpp_utils.cppify(self._points_sampled_noise_variance),
        )

    compute_objective_function = compute_log_likelihood

    def compute_grad_log_likelihood(self):
        r"""Compute the gradient (wrt hyperparameters) of the objective_type measure at the specified hyperparameters.

        :return: grad_log_likelihood: i-th entry is ``\pderiv{LL(y | X, \theta)}{\theta_i}``
        :rtype: array of float64 with shape (num_hyperparameters)

        """
        grad_log_marginal = C_GP.compute_hyperparameter_grad_log_likelihood(
            cpp_utils.cppify(self._points_sampled),
            cpp_utils.cppify(self._points_sampled_value),
            self.dim,
            self._num_sampled,
            self.objective_type,
            cpp_utils.cppify_hyperparameters(self.hyperparameters),
            cpp_utils.cppify(self._points_sampled_noise_variance),
        )
        return numpy.array(grad_log_marginal)

    compute_grad_objective_function = compute_grad_log_likelihood

    def compute_hessian_log_likelihood(self):
        """We do not currently support computation of the (hyperparameter) hessian of log likelihood-like metrics."""
        raise NotImplementedError('Currently C++ does not expose Hessian computation of log likelihood-like metrics.')

    compute_hessian_objective_function = compute_hessian_log_likelihood


class GaussianProcessLogMarginalLikelihood(GaussianProcessLogLikelihood):

    r"""Class for computing the Log Marginal Likelihood, ``log(p(y | X, \theta))``.

    That is, the probability of observing the training values, y, given the training points, X,
    and hyperparameters (of the covariance function), ``\theta``.

    This is a measure of how likely it is that the observed values came from our Gaussian Process Prior.

    .. Note:: This is a copy of LogMarginalLikelihoodEvaluator's class comments in gpp_model_selection.hpp.
      See this file's comments and :mod:`moe.optimal_learning.python.interfaces.log_likelihood_interface`
      for more details as well as the hpp and corresponding .cpp file.

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

    """

    def __init__(self, covariance_function, historical_data):
        """Construct a LogLikelihood object configured for Log Marginal Likelihood computation; see superclass ctor for details."""
        super(GaussianProcessLogMarginalLikelihood, self).__init__(
            covariance_function,
            historical_data,
            log_likelihood_type=C_GP.LogLikelihoodTypes.log_marginal_likelihood,
            )


class GaussianProcessLeaveOneOutLogLikelihood(GaussianProcessLogLikelihood):

    r"""Class for computing the Leave-One-Out Cross Validation (LOO-CV) Log Pseudo-Likelihood.

    Given a particular covariance function (including hyperparameters) and training data ((point, function value, measurement noise)
    tuples), the log LOO-CV pseudo-likelihood expresses how well the model explains itself.

    .. Note:: This is a copy of LeaveOneOutLogLikelihoodEvaluator's class comments in gpp_model_selection.hpp.
      See this file's comments and :mod:`moe.optimal_learning.python.interfaces.log_likelihood_interface`
      for more details as well as the hpp and corresponding .cpp file.

    That is, cross validation involves splitting the training set into a sub-training set and a validation set.  Then we measure
    the log likelihood that a model built on the sub-training set could produce the values in the validation set.

    Leave-One-Out CV does this process ``|y|`` times: on the ``i``-th run, the sub-training set is ``(X, y)`` with the ``i``-th point removed
    and the validation set is the ``i``-th point.  Then the predictive performance of each sub-model are aggregated into a
    psuedo-likelihood.

    This quantity primarily deals with the internal consistency of the model--how well it explains itself.  The LOO-CV
    likelihood gives an "estimate for the predictive probability, whether or not the assumptions of the model may be
    fulfilled." It is a more frequentist view of model selection. (Rasmussen & Williams p118)
    See Rasmussen & Williams 5.3 and 5.4.2 for more details.

    As with the log marginal likelihood, we can use this quantity to measure the performance of our model.  We can also
    maximize it (via hyperparameter modifications or covariance function changes) to improve model performance.
    It has also been argued that LOO-CV is better at detecting model mis-specification (e.g., wrong covariance function)
    than log marginal measures (Rasmussen & Williams p118).

    """

    def __init__(self, covariance_function, historical_data):
        """Construct a LogLikelihood object configured for Leave One Out Cross Validation Log Pseudo-Likelihood computation; see superclass ctor for details."""
        super(GaussianProcessLeaveOneOutLogLikelihood, self).__init__(
            covariance_function,
            historical_data,
            log_likelihood_type=C_GP.LogLikelihoodTypes.leave_one_out_log_likelihood,
            )

    def compute_hessian_log_likelihood(self, hyperparameters):
        """The (hyperparameter) hessian of LOO-CV has not been implemented in C++ yet."""
        raise NotImplementedError('Currently C++ does not support Hessian computation of LOO-CV.')
