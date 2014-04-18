# -*- coding: utf-8 -*-
"""Tools to compute ExpectedImprovement and optimize the next best point(s) to sample using EI through C++ calls.

This file contains a class to compute Expected Improvement + derivatives and a functions to solve the q,p-EI optimization problem.
The ExpectedImprovement class implements interfaces.ExpectedImproventInterface. The optimization functions are convenient
wrappers around the matching C++ calls.

See interfaces/expected_improvement_interface.py or gpp_math.hpp/cpp for further details on expected improvement.

"""
import numpy

import moe.build.GPP as C_GP
import moe.optimal_learning.EPI.src.python.cpp_wrappers.cpp_utils as cpp_utils
from moe.optimal_learning.EPI.src.python.interfaces.expected_improvement_interface import ExpectedImprovementInterface
from moe.optimal_learning.EPI.src.python.interfaces.optimization_interface import OptimizableInterface


def multistart_expected_improvement_optimization(ei_evaluator, ei_optimization_parameters, domain, num_samples_to_generate, points_to_sample=numpy.array([]), randomness=None, max_num_threads=1, status=None):
    """Solve the q,p-EI problem, returning the optimal set of q points to sample CONCURRENTLY in future experiments.

    When points_to_sample.shape[0] == 0 && num_samples_to_generate == 1, this function will use (fast) analytic EI computations.

    .. NOTE:: The following comments are copied from gpp_math.hpp, ComputeOptimalSetOfPointsToSample().

    Returns the optimal set of q points to sample CONCURRENTLY by solving the q,p-EI problem.  That is, we may want to run 4
    experiments at the same time and maximize the EI across all 4 experiments at once while knowing of 2 ongoing experiments
    (4,2-EI). This function handles this use case. Evaluation of q,p-EI (and its gradient) for q > 1 or p > 1 is expensive
    (requires monte-carlo iteration), so this method is usually very expensive.

    Compared to ComputeHeuristicSetOfPointsToSample() (``gpp_heuristic_expected_improvement_optimization.hpp``), this function
    makes no external assumptions about the underlying objective function. Instead, it utilizes a feature of the
    GaussianProcess that allows the GP to account for ongoing/incomplete experiments.

    If ``num_samples_to_generate = 1``, this is the same as ComputeOptimalPointToSampleWithRandomStarts().

    :param ei_evaluator: object specifying how to evaluate the expected improvement
    :type ei_evaluator: cpp_wrappers.expected_improvement.ExpectedImprovement
    :param ei_optimization_parameters: object specifying the desired optimization method (e.g., gradient descent, random search)
      and parameters controlling its behavior (e.g., tolerance, iterations, etc.)
    :type ei_optimization_parameters: cpp_wrappers.optimization_parameters.ExpectedImprovementOptimizationParameters
    :param domain: the domain over which to optimize (for the next best point(s) to sample)
    :type domain: DomainInterface, e.g., from cpp_wrappers/domain.py (TensorProductDomain, SimplexIntersectTensorProductDomain)
    :param num_samples_to_generate: how many simultaneous experiments you would like to run (i.e., the q in q,p-EI)
    :type num_samples_to_generate: int >= 1
    :param points_to_sample: points that are being sampled concurrently from the GP (i.e., the p in q,p-EI)
    :type points_to_sample: array of float64 with shape (num_to_sample, dim)
    :param randomness: RNGs used by C++ to generate initial guesses and as the source of normal random numbers when monte-carlo is used
    :type randomness: RandomnessSourceContainer (C++ object; e.g., from C_GP.RandomnessSourceContainer())
    :param max_num_threads: maximum number of threads to use, >= 1
    :type max_num_threads: int
    :param status: status messages from C++ (e.g., reporting on optimizer success, etc.)
    :type status: dict

    """
    # Create enough randomness sources if none are specified.
    if randomness is None:
        randomness = C_GP.RandomnessSourceContainer(max_num_threads)  # create randomness for max_num_threads
        randomness.SetRandomizedUniformGeneratorSeed(0)  # set seed based on less repeatable factors (e.g,. time)
        randomness.SetRandomizedNormalRNGSeed(0)  # set seed baesd on thread id & less repeatable factors (e.g,. time)

    # status must be an initialized dict for the call to C++.
    if status is None:
        status = {}

    ei_optimization_parameters.domain_type = domain._domain_type
    best_points_to_sample = C_GP.multistart_expected_improvement_optimization(
        ei_optimization_parameters,  # ExpectedImprovementOptimizationParameters object (see MOE_driver.py)
        ei_evaluator._gaussian_process._gaussian_process,  # C++ GaussianProcess object (e.g., from self._build_cpp_gaussian_process())
        cpp_utils.cppify(domain.domain_bounds),  # [lower, upper] bound pairs for each dimension
        cpp_utils.cppify(ei_evaluator._points_to_sample),  # points being sampled concurrently
        ei_evaluator._points_to_sample.shape[0],  # number of points to sample
        num_samples_to_generate,  # how many simultaneous experiments you would like to run
        ei_evaluator._best_so_far,  # best known value of objective so far
        ei_evaluator._num_mc_iterations,  # number of MC integration points in EI
        max_num_threads,
        randomness,  # C++ RandomnessSourceContainer that holds enough randomness sources for multithreading
        status,
    )

    # reform output to be a list of dim-dimensional points, dim = len(self.domain)
    return cpp_utils.uncppify(best_points_to_sample, (num_samples_to_generate, ei_evaluator.dim))


def _heuristic_expected_improvement_optimization(ei_evaluator, ei_optimization_parameters, domain, num_samples_to_generate, estimation_policy, randomness=None, max_num_threads=1, status=None):
    """Heuristically solve the q,0-EI problem (estimating multistart_expected_improvement_optimization()) using 1,0-EI solves.

    Consider this as an alternative when multistart_expected_improvement_optimization() is too expensive. Since this function
    kernalizes 1,0-EI, it always hits the analytic case; hence it is much faster than q,0-EI which requires monte-carlo.
    Users will probably call one of this function's wrappers (e.g., constant_liar_expected_improvement_optimization() or
    kriging_believer_expected_improvement_optimization()) instead of accessing this directly.

    Calls into heuristic_expected_improvement_optimization_wrapper in EPI/src/cpp/GPP_python_expected_improvement.cpp.

    .. NOTE:: The following comments are copied from gpp_heuristic_expected_improvement_optimization.hpp, ComputeHeuristicSetOfPointsToSample().

    It heuristically solves the q,0-EI optimization problem. As a reminder, that problem is finding the set of q points
    that maximizes the Expected Improvement (saved in the output, best_points_to_sample). Solving for q points simultaneously
    usually requires monte-carlo iteration and is expensive. The heuristic here solves q-EI as a sequence of 1-EI problems.
    We solve 1-EI, and then we *ASSUME* an objective function value at the resulting optima. This process is repeated q times.
    It is perhaps more clear in pseudocode:
    points_to_sample = {}  // This stays empty! We are only working with 1,0-EI solves
    for i = 0:num_samples_to_generate-1 {
      // First, solve the 1,0-EI problem*
      new_point = ComputeOptimalPointToSampleWithRandomStarts(gaussian_process, points_to_sample, other_parameters)
      // *Estimate* the objective function value at new_point
      new_function_value = ESTIMATED_OBJECTIVE_FUNCTION_VALUE(new_point, other_args)
      new_function_value_noise = ESTIMATED_NOISE_VARIANCE(new_point, other_args)
      // Write the estimated objective values to the GP as *truth*
      gaussian_process.AddPoint(new_point, new_function_value, new_function_value_noise)
      optimal_points_to_sample.append(new_point)
    }
    *Recall: each call to ComputeOptimalPointToSampleWithRandomStarts() (gpp_math.hpp) kicks off a round of MGD optimization of 1-EI.

    Note that ideally the estimated objective function value (and noise) would be measured from the real-world (e.g.,
    by running an experiment). Then this algorithm would be optimal. However, the estimate probably is not accurately
    representating of the true objective.

    The estimation is handled through the "estimation_policy" input. Passing a ConstantLiarEstimationPolicy or
    KrigingBelieverEstimationPolicy object to this function will produce the "Constant Liar" and "Kriging Believer"
    heuristics described in Ginsbourger 2008. The interface for estimation_policy is generic so users may specify
    other estimators as well.

    Contrast this appraoch with ComputeOptimalSetOfPointsToSample() (gpp_math.hpp) which solves all outputs of the q,0-EI
    problem simultaneously instead of one point (i.e., points_to_sample) at a time. This method is more accurate (b/c it
    does not attempt to estimate the behavior of the underlying objective function) but much more expensive (because it
    requires monte-carlo iteration).

    If num_samples_to_generate = 1, this is exactly the same as ComputeOptimalPointToSampleWithRandomStarts(); i.e.,
    both methods solve the 1-EI optimization problem the same way.

    Currently, during optimization, we recommend that the coordinates of the initial guesses not differ from the
    coordinates of the optima by more than about 1 order of magnitude. This is a very (VERY!) rough guideline for
    sizing the domain and num_multistarts; i.e., be wary of sets of initial guesses that cover the space too sparsely.

    Solution is guaranteed to lie within the region specified by "domain"; note that this may not be a
    local optima (i.e., the gradient may be substantially nonzero).

    .. WARNING:: this function fails if any step fails to find improvement! In that case, the return should not be
           read and status will report false.

    :param ei_evaluator: object specifying how to evaluate the expected improvement
    :type ei_evaluator: cpp_wrappers.expected_improvement.ExpectedImprovement
    :param ei_optimization_parameters: object specifying the desired optimization method and parameters controlling its behavior (e.g., tolerance, iterations, etc.)
    :type ei_optimization_parameters: cpp_wrappers.optimization_parameters.ExpectedImprovementOptimizationParameters
    :param domain: the domain over which to optimize (for the next best point(s) to sample)
    :type domain: DomainInterface, e.g., from cpp_wrappers/domain.py (TensorProductDomain, SimplexIntersectTensorProductDomain)
    :param num_samples_to_generate: how many simultaneous experiments you would like to run (i.e., the q in q,0-EI)
    :type num_samples_to_generate: int >= 1
    :param estimation_policy: the policy to use to produce (heuristic) objective function estimates during q,0-EI optimization
    :type estimation_policy: subclass of ObjectiveEstimationPolicyInterface (C++ pure abstract class)
       e.g., C_GP.KrigingBelieverEstimationPolicy, C_GP.ConstantLiarEstimationPolicy
       See gpp_heuristic_expected_improvement_optimization.hpp
    :param randomness: RNGs used by C++ to generate initial guesses
    :type randomness: RandomnessSourceContainer (C++ object; e.g., from C_GP.RandomnessSourceContainer())
    :param max_num_threads: maximum number of threads to use, >= 1
    :type max_num_threads: int
    :param status: status messages from C++ (e.g., reporting on optimizer success, etc.)
    :type status: dict

    """
    # Create enough randomness sources if none are specified.
    if randomness is None:
        randomness = C_GP.RandomnessSourceContainer(max_num_threads)  # create randomness for max_num_threads
        randomness.SetRandomizedUniformGeneratorSeed(0)  # set seed based on less repeatable factors (e.g,. time)
        randomness.SetRandomizedNormalRNGSeed(0)  # set seed baesd on thread id & less repeatable factors (e.g,. time)

    # status must be an initialized dict for the call to C++.
    if status is None:
        status = {}

    ei_optimization_parameters.domain_type = domain._domain_type
    best_points_to_sample = C_GP.heuristic_expected_improvement_optimization(
        ei_optimization_parameters,  # ExpectedImprovementOptimizationParameters object (see MOE_driver.py)
        ei_evaluator._gaussian_process._gaussian_process,  # C++ GaussianProcess object (e.g., from self._build_cpp_gaussian_process())
        cpp_utils.cppify(domain.domain_bounds),  # [lower, upper] bound pairs for each dimension
        estimation_policy,  # estimation policy to use for guessing objective function values (e.g., ConstantLiar, KrigingBeliever)
        num_samples_to_generate,  # how many simultaneous experiments you would like to run
        ei_evaluator._best_so_far,  # best known value of objective so far
        max_num_threads,
        randomness,  # C++ RandomnessSourceContainer that holds enough randomness sources for multithreading
        status,
    )

    # reform output to be a list of dim-dimensional points, dim = len(self.domain)
    return cpp_utils.uncppify(best_points_to_sample, (num_samples_to_generate, ei_evaluator.dim))


def constant_liar_expected_improvement_optimization(ei_evaluator, ei_optimization_parameters, domain, num_samples_to_generate, lie_value, lie_noise_variance=0.0, randomness=None, max_num_threads=1, status=None):
    """Heuristically solves q,0-EI using the Constant Liar policy; this wraps heuristic_expected_improvement_optimization().

    Note that this optimizer only uses the analytic 1,0-EI, so it is fast.

    See heuristic_expected_improvement_optimization() docs for general notes on how the heuristic optimization works.
    In this specific instance, we use the Constant Liar estimation policy.

    .. Note: comments copied from ConstantLiarEstimationPolicy in gpp_heuristic_expected_improvement_optimization.hpp.

    The "Constant Liar" objective function estimation policy is the simplest: it always returns the same value
    (Ginsbourger 2008). We call this the "lie. This object also allows users to associate a noise variance to
    the lie value.

    In Ginsbourger's work, the most common lie values have been the min and max of all previously observed objective
    function values; i.e., min, max of GP.points_sampled_value. The mean has also been considered.

    He also points out that larger lie values (e.g., max of prior measurements) will lead methods like
    ComputeEstimatedSetOfPointsToSample() to be more explorative and vice versa.

    :param ei_evaluator: object specifying how to evaluate the expected improvement
    :type ei_evaluator: cpp_wrappers.expected_improvement.ExpectedImprovement
    :param ei_optimization_parameters: object specifying the desired optimization method and parameters controlling its behavior (e.g., tolerance, iterations, etc.)
    :type ei_optimization_parameters: cpp_wrappers.optimization_parameters.ExpectedImprovementOptimizationParameters
    :param domain: the domain over which to optimize (for the next best point(s) to sample)
    :type domain: DomainInterface, e.g., from cpp_wrappers/domain.py (TensorProductDomain, SimplexIntersectTensorProductDomain)
    :param num_samples_to_generate: how many simultaneous experiments you would like to run (i.e., the q in q,0-EI)
    :type num_samples_to_generate: int >= 1
    :param lie_value: the "constant lie" that this estimator should return
    :type lie_value: float64
    :param lie_noise_variance: the noise_variance to associate to the lie_value (MUST be >= 0.0)
    :type lie_noise_variance: float64
    :param randomness: RNGs used by C++ to generate initial guesses
    :type randomness: RandomnessSourceContainer (C++ object; e.g., from C_GP.RandomnessSourceContainer())
    :param max_num_threads: maximum number of threads to use, >= 1
    :type max_num_threads: int
    :param status: status messages from C++ (e.g., reporting on optimizer success, etc.)
    :type status: dict

    """
    estimation_policy = C_GP.ConstantLiarEstimationPolicy(lie_value, lie_noise_variance)
    return _heuristic_expected_improvement_optimization(ei_evaluator, ei_optimization_parameters, domain, num_samples_to_generate, estimation_policy, randomness=randomness, max_num_threads=max_num_threads, status=status)


def kriging_believer_expected_improvement_optimization(ei_evaluator, ei_optimization_parameters, domain, num_samples_to_generate, std_deviation_coef=0.0, kriging_noise_variance=0.0, randomness=None, max_num_threads=1, status=None):
    """Heuristically solves q,0-EI using the Kriging Believer policy; this wraps heuristic_expected_improvement_optimization().

    Note that this optimizer only uses the analytic 1,0-EI, so it is fast.

    See heuristic_expected_improvement_optimization() docs for general notes on how the heuristic optimization works.
    In this specific instance, we use the Kriging Believer estimation policy.

    .. Note: comments copied from KrigingBelieverEstimationPolicy in gpp_heuristic_expected_improvement_optimization.hpp.

    The "Kriging Believer" objective function estimation policy uses the Gaussian Process (i.e., the prior)
    to produce objective function estimates. The simplest method is to trust the GP completely:
    estimate = GP.mean(point)
    This follows the usage in Ginsbourger 2008. Users may also want the estimate to depend on the GP variance
    at the evaluation point, so that the estimate reflects how confident the GP is in the prediction. Users may
    also specify std_devation_ceof:
    estimate = GP.mean(point) + std_deviation_coef * GP.variance(point)
    Note that the coefficient is signed, and analogously to ConstantLiar, larger positive values are more
    explorative and larger negative values are more exploitive.

    This object also allows users to associate a noise variance to the lie value.

    :param ei_evaluator: object specifying how to evaluate the expected improvement
    :type ei_evaluator: cpp_wrappers.expected_improvement.ExpectedImprovement
    :param ei_optimization_parameters: object specifying the desired optimization method and parameters controlling its behavior (e.g., tolerance, iterations, etc.)
    :type ei_optimization_parameters: cpp_wrappers.optimization_parameters.ExpectedImprovementOptimizationParameters
    :param domain: the domain over which to optimize (for the next best point(s) to sample)
    :type domain: DomainInterface, e.g., from cpp_wrappers/domain.py (TensorProductDomain, SimplexIntersectTensorProductDomain)
    :param num_samples_to_generate: how many simultaneous experiments you would like to run (i.e., the q in q,0-EI)
    :type num_samples_to_generate: int >= 1
    :param std_deviation_coef: the relative amount of bias (in units of GP std deviation) to introduce into the GP mean
    :type std_deviation_coef: float64
    :param kriging_noise_variance: the noise_variance to associate to each function value estimate (MUST be >= 0.0)
    :type kriging_noise_variance: float64
    :param randomness: RNGs used by C++ to generate initial guesses
    :type randomness: RandomnessSourceContainer (C++ object; e.g., from C_GP.RandomnessSourceContainer())
    :param max_num_threads: maximum number of threads to use, >= 1
    :type max_num_threads: int
    :param status: status messages from C++ (e.g., reporting on optimizer success, etc.)
    :type status: dict

    """
    estimation_policy = C_GP.KrigingBelieverEstimationPolicy(std_deviation_coef, kriging_noise_variance)
    return _heuristic_expected_improvement_optimization(ei_evaluator, ei_optimization_parameters, domain, num_samples_to_generate, estimation_policy, randomness=randomness, max_num_threads=max_num_threads, status=status)


def evaluate_expected_improvement_at_point_list(ei_evaluator, points_to_evaluate, points_to_sample=numpy.array([]), randomness=None, max_num_threads=1, status=None):
    """Evaluate Expected Improvement (1,p-EI) over a specified list of ``points_to_evaluate``.

    Generally gradient descent is preferred but when they fail to converge this may be the only "robust" option.
    This function is also useful for plotting or debugging purposes (just to get a bunch of EI values).

    :param ei_evaluator: object specifying how to evaluate the expected improvement
    :type ei_evaluator: cpp_wrappers.expected_improvement.ExpectedImprovement
    :param points_to_sample: points that are being sampled concurrently from the GP (i.e., the p in q,p-EI)
    :type points_to_sample: array of float64 with shape (num_to_sample, dim)
    :param randomness: RNGs used by C++ to generate initial guesses and as the source of normal random numbers when monte-carlo is used
    :type randomness: RandomnessSourceContainer (C++ object; e.g., from C_GP.RandomnessSourceContainer())
    :param max_num_threads: maximum number of threads to use, >= 1
    :type max_num_threads: int
    :param status: status messages from C++ (e.g., reporting on optimizer success, etc.)
    :type status: dict
    :return: EI evaluated at each of points_to_evaluate
    :rtype: array of float64 with shape )points_to_evaluate.shape[0])

    """
    # Create enough randomness sources if none are specified.
    if randomness is None:
        randomness = C_GP.RandomnessSourceContainer(max_num_threads)  # create randomness for max_num_threads
        randomness.SetRandomizedUniformGeneratorSeed(0)  # set seed based on less repeatable factors (e.g,. time)
        randomness.SetRandomizedNormalRNGSeed(0)  # set seed baesd on thread id & less repeatable factors (e.g,. time)

    # status must be an initialized dict for the call to C++.
    if status is None:
        status = {}

    ei_values = C_GP.evaluate_EI_at_point_list(
        ei_evaluator._gaussian_process._gaussian_process,  # C++ GaussianProcess object (e.g., from self._build_cpp_gaussian_process())
        cpp_utils.cppify(points_to_evaluate),  # points at which to evaluate EI
        cpp_utils.cppify(ei_evaluator._points_to_sample),  # points being sampled concurrently
        points_to_evaluate.shape[0],  # number of points to evaluate
        ei_evaluator._points_to_sample.shape[0],  # number of points to sample
        ei_evaluator._best_so_far,  # best known value of objective so far
        ei_evaluator._num_mc_iterations,  # number of MC integration points in EI
        max_num_threads,
        randomness,  # C++ RandomnessSourceContainer that holds enough randomness sources for multithreading
        status,
    )
    return numpy.array(ei_values)


class ExpectedImprovement(ExpectedImprovementInterface, OptimizableInterface):

    r"""Implementation of Expected Improvement computation via C++ wrappers: EI and its gradient at specified point(s) sampled from a GaussianProcess.

    A class to encapsulate the computation of expected improvement and its spatial gradient using points sampled from an
    associated GaussianProcess. The general EI computation requires monte-carlo integration; it can support q,p-EI optimization.
    It is designed to work with any GaussianProcess.

    See interfaces/expected_improvement_interface.py docs for further details.

    """

    def __init__(self, gaussian_process, current_point, points_to_sample=numpy.array([]), num_mc_iterations=1000, randomness=None):
        """Construct an ExpectedImprovement object that knows how to call C++ for evaluation of member functions.

        :param gaussian_process: GaussianProcess describing
        :type gaussian_process: cpp_wrappers.GaussianProcess object
        :param current_point: point at which to compute EI (i.e., q in q,p-EI)
        :type current_point: array of float64 with shape (dim)
        :param points_to_sample: points which are being sampled concurrently (i.e., p in q,p-EI)
        :type points_to_sample: array of float64 with shape (num_to_sample, dim)
        :param num_mc_iterations: number of monte-carlo iterations to use (when monte-carlo integration is used to compute EI)
        :type num_mc_iterations: int > 0
        :param randomness: RNGs used by C++ as the source of normal random numbers when monte-carlo is used
        :type randomness: RandomnessSourceContainer (C++ object; e.g., from C_GP.RandomnessSourceContainer())

        """
        self._num_mc_iterations = num_mc_iterations
        self._gaussian_process = gaussian_process
        if gaussian_process._historical_data.points_sampled_value.size > 0:
            self._best_so_far = numpy.amin(gaussian_process._historical_data.points_sampled_value)
        else:
            self._best_so_far = numpy.finfo(numpy.float64).max

        self._current_point = numpy.copy(current_point)
        self._points_to_sample = numpy.copy(points_to_sample)

        if randomness is None:
            self._randomness = C_GP.RandomnessSourceContainer(1)  # create randomness for only 1 thread
            self._randomness.SetRandomizedUniformGeneratorSeed(0)  # set seed based on less repeatable factors (e.g,. time)
            self._randomness.SetRandomizedNormalRNGSeed(0)  # set seed baesd on thread id & less repeatable factors (e.g,. time)
        else:
            self._randomness = randomness

    @property
    def dim(self):
        """Return the number of spatial dimensions."""
        return self._gaussian_process.dim

    @property
    def problem_size(self):
        """Return the number of independent parameters to optimize."""
        return self.dim

    def get_current_point(self):
        """Get the current_point (array of float64 with shape (problem_size)) at which this object is evaluating the objective function, ``f(x)``."""
        return numpy.copy(self._current_point)

    def set_current_point(self, current_point):
        """Set current_point to the specified point; ordering must match.

        :param current_point: current_point at which to evaluate the objective function, ``f(x)``
        :type current_point: array of float64 with shape (problem_size)

        """
        self._current_point = numpy.copy(current_point)

    def compute_expected_improvement(self, force_monte_carlo=False):
        r"""Compute the expected improvement at ``current_point``, with ``points_to_sample`` concurrent points being sampled.

        .. Note:: These comments were copied from this's superclass in expected_improvement_interface.py.

        ``current_points`` is the q and points_to_sample is the p in q,p-EI.

        We compute ``EI(Xs) = E_n[[f^*_n(X) - min(f(Xs_1),...,f(Xs_m))]^+]``, where ``Xs`` are potential points
        to sample and ``X`` are already sampled points.  The ``^+`` indicates that the expression in the expectation evaluates to 0
        if it is negative.  ``f^*(X)`` is the MINIMUM over all known function evaluations (``points_sampled_value``), whereas
        ``f(Xs)`` are *GP-predicted* function evaluations.

        The EI is the expected improvement in the current best known objective function value that would result from sampling
        at ``points_to_sample``.

        In general, the EI expression is complex and difficult to evaluate; hence we use Monte-Carlo simulation to approximate it.

        :param force_monte_carlo: whether to force monte carlo evaluation (vs using fast/accurate analytic eval when possible)
        :type force_monte_carlo: boolean
        :return: value of EI evaluated at ``current_point``
        :rtype: float64

        """
        num_points = 1 + self._points_to_sample.shape[0]
        union_of_points = numpy.reshape(numpy.append(self._current_point, self._points_to_sample), (num_points, self.dim))

        return C_GP.compute_expected_improvement(
            self._gaussian_process._gaussian_process,  # C++ GaussianProcess object (e.g., from self._build_cpp_gaussian_process())
            cpp_utils.cppify(union_of_points),  # points to sample
            num_points,  # number of points to sample
            self._num_mc_iterations,
            self._best_so_far,  # best known value of objective so far
            force_monte_carlo,
            self._randomness,
        )

    def compute_objective_function(self, **kwargs):
        """Wrapper for compute_expected_improvement; see that function's docstring."""
        return self.compute_expected_improvement(**kwargs)

    def compute_grad_expected_improvement(self, force_monte_carlo=False):
        r"""Compute the gradient of expected improvement at ``current_point`` wrt ``current_point``, with ``points_to_sample`` concurrent samples.

        .. Note:: These comments were copied from this's superclass in expected_improvement_interface.py.

        ``current_points`` is the q and points_to_sample is the p in q,p-EI.

        In general, the expressions for gradients of EI are complex and difficult to evaluate; hence we use
        Monte-Carlo simulation to approximate it.

        :param force_monte_carlo: whether to force monte carlo evaluation (vs using fast/accurate analytic eval when possible)
        :type force_monte_carlo: boolean
        :return: gradient of EI, i-th entry is ``\pderiv{EI(x)}{x_i}`` where ``x`` is ``current_point``
        :rtype: array of float64 with shape (dim)

        """
        grad_EI = C_GP.compute_grad_expected_improvement(
            self._gaussian_process._gaussian_process,  # C++ GaussianProcess object (e.g., from self._build_cpp_gaussian_process())
            cpp_utils.cppify(self._points_to_sample),  # points to sample
            self._points_to_sample.shape[0],  # number of points to sample
            self._num_mc_iterations,
            self._best_so_far,  # best known value of objective so far
            force_monte_carlo,
            self._randomness,
            cpp_utils.cppify(self._current_point),
        )
        return numpy.array(grad_EI)

    def compute_grad_objective_function(self, **kwargs):
        """Wrapper for compute_grad_expected_improvement; see that function's docstring."""
        return self.compute_grad_expected_improvement(**kwargs)

    def compute_hessian_objective_function(self, **kwargs):
        """We do not currently support computation of the (spatial) hessian of Expected Improvement."""
        raise NotImplementedError('Currently we cannot compute the hessian of expected improvement.')
