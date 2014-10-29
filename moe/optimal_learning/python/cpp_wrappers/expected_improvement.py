# -*- coding: utf-8 -*-
"""Tools to compute ExpectedImprovement and optimize the next best point(s) to sample using EI through C++ calls.

This file contains a class to compute Expected Improvement + derivatives and a functions to solve the q,p-EI optimization problem.
The :class:`moe.optimal_learning.python.cpp_wrappers.expected_improvement.ExpectedImprovement`
class implements :class:`moe.optimal_learning.python.interfaces.expected_improvement_interface.ExpectedImproventInterface`.
The optimization functions are convenient wrappers around the matching C++ calls.

See :class:`moe.optimal_learning.python.interfaces.expected_improvement_interface` or
gpp_math.hpp/cpp for further details on expected improvement.

"""
import numpy

import moe.build.GPP as C_GP
from moe.optimal_learning.python.constant import DEFAULT_EXPECTED_IMPROVEMENT_MC_ITERATIONS, DEFAULT_MAX_NUM_THREADS
import moe.optimal_learning.python.cpp_wrappers.cpp_utils as cpp_utils
from moe.optimal_learning.python.interfaces.expected_improvement_interface import ExpectedImprovementInterface
from moe.optimal_learning.python.interfaces.optimization_interface import OptimizableInterface


def multistart_expected_improvement_optimization(
        ei_optimizer,
        num_multistarts,
        num_to_sample,
        use_gpu=False,
        which_gpu=0,
        randomness=None,
        max_num_threads=DEFAULT_MAX_NUM_THREADS,
        status=None,
):
    """Solve the q,p-EI problem, returning the optimal set of q points to sample CONCURRENTLY in future experiments.

    When ``points_being_sampled.size == 0 && num_to_sample == 1``, this function will use (fast) analytic EI computations.

    .. NOTE:: The following comments are copied from gpp_math.hpp, ComputeOptimalPointsToSample().
      These comments are copied into
      :func:`moe.optimal_learning.python.python_version.expected_improvement.multistart_expected_improvement_optimization`

    This is the primary entry-point for EI optimization in the optimal_learning library. It offers our best shot at
    improving robustness by combining higher accuracy methods like gradient descent with fail-safes like random/grid search.

    Returns the optimal set of q points to sample CONCURRENTLY by solving the q,p-EI problem.  That is, we may want to run 4
    experiments at the same time and maximize the EI across all 4 experiments at once while knowing of 2 ongoing experiments
    (4,2-EI). This function handles this use case. Evaluation of q,p-EI (and its gradient) for q > 1 or p > 1 is expensive
    (requires monte-carlo iteration), so this method is usually very expensive.

    Compared to ComputeHeuristicPointsToSample() (``gpp_heuristic_expected_improvement_optimization.hpp``), this function
    makes no external assumptions about the underlying objective function. Instead, it utilizes a feature of the
    GaussianProcess that allows the GP to account for ongoing/incomplete experiments.

    If ``num_to_sample = 1``, this is the same as ComputeOptimalPointsToSampleWithRandomStarts().

    The option of using GPU to compute general q,p-EI via MC simulation is also available. To enable it, make sure you have
    installed GPU components of MOE, otherwise, it will throw Runtime excpetion.

    :param ei_optimizer: object that optimizes (e.g., gradient descent, newton) EI over a domain
    :type ei_optimizer: cpp_wrappers.optimization.*Optimizer object
    :param num_multistarts: number of times to multistart ``ei_optimizer`` (UNUSED, data is in ei_optimizer.optimizer_parameters)
    :type num_multistarts: int > 0
    :param num_to_sample: how many simultaneous experiments you would like to run (i.e., the q in q,p-EI)
    :type num_to_sample: int >= 1
    :param use_gpu: set to True if user wants to use GPU for MC simulation
    :type use_gpu: bool
    :param which_gpu: GPU device ID
    :type which_gpu: int >= 0
    :param randomness: RNGs used by C++ to generate initial guesses and as the source of normal random numbers when monte-carlo is used
    :type randomness: RandomnessSourceContainer (C++ object; e.g., from C_GP.RandomnessSourceContainer())
    :param max_num_threads: maximum number of threads to use, >= 1
    :type max_num_threads: int > 0
    :param status: (output) status messages from C++ (e.g., reporting on optimizer success, etc.)
    :type status: dict
    :return: point(s) that maximize the expected improvement (solving the q,p-EI problem)
    :rtype: array of float64 with shape (num_to_sample, ei_optimizer.objective_function.dim)

    """
    # Create enough randomness sources if none are specified.
    if randomness is None:
        randomness = C_GP.RandomnessSourceContainer(max_num_threads)
        # Set seeds based on less repeatable factors (e.g,. time)
        randomness.SetRandomizedUniformGeneratorSeed(0)
        randomness.SetRandomizedNormalRNGSeed(0)

    # status must be an initialized dict for the call to C++.
    if status is None:
        status = {}

    best_points_to_sample = C_GP.multistart_expected_improvement_optimization(
        ei_optimizer.optimizer_parameters,
        ei_optimizer.objective_function._gaussian_process._gaussian_process,
        cpp_utils.cppify(ei_optimizer.domain.domain_bounds),
        cpp_utils.cppify(ei_optimizer.objective_function._points_being_sampled),
        num_to_sample,
        ei_optimizer.objective_function.num_being_sampled,
        ei_optimizer.objective_function._best_so_far,
        ei_optimizer.objective_function._num_mc_iterations,
        max_num_threads,
        use_gpu,
        which_gpu,
        randomness,
        status,
    )

    # reform output to be a list of dim-dimensional points, dim = len(self.domain)
    return cpp_utils.uncppify(best_points_to_sample, (num_to_sample, ei_optimizer.objective_function.dim))


def _heuristic_expected_improvement_optimization(
        ei_optimizer,
        num_multistarts,
        num_to_sample,
        estimation_policy,
        randomness=None,
        max_num_threads=DEFAULT_MAX_NUM_THREADS,
        status=None,
):
    r"""Heuristically solve the q,0-EI problem (estimating multistart_expected_improvement_optimization()) using 1,0-EI solves.

    Consider this as an alternative when multistart_expected_improvement_optimization() is too expensive. Since this function
    kernalizes 1,0-EI, it always hits the analytic case; hence it is much faster than q,0-EI which requires monte-carlo.
    Users will probably call one of this function's wrappers (e.g., constant_liar_expected_improvement_optimization() or
    kriging_believer_expected_improvement_optimization()) instead of accessing this directly.

    Calls into heuristic_expected_improvement_optimization_wrapper in cpp/GPP_python_expected_improvement.cpp.

    .. NOTE:: The following comments are copied from gpp_heuristic_expected_improvement_optimization.hpp, ComputeHeuristicPointsToSample().

    It heuristically solves the q,0-EI optimization problem. As a reminder, that problem is finding the set of q points
    that maximizes the Expected Improvement (saved in the output, ``best_points_to_sample``). Solving for q points simultaneously
    usually requires monte-carlo iteration and is expensive. The heuristic here solves q-EI as a sequence of 1-EI problems.
    We solve 1-EI, and then we *ASSUME* an objective function value at the resulting optima. This process is repeated q times.
    It is perhaps more clear in pseudocode::

      points_being_sampled = {}  // This stays empty! We are only working with 1,0-EI solves
      for i = 0:num_to_sample-1 {
        // First, solve the 1,0-EI problem\*
        new_point = ComputeOptimalPointsToSampleWithRandomStarts(gaussian_process, points_being_sampled, other_parameters)
        // *Estimate* the objective function value at new_point
        new_function_value = ESTIMATED_OBJECTIVE_FUNCTION_VALUE(new_point, other_args)
        new_function_value_noise = ESTIMATED_NOISE_VARIANCE(new_point, other_args)
        // Write the estimated objective values to the GP as *truth*
        gaussian_process.AddPoint(new_point, new_function_value, new_function_value_noise)
        optimal_points_to_sample.append(new_point)
      }

    \*Recall: each call to ComputeOptimalPointsToSampleWithRandomStarts() (gpp_math.hpp) kicks off a round of MGD optimization of 1-EI.

    Note that ideally the estimated objective function value (and noise) would be measured from the real-world (e.g.,
    by running an experiment). Then this algorithm would be optimal. However, the estimate probably is not accurately
    representating of the true objective.

    The estimation is handled through the "estimation_policy" input. Passing a ConstantLiarEstimationPolicy or
    KrigingBelieverEstimationPolicy object to this function will produce the "Constant Liar" and "Kriging Believer"
    heuristics described in Ginsbourger 2008. The interface for estimation_policy is generic so users may specify
    other estimators as well.

    Contrast this approach with ComputeOptimalPointsToSample() (gpp_math.hpp) which solves all outputs of the q,0-EI
    problem simultaneously instead of one point at a time. That method is more accurate (b/c it
    does not attempt to estimate the behavior of the underlying objective function) but much more expensive (because it
    requires monte-carlo iteration).

    If ``num_to_sample = 1``, this is exactly the same as ComputeOptimalPointsToSampleWithRandomStarts(); i.e.,
    both methods solve the 1-EI optimization problem the same way.

    Currently, during optimization, we recommend that the coordinates of the initial guesses not differ from the
    coordinates of the optima by more than about 1 order of magnitude. This is a very (VERY!) rough guideline for
    sizing the domain and num_multistarts; i.e., be wary of sets of initial guesses that cover the space too sparsely.

    Solution is guaranteed to lie within the region specified by "domain"; note that this may not be a
    local optima (i.e., the gradient may be substantially nonzero).

    .. WARNING:: this function fails if any step fails to find improvement! In that case, the return should not be
           read and status will report false.

    :param ei_optimizer: object that optimizes (e.g., gradient descent, newton) EI over a domain
    :type ei_optimizer: cpp_wrappers.optimization.*Optimizer object
    :param num_multistarts: number of times to multistart ``ei_optimizer`` (UNUSED, data is in ei_optimizer.optimizer_parameters)
    :type num_multistarts: int > 0
    :param num_to_sample: how many simultaneous experiments you would like to run (i.e., the q in q,0-EI)
    :type num_to_sample: int >= 1
    :param estimation_policy: the policy to use to produce (heuristic) objective function estimates during q,0-EI optimization
    :type estimation_policy: subclass of ObjectiveEstimationPolicyInterface (C++ pure abstract class)
       e.g., C_GP.KrigingBelieverEstimationPolicy, C_GP.ConstantLiarEstimationPolicy
       See gpp_heuristic_expected_improvement_optimization.hpp
    :param randomness: RNGs used by C++ to generate initial guesses
    :type randomness: RandomnessSourceContainer (C++ object; e.g., from C_GP.RandomnessSourceContainer())
    :param max_num_threads: maximum number of threads to use, >= 1
    :type max_num_threads: int > 0
    :param status: (output) status messages from C++ (e.g., reporting on optimizer success, etc.)
    :type status: dict
    :return: point(s) that approximately maximize the expected improvement (solving the q,0-EI problem)
    :rtype: array of float64 with shape (num_to_sample, ei_optimizer.objective_function.dim)

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

    best_points_to_sample = C_GP.heuristic_expected_improvement_optimization(
        ei_optimizer.optimizer_parameters,
        ei_optimizer.objective_function._gaussian_process._gaussian_process,
        cpp_utils.cppify(ei_optimizer.domain._domain_bounds),
        estimation_policy,
        num_to_sample,
        ei_optimizer.objective_function._best_so_far,
        max_num_threads,
        randomness,
        status,
    )

    # reform output to be a list of dim-dimensional points, dim = len(self.domain)
    return cpp_utils.uncppify(best_points_to_sample, (num_to_sample, ei_optimizer.objective_function.dim))


def constant_liar_expected_improvement_optimization(
        ei_optimizer,
        num_multistarts,
        num_to_sample,
        lie_value,
        lie_noise_variance=0.0,
        randomness=None,
        max_num_threads=DEFAULT_MAX_NUM_THREADS,
        status=None,
):
    """Heuristically solves q,0-EI using the Constant Liar policy; this wraps heuristic_expected_improvement_optimization().

    Note that this optimizer only uses the analytic 1,0-EI, so it is fast.

    See heuristic_expected_improvement_optimization() docs for general notes on how the heuristic optimization works.
    In this specific instance, we use the Constant Liar estimation policy.

    .. Note:: comments copied from ConstantLiarEstimationPolicy in gpp_heuristic_expected_improvement_optimization.hpp.

    The "Constant Liar" objective function estimation policy is the simplest: it always returns the same value
    (Ginsbourger 2008). We call this the "lie. This object also allows users to associate a noise variance to
    the lie value.

    In Ginsbourger's work, the most common lie values have been the min and max of all previously observed objective
    function values; i.e., min, max of GP.points_sampled_value. The mean has also been considered.

    He also points out that larger lie values (e.g., max of prior measurements) will lead methods like
    ComputeEstimatedSetOfPointsToSample() to be more explorative and vice versa.

    :param ei_optimizer: object that optimizes (e.g., gradient descent, newton) EI over a domain
    :type ei_optimizer: cpp_wrappers.optimization.*Optimizer object
    :param num_multistarts: number of times to multistart ``ei_optimizer`` (UNUSED, data is in ei_optimizer.optimizer_parameters)
    :type num_multistarts: int > 0
    :param num_to_sample: how many simultaneous experiments you would like to run (i.e., the q in q,0-EI)
    :type num_to_sample: int >= 1
    :param lie_value: the "constant lie" that this estimator should return
    :type lie_value: float64
    :param lie_noise_variance: the noise_variance to associate to the lie_value (MUST be >= 0.0)
    :type lie_noise_variance: float64
    :param randomness: RNGs used by C++ to generate initial guesses
    :type randomness: RandomnessSourceContainer (C++ object; e.g., from C_GP.RandomnessSourceContainer())
    :param max_num_threads: maximum number of threads to use, >= 1
    :type max_num_threads: int > 0
    :param status: (output) status messages from C++ (e.g., reporting on optimizer success, etc.)
    :type status: dict
    :return: point(s) that approximately maximize the expected improvement (solving the q,0-EI problem)
    :rtype: array of float64 with shape (num_to_sample, ei_optimizer.objective_function.dim)

    """
    estimation_policy = C_GP.ConstantLiarEstimationPolicy(lie_value, lie_noise_variance)
    return _heuristic_expected_improvement_optimization(
        ei_optimizer,
        num_multistarts,
        num_to_sample,
        estimation_policy,
        randomness=randomness,
        max_num_threads=max_num_threads,
        status=status,
    )


def kriging_believer_expected_improvement_optimization(
        ei_optimizer,
        num_multistarts,
        num_to_sample,
        std_deviation_coef=0.0,
        kriging_noise_variance=0.0,
        randomness=None,
        max_num_threads=DEFAULT_MAX_NUM_THREADS,
        status=None,
):
    """Heuristically solves q,0-EI using the Kriging Believer policy; this wraps heuristic_expected_improvement_optimization().

    Note that this optimizer only uses the analytic 1,0-EI, so it is fast.

    See heuristic_expected_improvement_optimization() docs for general notes on how the heuristic optimization works.
    In this specific instance, we use the Kriging Believer estimation policy.

    .. Note:: comments copied from KrigingBelieverEstimationPolicy in gpp_heuristic_expected_improvement_optimization.hpp.

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

    :param ei_optimizer: object that optimizes (e.g., gradient descent, newton) EI over a domain
    :type ei_optimizer: cpp_wrappers.optimization.*Optimizer object
    :param num_multistarts: number of times to multistart ``ei_optimizer`` (UNUSED, data is in ei_optimizer.optimizer_parameters)
    :type num_multistarts: int > 0
    :param num_to_sample: how many simultaneous experiments you would like to run (i.e., the q in q,0-EI)
    :type num_to_sample: int >= 1
    :param std_deviation_coef: the relative amount of bias (in units of GP std deviation) to introduce into the GP mean
    :type std_deviation_coef: float64
    :param kriging_noise_variance: the noise_variance to associate to each function value estimate (MUST be >= 0.0)
    :type kriging_noise_variance: float64
    :param randomness: RNGs used by C++ to generate initial guesses
    :type randomness: RandomnessSourceContainer (C++ object; e.g., from C_GP.RandomnessSourceContainer())
    :param max_num_threads: maximum number of threads to use, >= 1
    :type max_num_threads: int > 0
    :param status: (output) status messages from C++ (e.g., reporting on optimizer success, etc.)
    :type status: dict
    :return: point(s) that approximately maximize the expected improvement (solving the q,0-EI problem)
    :rtype: array of float64 with shape (num_to_sample, ei_optimizer.objective_function.dim)

    """
    estimation_policy = C_GP.KrigingBelieverEstimationPolicy(std_deviation_coef, kriging_noise_variance)
    return _heuristic_expected_improvement_optimization(
        ei_optimizer,
        num_multistarts,
        num_to_sample,
        estimation_policy,
        randomness=randomness,
        max_num_threads=max_num_threads,
        status=status,
    )


class ExpectedImprovement(ExpectedImprovementInterface, OptimizableInterface):

    r"""Implementation of Expected Improvement computation via C++ wrappers: EI and its gradient at specified point(s) sampled from a GaussianProcess.

    A class to encapsulate the computation of expected improvement and its spatial gradient using points sampled from an
    associated GaussianProcess. The general EI computation requires monte-carlo integration; it can support q,p-EI optimization.
    It is designed to work with any GaussianProcess.

    .. Note:: Equivalent methods of ExpectedImprovementInterface and OptimizableInterface are aliased below (e.g.,
      compute_expected_improvement and compute_objective_function, etc).

    See :mod:`moe.optimal_learning.python.interfaces.expected_improvement_interface` docs for further details.

    """

    def __init__(
            self,
            gaussian_process,
            points_to_sample=None,
            points_being_sampled=None,
            num_mc_iterations=DEFAULT_EXPECTED_IMPROVEMENT_MC_ITERATIONS,
            randomness=None
    ):
        """Construct an ExpectedImprovement object that knows how to call C++ for evaluation of member functions.

        :param gaussian_process: GaussianProcess describing
        :type gaussian_process: :class:`moe.optimal_learning.python.cpp_wrappers.gaussian_process.GaussianProcess` object
        :param points_to_sample: points at which to evaluate EI and/or its gradient to check their value in future experiments (i.e., "q" in q,p-EI)
        :type points_to_sample: array of float64 with shape (num_to_sample, dim)
        :param points_being_sampled: points being sampled in concurrent experiments (i.e., "p" in q,p-EI)
        :type points_being_sampled: array of float64 with shape (num_being_sampled, dim)
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

        if points_being_sampled is None:
            self._points_being_sampled = numpy.array([])
        else:
            self._points_being_sampled = numpy.copy(points_being_sampled)

        if points_to_sample is None:
            # set an arbitrary point
            self.current_point = numpy.zeros((1, gaussian_process.dim))
        else:
            self.current_point = points_to_sample

        if randomness is None:
            self._randomness = C_GP.RandomnessSourceContainer(1)  # create randomness for only 1 thread
            # Set seed based on less repeatable factors (e.g,. time)
            self._randomness.SetRandomizedUniformGeneratorSeed(0)
            self._randomness.SetRandomizedNormalRNGSeed(0)
        else:
            self._randomness = randomness

        self.objective_type = None  # Not used for EI, but the field is expected in C++

    @property
    def dim(self):
        """Return the number of spatial dimensions."""
        return self._gaussian_process.dim

    @property
    def num_to_sample(self):
        """Number of points at which to compute/optimize EI, aka potential points to sample in future experiments; i.e., the ``q`` in ``q,p-EI``."""
        return self._points_to_sample.shape[0]

    @property
    def num_being_sampled(self):
        """Number of points being sampled in concurrent experiments; i.e., the ``p`` in ``q,p-EI``."""
        return self._points_being_sampled.shape[0]

    @property
    def problem_size(self):
        """Return the number of independent parameters to optimize."""
        return self.num_to_sample * self.dim

    def get_current_point(self):
        """Get the current_point (array of float64 with shape (problem_size)) at which this object is evaluating the objective function, ``f(x)``."""
        return numpy.copy(self._points_to_sample)

    def set_current_point(self, points_to_sample):
        """Set current_point to the specified point; ordering must match.

        :param points_to_sample: current_point at which to evaluate the objective function, ``f(x)``
        :type points_to_sample: array of float64 with shape (problem_size)

        """
        self._points_to_sample = numpy.copy(numpy.atleast_2d(points_to_sample))

    current_point = property(get_current_point, set_current_point)

    def evaluate_at_point_list(
            self,
            points_to_evaluate,
            randomness=None,
            max_num_threads=DEFAULT_MAX_NUM_THREADS,
            status=None,
    ):
        """Evaluate Expected Improvement (1,p-EI) over a specified list of ``points_to_evaluate``.

        .. Note:: We use ``points_to_evaluate`` instead of ``self._points_to_sample`` and compute the EI at those points only.
            ``self._points_to_sample`` is unchanged.

        Generally gradient descent is preferred but when they fail to converge this may be the only "robust" option.
        This function is also useful for plotting or debugging purposes (just to get a bunch of EI values).

        :param points_to_evaluate: points at which to compute EI
        :type points_to_evaluate: array of float64 with shape (num_to_evaluate, self.dim)
        :param randomness: RNGs used by C++ to generate initial guesses and as the source of normal random numbers when monte-carlo is used
        :type randomness: RandomnessSourceContainer (C++ object; e.g., from C_GP.RandomnessSourceContainer())
        :param max_num_threads: maximum number of threads to use, >= 1
        :type max_num_threads: int > 0
        :param status: (output) status messages from C++ (e.g., reporting on optimizer success, etc.)
        :type status: dict
        :return: EI evaluated at each of points_to_evaluate
        :rtype: array of float64 with shape (points_to_evaluate.shape[0])

        """
        # Create enough randomness sources if none are specified.
        if randomness is None:
            if max_num_threads == 1:
                randomness = self._randomness
            else:
                randomness = C_GP.RandomnessSourceContainer(max_num_threads)
                # Set seeds based on less repeatable factors (e.g,. time)
                randomness.SetRandomizedUniformGeneratorSeed(0)
                randomness.SetRandomizedNormalRNGSeed(0)

        # status must be an initialized dict for the call to C++.
        if status is None:
            status = {}

        # num_to_sample need not match ei_evaluator.num_to_sample since points_to_evaluate
        # overrides any data inside ei_evaluator
        num_to_evaluate, num_to_sample, _ = points_to_evaluate.shape

        ei_values = C_GP.evaluate_EI_at_point_list(
            self._gaussian_process._gaussian_process,
            cpp_utils.cppify(points_to_evaluate),
            cpp_utils.cppify(self._points_being_sampled),
            num_to_evaluate,
            num_to_sample,
            self.num_being_sampled,
            self._best_so_far,
            self._num_mc_iterations,
            max_num_threads,
            randomness,
            status,
        )
        return numpy.array(ei_values)

    def compute_expected_improvement(self, force_monte_carlo=False):
        r"""Compute the expected improvement at ``points_to_sample``, with ``points_being_sampled`` concurrent points being sampled.

        .. Note:: These comments were copied from
          :meth:`moe.optimal_learning.python.interfaces.expected_improvement_interface.ExpectedImprovementInterface.compute_expected_improvement`

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

        :param force_monte_carlo: whether to force monte carlo evaluation (vs using fast/accurate analytic eval when possible)
        :type force_monte_carlo: boolean
        :return: the expected improvement from sampling ``points_to_sample`` with ``points_being_sampled`` concurrent experiments
        :rtype: float64

        """
        return C_GP.compute_expected_improvement(
            self._gaussian_process._gaussian_process,
            cpp_utils.cppify(self._points_to_sample),
            cpp_utils.cppify(self._points_being_sampled),
            self.num_to_sample,
            self.num_being_sampled,
            self._num_mc_iterations,
            self._best_so_far,
            force_monte_carlo,
            self._randomness,
        )

    compute_objective_function = compute_expected_improvement

    def compute_grad_expected_improvement(self, force_monte_carlo=False):
        r"""Compute the gradient of expected improvement at ``points_to_sample`` wrt ``points_to_sample``, with ``points_being_sampled`` concurrent samples.

        .. Note:: These comments were copied from
          :meth:`moe.optimal_learning.python.interfaces.expected_improvement_interface.ExpectedImprovementInterface.compute_grad_expected_improvement`

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

        :param force_monte_carlo: whether to force monte carlo evaluation (vs using fast/accurate analytic eval when possible)
        :type force_monte_carlo: boolean
        :return: gradient of EI, ``\pderiv{EI(Xq \cup Xp)}{Xq_{i,d}}`` where ``Xq`` is ``points_to_sample``
          and ``Xp`` is ``points_being_sampled`` (grad EI from sampling ``points_to_sample`` with
          ``points_being_sampled`` concurrent experiments wrt each dimension of the points in ``points_to_sample``)
        :rtype: array of float64 with shape (num_to_sample, dim)

        """
        grad_ei = C_GP.compute_grad_expected_improvement(
            self._gaussian_process._gaussian_process,
            cpp_utils.cppify(self._points_to_sample),
            cpp_utils.cppify(self._points_being_sampled),
            self.num_to_sample,
            self.num_being_sampled,
            self._num_mc_iterations,
            self._best_so_far,
            force_monte_carlo,
            self._randomness,
        )
        return cpp_utils.uncppify(grad_ei, (self.num_to_sample, self.dim))

    compute_grad_objective_function = compute_grad_expected_improvement

    def compute_hessian_objective_function(self, **kwargs):
        """We do not currently support computation of the (spatial) hessian of Expected Improvement."""
        raise NotImplementedError('Currently we cannot compute the hessian of expected improvement.')
