# -*- coding: utf-8 -*-
"""Classes (Python) to compute the Expected Improvement, including monte carlo and analytic (where applicable) implementations.

See interfaces/expected_improvement_interface.py or gpp_math.hpp/cpp for further details on expected improvement.

"""
import numpy
import scipy.linalg
import scipy.stats

from moe.optimal_learning.python.constant import DEFAULT_EXPECTED_IMPROVEMENT_MC_ITERATIONS, DEFAULT_EXPECTED_IMPROVEMENT_MAX_NUM_THREADS
from moe.optimal_learning.python.interfaces.expected_improvement_interface import ExpectedImprovementInterface
from moe.optimal_learning.python.interfaces.optimization_interface import OptimizableInterface
from moe.optimal_learning.python.python_version.optimization import multistart_optimize, NullOptimizer


def multistart_expected_improvement_optimization(
        ei_optimizer,
        num_multistarts,
        num_to_sample,
        randomness=None,
        max_num_threads=DEFAULT_EXPECTED_IMPROVEMENT_MAX_NUM_THREADS,
        status=None,
):
    """Solve the q,p-EI problem, returning the optimal set of q points to sample CONCURRENTLY in future experiments.

    When ``points_being_sampled.shape[0] == 0 && num_to_sample == 1``, this function will use (fast) analytic EI computations.

    .. NOTE:: The following comments are copied from multistart_expected_improvement_optimization() in cpp_wrappers/expected_improvement.py

    This is the primary entry-point for EI optimization in the optimal_learning library. It offers our best shot at
    improving robustness by combining higher accuracy methods like gradient descent with fail-safes like random/grid search.

    Returns the optimal set of q points to sample CONCURRENTLY by solving the q,p-EI problem.  That is, we may want to run 4
    experiments at the same time and maximize the EI across all 4 experiments at once while knowing of 2 ongoing experiments
    (4,2-EI). This function handles this use case. Evaluation of q,p-EI (and its gradient) for q > 1 or p > 1 is expensive
    (requires monte-carlo iteration), so this method is usually very expensive.

    Compared to ComputeHeuristicPointsToSample() (``gpp_heuristic_expected_improvement_optimization.hpp``), this function
    makes no external assumptions about the underlying objective function. Instead, it utilizes the Expected (Parallel)
    Improvement, allowing the GP to account for ongoing/incomplete experiments.

    If ``num_to_sample = 1``, this is the same as ComputeOptimalPointsToSampleWithRandomStarts().

    TODO(eliu): (GH-56) Allow callers to pass in a source of randomness.

    :param ei_optimizer: object that optimizes (e.g., gradient descent, newton) EI over a domain
    :type ei_optimizer: interfaces.optimization_interfaces.OptimizerInterface subclass
    :param num_multistarts: number of times to multistart ``ei_optimizer``
    :type num_multistarts: int > 0
    :param num_to_sample: how many simultaneous experiments you would like to run (i.e., the q in q,p-EI) (UNUSED, specify through ei_optimizer)
    :type num_to_sample: int >= 1
    :param randomness: random source(s) used to generate multistart points and perform monte-carlo integration (when applicable) (UNUSED)
    :type randomness: (UNUSED)
    :param max_num_threads: maximum number of threads to use, >= 1 (UNUSED)
    :type max_num_threads: int > 0
    :param status: status messages from C++ (e.g., reporting on optimizer success, etc.)
    :type status: dict
    :return: point(s) that maximize the expected improvement (solving the q,p-EI problem)
    :rtype: array of float64 with shape (num_to_sample, ei_evaluator.dim)

    """
    random_starts = ei_optimizer.domain.generate_uniform_random_points_in_domain(num_points=num_multistarts)
    best_point, _ = multistart_optimize(ei_optimizer, starting_points=random_starts)

    # TODO(eliu): (GH-59) Have GD actually indicate whether updates were found.
    found_flag = True
    if status is not None:
        status["gradient_descent_found_update"] = found_flag

    return best_point


class ExpectedImprovement(ExpectedImprovementInterface, OptimizableInterface):

    r"""Implementation of Expected Improvement computation in Python: EI and its gradient at specified point(s) sampled from a GaussianProcess.

    A class to encapsulate the computation of expected improvement and its spatial gradient using points sampled from an
    associated GaussianProcess. The general EI computation requires monte-carlo integration; it can support q,p-EI optimization.
    It is designed to work with any GaussianProcess.

    When available, fast, analytic formulas replace monte-carlo loops.

    See interfaces/expected_improvement_interface.py docs for further details.

    """

    def __init__(
            self,
            gaussian_process,
            points_to_sample=None,
            points_being_sampled=None,
            num_mc_iterations=DEFAULT_EXPECTED_IMPROVEMENT_MC_ITERATIONS,
            randomness=None,
    ):
        """Construct an ExpectedImprovement object that supports q,p-EI.

        TODO(eliu): (GH-56) Allow callers to pass in a source of randomness.

        :param gaussian_process: GaussianProcess describing
        :type gaussian_process: interfaces.gaussian_process_interface.GaussianProcessInterface subclass
        :param points_to_sample: points at which to evaluate EI and/or its gradient to check their value in future experiments (i.e., "q" in q,p-EI)
        :type points_to_sample: array of float64 with shape (num_to_sample, dim)
        :param points_being_sampled: points being sampled in concurrent experiments (i.e., "p" in q,p-EI)
        :type points_being_sampled: array of float64 with shape (num_being_sampled, dim)
        :param num_mc_iterations: number of monte-carlo iterations to use (when monte-carlo integration is used to compute EI)
        :type num_mc_iterations: int > 0
        :param randomness: random source(s) used for monte-carlo integration (when applicable) (UNUSED)
        :type randomness: (UNUSED)

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
            self.set_current_point(numpy.zeros((1, gaussian_process.dim)))
        else:
            self.set_current_point(points_to_sample)

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

    def evaluate_at_point_list(
            self,
            points_to_evaluate,
            randomness=None,
            max_num_threads=DEFAULT_EXPECTED_IMPROVEMENT_MAX_NUM_THREADS,
            status=None,
    ):
        """Evaluate Expected Improvement (q,p-EI) over a specified list of ``points_to_evaluate``.

        .. Note:: We use ``points_to_evaluate`` instead of ``self._points_to_sample`` and compute the EI at those points only.
            ``self._points_to_sample`` will be changed.

        Generally gradient descent is preferred but when it fails to converge this may be the only "robust" option.
        This function is also useful for plotting or debugging purposes (just to get a bunch of EI values).

        TODO(eliu): (GH-56) Allow callers to pass in a source of randomness.

        :param ei_evaluator: object specifying how to evaluate the expected improvement
        :type ei_evaluator: interfaces.expected_improvement_interface.ExpectedImprovementInterface subclass
        :param points_to_evaluate: points at which to compute EI
        :type points_to_evaluate: array of float64 with shape (num_to_evaluate, num_to_sample, ei_evaluator.dim)
        :param randomness: random source(s) used for monte-carlo integration (when applicable) (UNUSED)
        :type randomness: (UNUSED)
        :param max_num_threads: maximum number of threads to use, >= 1 (UNUSED)
        :type max_num_threads: int > 0
        :param status: status messages from C++ (e.g., reporting on optimizer success, etc.)
        :type status: dict
        :return: EI evaluated at each of points_to_evaluate
        :rtype: array of float64 with shape (points_to_evaluate.shape[0])

        """
        null_optimizer = NullOptimizer(None, self)
        _, values = multistart_optimize(null_optimizer, starting_points=points_to_evaluate)

        # TODO(eliu): (GH-59) Have multistart actually indicate whether updates were found.
        found_flag = True
        if status is not None:
            status["evaluate_EI_at_point_list"] = found_flag

        return values

    def _compute_expected_improvement_1d_analytic(self, mu_star, var_star):
        """Compute EI when the number of potential samples is 1 (i.e., points_being_sampled.size = 0) using *fast* analytic methods.

        This function can only support the computation of 1,0-EI. In this case, we have analytic formulas
        for computing EI (and its gradient).

        Thus this function does not perform any explicit numerical integration, nor does it require access to a
        random number generator.

        See Ginsbourger, Le Riche, and Carraro.

        :param mu_star: the mean of the GP evaluated at points_to_sample
        :type mu_star: float64
        :param var_star: the variance of the GP evaluated at points_to_sample
        :type var_star: float64
        :return: the expected improvement from sampling ``point_to_sample``
        :rtype: float64

        """
        sigma_star = numpy.sqrt(var_star)
        temp = self._best_so_far - mu_star
        expected_improvement = temp * scipy.stats.norm.cdf(temp / sigma_star) + sigma_star * scipy.stats.norm.pdf(temp / sigma_star)
        return numpy.fmax(0.0, expected_improvement)

    def _compute_grad_expected_improvement_1d_analytic(self, mu_star, var_star, grad_mu, grad_chol_decomp):
        r"""Compute the gradient of EI when the number of potential samples is 1 (i.e., points_being_sampled.size = 0) using *fast* analytic methods.

        This function can only support the computation of 1,0-EI. In this case, we have analytic formulas
        for computing EI and its gradient.

        Thus this function does not perform any explicit numerical integration, nor does it require access to a
        random number generator.

        See Ginsbourger, Le Riche, and Carraro.

        :param mu_star: the mean of the GP evaluated at points_to_sample
        :type mu_star: float64
        :param var_star: the variance of the GP evaluated at points_to_sample
        :type var_star: float64
        :param grad_mu: the gradient of the mean of the GP evaluated at points_to_sample, wrt points_to_sample
        :type grad_mu: array of float64 with shape (self.dim)
        :param grad_chol_decomp: the gradient of the variance of the GP evaluated at points_to_sample, wrt points_to_sample
        :type grad_chol_decomp: array of float64 with shape (self.dim)
        :return: gradient of EI, ``\pderiv{EI(x)}{x_d}``, where ``x`` is ``points_to_sample``
        :rtype: array of float64 with shape (self.dim)

        """
        sigma_star = numpy.sqrt(var_star)
        mu_diff = self._best_so_far - mu_star
        c = mu_diff / sigma_star
        pdf_c = scipy.stats.norm.pdf(c)
        cdf_c = scipy.stats.norm.cdf(c)

        d_c = (-sigma_star * grad_mu - grad_chol_decomp * mu_diff) / var_star
        d_a = -grad_mu * cdf_c + mu_diff * pdf_c * d_c
        d_b = grad_chol_decomp * pdf_c - sigma_star * c * pdf_c * d_c

        d_a += d_b
        return numpy.atleast_2d(d_a)

    def _compute_expected_improvement_monte_carlo_naive(self, mu_star, var_star):
        """Compute EI using (naive) monte carlo integration.

        See _compute_expected_improvement_monte_carlo (below) for more details on EI.

        This function produces the *exact* same output as the vectorized version. But
        since the loop over num_mc_iterations runs in Python, it is > 100x slower.
        However, this code is easy to verify b/c it follows the algorithmic description
        faithfully. We use it only for verification.

        :param mu_star: self._gaussian_process.compute_mean_of_points(union_of_points)
        :type mu_star: array of float64 with shape (num_points)
        :param var_star: self._gaussian_process.compute_variance_of_points(union_of_points)
        :type var_star: array of float64 with shape (num_points, num_points)
        :return: the expected improvement from sampling ``points_to_sample`` with ``points_being_sampled`` concurrent experiments
        :rtype: float64

        """
        num_points = self.num_to_sample + self.num_being_sampled
        chol_var = scipy.linalg.cholesky(var_star, lower=True)

        normals = numpy.random.normal(size=(self._num_mc_iterations, num_points))

        aggregate = 0.0
        for normal_draws in normals:
            improvement_this_iter = numpy.amax(self._best_so_far - mu_star - numpy.dot(chol_var, normal_draws.T))
            if improvement_this_iter > 0.0:
                aggregate += improvement_this_iter
        return aggregate / float(self._num_mc_iterations)

    def _compute_expected_improvement_monte_carlo(self, mu_star, var_star):
        r"""Compute EI using (vectorized) monte-carlo integration; this is a general method that works for any input.

        This function cal support the computation of q,p-EI.
        This function requires access to a random number generator.

        .. Note:: comments here are copied from gpp_math.cpp, ExpectedImprovementEvaluator::ComputeExpectedImprovement().

        Let ``Ls * Ls^T = Vars`` and ``w`` = vector of IID normal(0,1) variables
        Then:
        ``y = mus + Ls * w``  (Equation 4, from file docs)
        simulates drawing from our GP with mean mus and variance Vars.

        Then as given in the file docs, we compute the improvement:
        Then the improvement for this single sample is:
        ``I = { best_known - min(y)   if (best_known - min(y) > 0)      (Equation 5 from file docs)``
        ``    {          0               else``
        This is implemented as ``max_{y} (best_known - y)``.  Notice that improvement takes the value 0 if it would be negative.

        Since we cannot compute ``min(y)`` directly, we do so via monte-carlo (MC) integration.  That is, we draw from the GP
        repeatedly, computing improvement during each iteration, and averaging the result.

        See Scott's PhD thesis, sec 6.2.

        For performance, this function vectorizes the monte-carlo integration loop, using numpy's mask feature to skip
        iterations where the improvement is not positive.

        :param mu_star: self._gaussian_process.compute_mean_of_points(union_of_points)
        :type mu_star: array of float64 with shape (num_points)
        :param var_star: self._gaussian_process.compute_variance_of_points(union_of_points)
        :type var_star: array of float64 with shape (num_points, num_points)
        :return: the expected improvement from sampling ``points_to_sample`` with ``points_being_sampled`` concurrent experiments
        :rtype: float64

        """
        num_points = self.num_to_sample + self.num_being_sampled
        chol_var = -scipy.linalg.cholesky(var_star, lower=True)

        normals = numpy.random.normal(size=(self._num_mc_iterations, num_points))

        # TODO(eliu): (GH-60) Partition num_mc_iterations up into smaller blocks if it helps.
        # so that we don't waste as much mem bandwidth (since each entry of normals is
        # only used once)
        mu_star = self._best_so_far - mu_star
        # Compute Ls * w; note the shape is (self._num_mc_iterations, num_points)
        improvement_each_iter = numpy.einsum('kj, ij', chol_var, normals)
        # Now we have improvement = best_so_far - y = best_so_far - (mus + Ls * w)
        improvement_each_iter += mu_star
        # We want the maximum improvement each step; note the shape is (self._num_mc_iterations)
        best_improvement_each_iter = numpy.amax(improvement_each_iter, axis=1)
        # Only keep the *positive* improvements
        best_improvement_each_iter = numpy.ma.masked_less_equal(best_improvement_each_iter, 0.0, copy=False)
        result = best_improvement_each_iter.sum(dtype=numpy.float64) / float(self._num_mc_iterations)
        # If all iterations yielded non-positive improvement, sum returns numpy.ma.masked (instead of 0.0)
        if result is numpy.ma.masked:
            return 0.0
        else:
            return result

    def _compute_grad_expected_improvement_monte_carlo_naive(self, mu_star, var_star, grad_mu, grad_chol_decomp):
        r"""Compute the gradient of EI using (naive) monte carlo integration.

        See _compute_grad_expected_improvement_monte_carlo (below) for more details on grad EI and how it is computed.

        This function produces the *exact* same output as the vectorized version. But
        since the loop over num_mc_iterations runs in Python, it is > 100x slower.
        However, this code is easy to verify b/c it follows the algorithmic description
        faithfully. We use it only for verification.

        :param mu_star: self._gaussian_process.compute_mean_of_points(union_of_points)
        :type mu_star: array of float64 with shape (num_points)
        :param var_star: self._gaussian_process.compute_variance_of_points(union_of_points)
        :type var_star: array of float64 with shape (num_points, num_points)
        :param grad_mu: self._gaussian_process.compute_grad_mean_of_points(union_of_points)
        :type grad_mu: array of float64 with shape (num_points, self.dim)
        :param grad_chol_decomp: self._gaussian_process.compute_grad_cholesky_variance_of_points(union_of_points)
        :type grad_chol_decomp: array of float64 with shape (self.num_to_sample, num_points, num_points, self.dim)
        :return: gradient of EI, ``\pderiv{EI(Xq \cup Xp)}{Xq_{i,d}}`` where ``Xq`` is ``points_to_sample``
          and ``Xp`` is ``points_being_sampled`` (grad EI from sampling ``points_to_sample`` with
          ``points_being_sampled`` concurrent experiments wrt each dimension of the points in ``points_to_sample``)
        :rtype: array of float64 with shape (self.num_to_sample, self.dim)

        """
        num_points = self.num_to_sample + self.num_being_sampled
        chol_var = scipy.linalg.cholesky(var_star, lower=True)

        normals = numpy.random.normal(size=(self._num_mc_iterations, num_points))

        # Differentiating wrt each point of self._points_to_sample
        aggregate_dx = numpy.zeros_like(self._points_to_sample)
        for normal_draws in normals:
            improvements_this_iter = self._best_so_far - mu_star - numpy.dot(chol_var, normal_draws.T)
            if numpy.amax(improvements_this_iter) > 0.0:
                winner = numpy.argmax(improvements_this_iter)
                if winner < self.num_to_sample:
                    aggregate_dx[winner, ...] -= grad_mu[winner, ...]
                for diff_index in xrange(self.num_to_sample):
                    # grad_chol_decomp_{diff_index, winner, i, j} * normal_draws_{i}
                    aggregate_dx[diff_index, ...] -= numpy.dot(grad_chol_decomp[diff_index, winner, ...].T, normal_draws)

        return aggregate_dx / float(self._num_mc_iterations)

    def _compute_grad_expected_improvement_monte_carlo(self, mu_star, var_star, grad_mu, grad_chol_decomp):
        r"""Compute the gradient of EI using (vectorized) monte-carlo integration; this is a general method that works for any input.

        This function cal support the computation of q,p-EI.
        This function requires access to a random number generator.

        .. Note:: comments here are copied from gpp_math.cpp, ExpectedImprovementEvaluator::ComputeGradExpectedImprovement().

        Mechanism is similar to the computation of EI, where points' contributions to the gradient are thrown out of their
        corresponding ``improvement <= 0.0``.

        Thus ``\nabla(\mu)`` only contributes when the ``winner`` (point w/best improvement this iteration) is the current point.
        That is, the gradient of ``\mu`` at ``x_i`` wrt ``x_j`` is 0 unless ``i == j`` (and only this result is stored in
        ``grad_mu``).  The interaction with ``grad_chol_decomp`` is harder to know a priori (like with
        ``grad_mu``) and has a more complex structure (rank 3 tensor), so the derivative wrt ``x_j`` is computed fully, and
        the relevant submatrix (indexed by the current ``winner``) is accessed each iteration.

        For performance, this function vectorizes the monte-carlo integration loop, using numpy's mask feature to skip
        iterations where the improvement is not positive. Some additional cleverness is required to vectorize the
        accesses into grad_chol_decomp, since we cannot afford to run a loop (even over ``normals_compressed``) in Python.

        :param mu_star: self._gaussian_process.compute_mean_of_points(union_of_points)
        :type mu_star: array of float64 with shape (num_points)
        :param var_star: self._gaussian_process.compute_variance_of_points(union_of_points)
        :type var_star: array of float64 with shape (num_points, num_points)
        :param grad_mu: self._gaussian_process.compute_grad_mean_of_points(union_of_points)
        :type grad_mu: array of float64 with shape (num_points, self.dim)
        :param grad_chol_decomp: self._gaussian_process.compute_grad_cholesky_variance_of_points(union_of_points)
        :type grad_chol_decomp: array of float64 with shape (self.num_to_sample, num_points, num_points, self.dim)
        :return: gradient of EI, ``\pderiv{EI(Xq \cup Xp)}{Xq_{i,d}}`` where ``Xq`` is ``points_to_sample``
          and ``Xp`` is ``points_being_sampled`` (grad EI from sampling ``points_to_sample`` with
          ``points_being_sampled`` concurrent experiments wrt each dimension of the points in ``points_to_sample``)
        :rtype: array of float64 with shape (self.num_to_sample, self.dim)

        """
        num_points = self.num_to_sample + self.num_being_sampled
        chol_var = -scipy.linalg.cholesky(var_star, lower=True)

        normals = numpy.random.normal(size=(self._num_mc_iterations, num_points))

        # TODO(eliu): (GH-60) Partition num_mc_iterations up into smaller blocks if it helps.
        # so that we don't waste as much mem bandwidth (since each entry of normals is
        # only used once)
        mu_star = self._best_so_far - mu_star
        # Compute Ls * w; note the shape is (self._num_mc_iterations, num_points)
        improvement_each_iter = numpy.einsum('kj, ij', chol_var, normals)
        # Now we have improvement = best_so_far - y = best_so_far - (mus + Ls * w)
        improvement_each_iter += mu_star
        # We want the maximum improvement each step; note the shape is (self._num_mc_iterations)
        best_improvement_each_iter = numpy.amax(improvement_each_iter, axis=1)
        # Index of the point that yielded the best improvement
        winner_indexes = numpy.argmax(improvement_each_iter, axis=1)
        # Only keep the *positive* improvements
        best_improvement_each_iter = numpy.ma.masked_less_equal(best_improvement_each_iter, 0.0, copy=False)

        # If all improvements are positive, then the mask is a scalar.
        # We want to expand the mask into a full vector so that subsequent code will work. Kind of hacky.
        if best_improvement_each_iter.mask is numpy.False_:
            best_improvement_each_iter.mask = [False]

        # Keep only the indexes with positive improvement
        winner_indexes = numpy.ma.array(winner_indexes, mask=best_improvement_each_iter.mask, copy=False)
        # Drop the masked terms
        winner_indexes_compressed = numpy.ma.compressed(winner_indexes)

        winner_indexes_tiled = numpy.tile(winner_indexes_compressed, (num_points, 1))
        # Indexes where the winner was point diff_index (the point we're differentiating against)
        # for each possible diff_index value
        winner_indexes_tiled_equal_to_diff_index = numpy.ma.masked_not_equal(winner_indexes_tiled.T, numpy.arange(num_points)).T

        # Differentiating wrt each point of self._points_to_sample
        # Handle derivative terms from grad_mu; only grab terms from winners 0:self.num_to_sample
        aggregate_dx = (-grad_mu[:self.num_to_sample, ...].T *
                        numpy.ma.count(winner_indexes_tiled_equal_to_diff_index[:self.num_to_sample, ...], axis=1)).T

        # Handle derivative terms from grad_chol_decomp
        # Mask rows of normals that did not show positive improvement
        # TODO(eliu): (GH-61) Use numpy.tile, numpy.repeat or something more sensical if possible.
        normals_mask = numpy.empty(normals.shape, dtype=bool)
        normals_mask[...] = best_improvement_each_iter.mask[:, numpy.newaxis]
        # Compress out the masked data
        normals_compressed = numpy.ma.array(normals, mask=normals_mask)
        # We'd like to use numpy.ma.compress_rows but somehow that is REALLY slow, like N^2 slow
        normals_compressed = normals_compressed[~normals_compressed.mask].reshape((winner_indexes_compressed.size, num_points))

        # We now want to compute: grad_chol_decomp[winner_index, i, j] * normals[k, i]
        # And sum over k for each winner_index.
        # To do this loop in numpy, we have to create grad_chol_decomp_tiled:
        # for k in xrange(self._num_mc_iterations):
        #   grad_chol_decomp_tiled[k, ...] = grad_chol_decomp[diff_index, winner_indexes[k], ...]
        # for each diff_index = 0:self.num_to_sample.
        # Except we make two optimizations:
        # 1) We skip all the masked terms (so we use the compressed arrays)
        # 2) We vectorize the tiling process.
        # Do not vectorize the loop over self.num_to_sample: the extra memory cost hurts performance. We store self.num_to_sample
        # times more copies but each copy is only used once. Not vectorizing produces better locality and self.num_to_sample
        # will never be very large.
        # This tradeoff may change when GH-60 is done.
        grad_chol_decomp_tiled = numpy.empty((normals_compressed.shape[0], grad_chol_decomp.shape[2], grad_chol_decomp.shape[3]))
        for diff_index in xrange(self.num_to_sample):
            grad_chol_decomp_tiled[...] = 0.0
            for i in xrange(num_points):
                # Only track the iterations where point i had the best improvement (winner)
                winner_indexes_equal_to_i = winner_indexes_tiled_equal_to_diff_index[i, ...]

                # If all winners were index i, then the mask is a scalar.
                # We want to expand the mask into a full vector so that subsequent code will work. Kind of hacky.
                if winner_indexes_equal_to_i.mask is numpy.False_:
                    # In fact we could stop here b/c this means index i won every time
                    winner_indexes_equal_to_i.mask = [False]

                # Expand winner_indexes_equal_to_i.mask to cover the full shape of grad_chol_decomp_tiled
                # This is the same idea as normals_mask above
                # TODO(eliu): (GH-61) Use numpy.tile, numpy.repeat or something more sensical if possible.
                grad_chol_decomp_block_i_tile_mask = numpy.empty(grad_chol_decomp_tiled.shape, dtype=bool)
                grad_chol_decomp_block_i_tile_mask[...] = winner_indexes_equal_to_i.mask[:, numpy.newaxis, numpy.newaxis]

                # TODO(eliu): (GH-61) Is there a way to produce the desired block pattern directly, without copy + mask? Can I avoid duplicating grad_chol entirely?
                # Tile the appropriate block of grad_chol_decomp to *FILL* all blocks
                grad_chol_decomp_block_i_tile = numpy.tile(
                    grad_chol_decomp[diff_index, i, ...],
                    (normals_compressed.shape[0], 1),
                ).reshape(grad_chol_decomp_tiled.shape)
                # Zero out blocks where the winner was not point i
                grad_chol_decomp_block_i_tile = numpy.ma.filled(
                    numpy.ma.array(
                        grad_chol_decomp_block_i_tile,
                        mask=grad_chol_decomp_block_i_tile_mask,
                    ),
                    fill_value=0.0,
                )
                # Add the tiles for this index into grad_chol_decomp_tiled
                # Note that since we zero all irrelevant blocks, we are never overwriting anything
                grad_chol_decomp_tiled += grad_chol_decomp_block_i_tile

            # Now we can compute the contribution from the variance in a fast C loop.
            aggregate_dx[diff_index, ...] -= numpy.einsum('ki, kij', normals_compressed, grad_chol_decomp_tiled)

        # For reference, the above block replaces the following code:
        # for it, normal in enumerate(normals_compressed):
        #     for diff_index in xrange(self.num_to_sample):
        #         aggregate_dx[diff_index, ...] -= numpy.dot(normal, grad_chol_decomp[diff_index, winner_indexes_compressed[it], ...])
        # The vectorized version performs exactly the same number of arithmetic operations in exactly the same order but
        # is at least 30x faster (difference grows with self._num_mc_iterations). Looping in Python is REALLY slow.

        aggregate_dx /= float(self._num_mc_iterations)
        return aggregate_dx

    def compute_expected_improvement(self, force_monte_carlo=False):
        r"""Compute the expected improvement at ``points_to_sample``, with ``points_being_sampled`` concurrent points being sampled.

        .. Note:: These comments were copied from this's superclass in expected_improvement_interface.py.

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
        num_points = self.num_to_sample + self.num_being_sampled
        union_of_points = numpy.reshape(numpy.append(self._points_to_sample, self._points_being_sampled), (num_points, self.dim))

        mu_star = self._gaussian_process.compute_mean_of_points(union_of_points)
        var_star = self._gaussian_process.compute_variance_of_points(union_of_points)

        if num_points == 1 and force_monte_carlo is False:
            return self._compute_expected_improvement_1d_analytic(mu_star[0], var_star[0, 0])
        else:
            return self._compute_expected_improvement_monte_carlo(mu_star, var_star)

    def compute_objective_function(self, **kwargs):
        """Wrapper for compute_expected_improvement; see that function's docstring."""
        return self.compute_expected_improvement(**kwargs)

    def compute_grad_expected_improvement(self, force_monte_carlo=False):
        r"""Compute the gradient of expected improvement at ``points_to_sample`` wrt ``points_to_sample``, with ``points_being_sampled`` concurrent samples.

        .. Note:: These comments were copied from this's superclass in expected_improvement_interface.py.

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
        num_points = self.num_to_sample + self.num_being_sampled
        union_of_points = numpy.reshape(numpy.append(self._points_to_sample, self._points_being_sampled), (num_points, self.dim))

        mu_star = self._gaussian_process.compute_mean_of_points(union_of_points)
        var_star = self._gaussian_process.compute_variance_of_points(union_of_points)
        grad_mu = self._gaussian_process.compute_grad_mean_of_points(union_of_points, self.num_to_sample)
        grad_chol_decomp = self._gaussian_process.compute_grad_cholesky_variance_of_points(union_of_points, self.num_to_sample)

        if num_points == 1 and force_monte_carlo is False:
            return self._compute_grad_expected_improvement_1d_analytic(
                mu_star[0],
                var_star[0, 0],
                grad_mu[0, ...],
                grad_chol_decomp[0, 0, 0, ...],
            )
        else:
            return self._compute_grad_expected_improvement_monte_carlo(
                mu_star,
                var_star,
                grad_mu,
                grad_chol_decomp,
            )

    def compute_grad_objective_function(self, **kwargs):
        """Wrapper for compute_grad_expected_improvement; see that function's docstring."""
        return self.compute_grad_expected_improvement(**kwargs)

    def compute_hessian_objective_function(self, **kwargs):
        """We do not currently support computation of the (spatial) hessian of Expected Improvement."""
        raise NotImplementedError('Currently we cannot compute the hessian of expected improvement.')
