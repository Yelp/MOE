# -*- coding: utf-8 -*-
"""Classes (Python) to compute the Expected Improvement, including monte carlo and analytic (where applicable) implementations.

See interfaces/expected_improvement_interface.py or gpp_math.hpp/cpp for further details on expected improvement.

"""
import numpy
import scipy.linalg
import scipy.stats

from moe.optimal_learning.python.interfaces.expected_improvement_interface import ExpectedImprovementInterface
from moe.optimal_learning.python.interfaces.optimization_interface import OptimizableInterface
from moe.optimal_learning.python.python_version.optimization import multistart_optimize, NullOptimizer


def multistart_expected_improvement_optimization(
        ei_optimizer,
        num_multistarts,
        num_samples_to_generate,
        randomness=None,
        max_num_threads=1,
        status=None,
):
    """Solve the q,p-EI problem, returning the optimal set of q points to sample CONCURRENTLY in future experiments.

    When points_to_sample.shape[0] == 0 && num_samples_to_generate == 1, this function will use (fast) analytic EI computations.

    .. NOTE:: The following comments are copied from multistart_expected_improvement_optimization() in cpp_wrappers/expected_improvement.py

    Returns the optimal set of q points to sample CONCURRENTLY by solving the q,p-EI problem.  That is, we may want to run 4
    experiments at the same time and maximize the EI across all 4 experiments at once while knowing of 2 ongoing experiments
    (4,2-EI). This function handles this use case. Evaluation of q,p-EI (and its gradient) for q > 1 or p > 1 is expensive
    (requires monte-carlo iteration), so this method is usually very expensive.

    Compared to ComputeHeuristicSetOfPointsToSample() (``gpp_heuristic_expected_improvement_optimization.hpp``), this function
    makes no external assumptions about the underlying objective function. Instead, it utilizes the Expected (Parallel)
    Improvement, allowing the GP to account for ongoing/incomplete experiments.

    If ``num_samples_to_generate = 1``, this is the same as ComputeOptimalPointToSampleWithRandomStarts().

    :param ei_optimizer: object that optimizes (e.g., gradient descent, newton) EI over a domain
    :type ei_optimizer: interfaces.optimization_interfaces.OptimizerInterface subclass
    :param num_multistarts: number of times to multistart ``ei_optimizer``
    :type num_multistarts: int > 0
    :param num_samples_to_generate: how many simultaneous experiments you would like to run (i.e., the q in q,p-EI)
    :type num_samples_to_generate: int >= 1
    :param randomness: ?? (UNUSED)
    :type randomness: ?? (UNUSED)
    :param max_num_threads: maximum number of threads to use, >= 1 (UNUSED)
    :type max_num_threads: int > 0
    :param status: status messages from C++ (e.g., reporting on optimizer success, etc.)
    :type status: dict
    :return: point(s) that maximize the expected improvement (solving the q,p-EI problem)
    :rtype: array of float64 with shape (num_samples_to_generate, ei_evaluator.dim)

    """
    # TODO(eliu): implement code to generate a set of points to sample instead of only 1 (ADS-3094)
    if num_samples_to_generate != 1:
        raise ValueError('num_samples_to_generate = %s must be 1. Other cases not implemented yet.' % num_samples_to_generate)

    random_starts = ei_optimizer.domain.generate_uniform_random_points_in_domain(num_points=num_multistarts)
    best_point, _ = multistart_optimize(ei_optimizer, starting_points=random_starts)

    # TODO(eliu): have GD actually indicate whether updates were found (GH-59)
    found_flag = True
    if status is not None:
        status["gradient_descent_found_update"] = found_flag

    return best_point


def evaluate_expected_improvement_at_point_list(
        ei_evaluator,
        points_to_evaluate,
        randomness=None,
        max_num_threads=1,
        status=None,
):
    """Evaluate Expected Improvement (1,p-EI) over a specified list of ``points_to_evaluate``.

    Generally gradient descent is preferred but when it fails to converge this may be the only "robust" option.
    This function is also useful for plotting or debugging purposes (just to get a bunch of EI values).

    :param ei_evaluator: object specifying how to evaluate the expected improvement
    :type ei_evaluator: cpp_wrappers.expected_improvement.ExpectedImprovement
    :param points_to_evaluate: points at which to compute EI
    :type points_to_evaluate: array of float64 with shape (num_to_evaluate, ei_evaluator.dim)
    :param randomness: ??? (UNUSED)
    :type randomness: ???
    :param max_num_threads: maximum number of threads to use, >= 1 (UNUSED)
    :type max_num_threads: int > 0
    :param status: status messages from C++ (e.g., reporting on optimizer success, etc.)
    :type status: dict
    :return: EI evaluated at each of points_to_evaluate
    :rtype: array of float64 with shape (points_to_evaluate.shape[0])

    """
    null_optimizer = NullOptimizer(None, ei_evaluator)
    _, values = multistart_optimize(null_optimizer, starting_points=points_to_evaluate)

    # TODO(eliu): have multistart actually indicate whether updates were found (GH-59)
    found_flag = True
    if status is not None:
        status["evaluate_EI_at_point_list"] = found_flag

    return values


class ExpectedImprovement(ExpectedImprovementInterface, OptimizableInterface):

    r"""Implementation of Expected Improvement computation in Python: EI and its gradient at specified point(s) sampled from a GaussianProcess.

    A class to encapsulate the computation of expected improvement and its spatial gradient using points sampled from an
    associated GaussianProcess. The general EI computation requires monte-carlo integration; it can support q,p-EI optimization.
    It is designed to work with any GaussianProcess.

    When available, fast, analytic formulas replace monte-carlo loops.

    See interfaces/expected_improvement_interface.py docs for further details.

    """

    def __init__(self, gaussian_process, current_point, points_to_sample=numpy.array([]), num_mc_iterations=1000, randomness=None):
        """Construct an ExpectedImprovement object that supports q,p-EI.

        :param gaussian_process: GaussianProcess describing
        :type gaussian_process: cpp_wrappers.GaussianProcess object
        :param current_point: point at which to compute EI (i.e., q in q,p-EI)
        :type current_point: array of float64 with shape (dim)
        :param points_to_sample: points which are being sampled concurrently (i.e., p in q,p-EI)
        :type points_to_sample: array of float64 with shape (num_to_sample, dim)
        :param num_mc_iterations: number of monte-carlo iterations to use (when monte-carlo integration is used to compute EI)
        :type num_mc_iterations: int > 0
        :param randomness: ???
        :type randomness: ???

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
            pass  # TODO(eliu): WHAT TO DO HERE

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

    def _compute_expected_improvement_1D_analytic(self, mu_star, var_star):
        """Compute EI when the number of potential samples is 1 (i.e., points_to_sample.size = 0) using *fast* analytic methods.

        This function can only support the computation of 1,0-EI. In this case, we have analytic formulas
        for computing EI (and its gradient).

        Thus this function does not perform any explicit numerical integration, nor does it require access to a
        random number generator.

        See Ginsbourger, Le Riche, and Carraro.

        :param mu_star: the mean of the GP evaluated at current_point
        :type mu_star: float64
        :param var_star: the variance of the GP evaluated at current_point
        :type var_star: float64
        :return: value of EI evaluated at ``current_point``
        :rtype: float64

        """
        sigma_star = numpy.sqrt(var_star)
        temp = self._best_so_far - mu_star
        EI = temp * scipy.stats.norm.cdf(temp / sigma_star) + sigma_star * scipy.stats.norm.pdf(temp / sigma_star)
        return numpy.fmax(0.0, EI)

    def _compute_grad_expected_improvement_1D_analytic(self, mu_star, var_star, grad_mu, grad_chol_decomp):
        """Compute the gradient of EI when the number of potential samples is 1 (i.e., points_to_sample.size = 0) using *fast* analytic methods.

        This function can only support the computation of 1,0-EI. In this case, we have analytic formulas
        for computing EI and its gradient.

        Thus this function does not perform any explicit numerical integration, nor does it require access to a
        random number generator.

        See Ginsbourger, Le Riche, and Carraro.

        :param mu_star: the mean of the GP evaluated at current_point
        :type mu_star: float64
        :param var_star: the variance of the GP evaluated at current_point
        :type var_star: float64
        :param grad_mu: the gradient of the mean of the GP evaluated at current_point, wrt current_point
        :type grad_mu: array of float64 with shape (self.dim)
        :param grad_chol_decomp: the gradient of the variance of the GP evaluated at current_point, wrt current_point
        :type grad_chol_decomp: array of float64 with shape (self.dim)
        :return: gradient of EI evaluated at ``current_point`` wrt ``current_point``
        :rtype: array of float64 with shape (self.dim)

        """
        sigma_star = numpy.sqrt(var_star)
        mu_diff = self._best_so_far - mu_star
        C = mu_diff / sigma_star
        pdf_C = scipy.stats.norm.pdf(C)
        cdf_C = scipy.stats.norm.cdf(C)

        d_C = (-sigma_star * grad_mu - grad_chol_decomp * mu_diff) / var_star
        d_A = -grad_mu * cdf_C + mu_diff * pdf_C * d_C
        d_B = grad_chol_decomp * pdf_C - sigma_star * C * pdf_C * d_C

        d_A += d_B
        return d_A

    def _compute_expected_improvement_monte_carlo_naive(self, num_points, mu_star, var_star):
        """Compute EI using (naive) monte carlo integration.

        See _compute_expected_improvement_monte_carlo (below) for more details on EI.

        This function produces the *exact* same output as the vectorized version. But
        since the loop over num_mc_iterations runs in Python, it is > 100x slower.
        However, this code is easy to verify b/c it follows the algorithmic description
        faithfully. We use it only for verification.

        :param num_points: number of points (q + p) at which EI is being computed
        :type num_points: int > 1
        :param mu_star: self._gaussian_process.compute_mean_of_points(union_of_points)
        :type mu_star: array of float64 with shape (num_points)
        :param var_star: self._gaussian_process.compute_variance_of_points(union_of_points)
        :type var_star: array of float64 with shape (num_points, num_points)
        :return: value of EI evaluated at ``current_point``
        :rtype: float64

        """
        chol_var = scipy.linalg.cholesky(var_star, lower=True)

        normals = numpy.random.normal(size=(self._num_mc_iterations, num_points))

        aggregate = 0.0
        for normal_draws in normals:
            improvement_this_iter = numpy.amax(self._best_so_far - mu_star - numpy.dot(chol_var, normal_draws.T))
            if improvement_this_iter > 0.0:
                aggregate += improvement_this_iter
        return aggregate / float(self._num_mc_iterations)

    def _compute_expected_improvement_monte_carlo(self, num_points, mu_star, var_star):
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

        :param num_points: number of points (q + p) at which EI is being computed
        :type num_points: int > 1
        :param mu_star: self._gaussian_process.compute_mean_of_points(union_of_points)
        :type mu_star: array of float64 with shape (num_points)
        :param var_star: self._gaussian_process.compute_variance_of_points(union_of_points)
        :type var_star: array of float64 with shape (num_points, num_points)
        :return: value of EI evaluated at ``current_point``
        :rtype: float64

        """
        chol_var = -scipy.linalg.cholesky(var_star, lower=True)

        normals = numpy.random.normal(size=(self._num_mc_iterations, num_points))

        # TODO(eliu): might be worth breaking num_mc_iterations up into smaller blocks
        # so that we don't waste as much mem bandwidth (since each entry of normals is
        # only used once) (GH-60)
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

    def _compute_grad_expected_improvement_monte_carlo_naive(self, num_points, mu_star, var_star, grad_mu, grad_chol_decomp):
        """Compute the gradient of EI using (naive) monte carlo integration.

        See _compute_grad_expected_improvement_monte_carlo (below) for more details on EI.

        This function produces the *exact* same output as the vectorized version. But
        since the loop over num_mc_iterations runs in Python, it is > 100x slower.
        However, this code is easy to verify b/c it follows the algorithmic description
        faithfully. We use it only for verification.

        :param num_points: number of points (q + p) at which EI is being computed
        :type num_points: int > 1
        :param mu_star: self._gaussian_process.compute_mean_of_points(union_of_points)
        :type mu_star: array of float64 with shape (num_points)
        :param var_star: self._gaussian_process.compute_variance_of_points(union_of_points)
        :type var_star: array of float64 with shape (num_points, num_points)
        :param grad_mu: self._gaussian_process.compute_grad_mean_of_points(union_of_points)
        :type grad_mu: array of float64 with shape (num_points, self.dim)
        :param grad_chol_decomp: self._gaussian_process.compute_grad_cholesky_variance_of_points(union_of_points, 0)
        :type grad_chol_decomp: array of float64 with shape (num_points, num_points, self.dim)
        :return: gradient of EI evaluated at ``current_point`` wrt ``current_point``
        :rtype: array of float64 with shape (self.dim)

        """
        # Differentiating wrt point 0 in self._current_point
        diff_index = 0
        chol_var = scipy.linalg.cholesky(var_star, lower=True)

        normals = numpy.random.normal(size=(self._num_mc_iterations, num_points))

        aggregate_dx = numpy.zeros(self.dim)
        for normal_draws in normals:
            improvements_this_iter = self._best_so_far - mu_star - numpy.dot(chol_var, normal_draws.T)
            if numpy.amax(improvements_this_iter) > 0.0:
                winner = numpy.argmax(improvements_this_iter)
                if winner == diff_index:
                    aggregate_dx -= grad_mu[diff_index, ...]
                # grad_chol_decomp_{winner, i, j} * normal_draws_{i}
                aggregate_dx -= numpy.dot(grad_chol_decomp[winner, ...].T, normal_draws)

        return aggregate_dx / float(self._num_mc_iterations)

    def _compute_grad_expected_improvement_monte_carlo(self, num_points, mu_star, var_star, grad_mu, grad_chol_decomp):
        r"""Compute the gradient of EI using (vectorized) monte-carlo integration; this is a general method that works for any input.

        This function cal support the computation of q,p-EI.
        This function requires access to a random number generator.

        .. Note:: comments here are copied from gpp_math.cpp, ExpectedImprovementEvaluator::ComputeGradExpectedImprovement().

        Computes gradient of EI (see ExpectedImprovementEvaluator::ComputeGradExpectedImprovement) wrt current_point.

        Mechanism is similar to the computation of EI, where points' contributions to the gradient are thrown out of their
        corresponding ``improvement <= 0.0``.  There is some additional subtlety here because we are only computing the gradient
        of EI with respect to the current point (stored at index ``index_of_current_point``).

        Thus ``\nabla(\mu)`` only contributes when the ``winner`` (point w/best improvement this iteration) is the current point.
        That is, the gradient of ``\mu`` at ``x_i`` wrt ``x_j`` is 0 unless ``i == j`` (and only this result is stored in
        ``grad_mu``).  The interaction with ``grad_chol_decomp`` is harder to know a priori (like with
        ``grad_mu``) and has a more complex structure (rank 3 tensor), so the derivative wrt ``x_j`` is computed fully, and
        the relevant submatrix (indexed by the current ``winner``) is accessed each iteration.

        For performance, this function vectorizes the monte-carlo integration loop, using numpy's mask feature to skip
        iterations where the improvement is not positive. Some additional cleverness is required to vectorize the
        accesses into grad_chol_decomp, since we cannot afford to run a loop (even over ``normals_compressed``) in Python.

        :param num_points: number of points (q + p) at which EI is being computed
        :type num_points: int > 1
        :param mu_star: self._gaussian_process.compute_mean_of_points(union_of_points)
        :type mu_star: array of float64 with shape (num_points)
        :param var_star: self._gaussian_process.compute_variance_of_points(union_of_points)
        :type var_star: array of float64 with shape (num_points, num_points)
        :param grad_mu: self._gaussian_process.compute_grad_mean_of_points(union_of_points)
        :type grad_mu: array of float64 with shape (num_points, self.dim)
        :param grad_chol_decomp: self._gaussian_process.compute_grad_cholesky_variance_of_points(union_of_points, 0)
        :type grad_chol_decomp: array of float64 with shape (num_points, num_points, self.dim)
        :return: gradient of EI evaluated at ``current_point`` wrt ``current_point``
        :rtype: array of float64 with shape (self.dim)

        """
        # Differentiating wrt point 0 in self._current_point
        diff_index = 0
        chol_var = -scipy.linalg.cholesky(var_star, lower=True)

        normals = numpy.random.normal(size=(self._num_mc_iterations, num_points))

        # TODO(eliu): might be worth breaking num_mc_iterations up into smaller blocks
        # so that we don't waste as much mem bandwidth (since each entry of normals is
        # only used once) (GH-60)
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

        # Indexes where the winner was point 0 (the point we're differentiating against)
        winner_indexes_equal_to_diff_index = numpy.ma.masked_not_equal(winner_indexes, diff_index)

        # Handle derivative terms from grad_mu
        total_winners_equal_to_diff_index = numpy.ma.count(winner_indexes_equal_to_diff_index)
        aggregate_dx = -grad_mu[diff_index, ...] * total_winners_equal_to_diff_index

        # Handle derivative terms from grad_chol_decomp
        # Drop the masked terms
        winner_indexes_compressed = numpy.ma.compressed(winner_indexes)

        # Mask rows of normals that did not show positive improvement
        # TODO(eliu): can this be done with numpy.tile, numpy.repeat or something more sensical? (GH-61)
        normals_mask = numpy.empty(normals.shape, dtype=bool)
        normals_mask[...] = best_improvement_each_iter.mask[:, numpy.newaxis]
        # Compress out the masked data
        normals_compressed = numpy.ma.array(normals, mask=normals_mask)
        # We'd like to use numpy.ma.compress_rows but somehow that is REALLY slow, like N^2 slow
        normals_compressed = normals_compressed[~normals_compressed.mask].reshape((numpy.ma.count(winner_indexes), num_points))

        # We now want to compute: grad_chol_decomp[winner_index, i, j] * normals[k, i]
        # And sum over k.
        # To do this loop in numpy, we have to create grad_chol_decomp_tiled:
        # for k in xrange(self._num_mc_iterations):
        #   grad_chol_decomp_tiled[k, ...] = grad_chol_decomp[winner_indexes[k], ...]
        # Except we make two optimizations:
        # 1) We skip all the masked terms (so we use the compressed arrays)
        # 2) We vectorize the tiling process.
        grad_chol_decomp_tiled = numpy.zeros((normals_compressed.shape[0], grad_chol_decomp.shape[1], grad_chol_decomp.shape[2]))
        for i in xrange(num_points):
            # Only track the iterations where point i had the best improvement (winner)
            winner_indexes_equal_to_i = numpy.ma.masked_not_equal(winner_indexes_compressed, i)

            # If all winners were index i, then the mask is a scalar.
            # We want to expand the mask into a full vector so that subsequent code will work. Kind of hacky.
            if winner_indexes_equal_to_i.mask is numpy.False_:
                # In fact we could stop here b/c this means index i won every time
                winner_indexes_equal_to_i.mask = [False]

            # Expand winner_indexes_equal_to_i.mask to cover the full shape of grad_chol_decomp_tiled
            # This is the same idea as normals_mask above
            # TODO(eliu): can I do this with numpy.tile, numpy.repeat or something more sensical? (GH-61)
            grad_chol_decomp_block_i_tile_mask = numpy.empty(grad_chol_decomp_tiled.shape, dtype=bool)
            grad_chol_decomp_block_i_tile_mask[...] = winner_indexes_equal_to_i.mask[:, numpy.newaxis, numpy.newaxis]

            # TODO(eliu): there has to be smarter way to do this! (GH-61)
            # Tile the appropriate block of grad_chol_decomp to *FILL* all blocks
            grad_chol_decomp_block_i_tile = numpy.tile(grad_chol_decomp[i, ...], (normals_compressed.shape[0], 1)).reshape((normals_compressed.shape[0], num_points, aggregate_dx.size))
            # Zero out blocks where the winner was not point i
            grad_chol_decomp_block_i_tile = numpy.ma.filled(numpy.ma.array(grad_chol_decomp_block_i_tile, mask=grad_chol_decomp_block_i_tile_mask), fill_value=0.0)
            # Add the tiles for this index into grad_chol_decomp_tiled
            # Note that since we zero all irrelevant blocks, we are never overwriting anything
            grad_chol_decomp_tiled += grad_chol_decomp_block_i_tile

        # Now we can compute the contribution from the variance in a fast C loop.
        aggregate_dx -= numpy.einsum('ki, kij', normals_compressed, grad_chol_decomp_tiled)

        # For reference, the above block replaces the following code:
        # for it, normal in enumerate(normals_compressed):
        #     aggregate_dx -= numpy.dot(normal, grad_chol_decomp[winner_indexes_compressed[it], ...])
        # The vectorized version performs exactly the same number of arithmetic operations in exactly the same order but
        # is at least 30x faster (difference grows with self._num_mc_iterations). Looping in Python is REALLY slow.

        aggregate_dx /= float(self._num_mc_iterations)
        return aggregate_dx

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

        mu_star = self._gaussian_process.compute_mean_of_points(union_of_points)
        var_star = self._gaussian_process.compute_variance_of_points(union_of_points)

        if num_points == 1 and force_monte_carlo is False:
            return self._compute_expected_improvement_1D_analytic(mu_star[0], var_star[0, 0])
        else:
            return self._compute_expected_improvement_monte_carlo(num_points, mu_star, var_star)

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
        num_points = 1 + self._points_to_sample.shape[0]
        union_of_points = numpy.reshape(numpy.append(self._current_point, self._points_to_sample), (num_points, self.dim))

        mu_star = self._gaussian_process.compute_mean_of_points(union_of_points)
        var_star = self._gaussian_process.compute_variance_of_points(union_of_points)
        grad_mu = self._gaussian_process.compute_grad_mean_of_points(union_of_points)
        grad_chol_decomp = self._gaussian_process.compute_grad_cholesky_variance_of_points(union_of_points, 0)

        if num_points == 1 and force_monte_carlo is False:
            return self._compute_grad_expected_improvement_1D_analytic(
                mu_star[0],
                var_star[0, 0],
                grad_mu[0, ...],
                grad_chol_decomp[0, 0, ...],
            )
        else:
            return self._compute_grad_expected_improvement_monte_carlo(
                num_points,
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
