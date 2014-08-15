# -*- coding: utf-8 -*-
"""Test the Python implementation of Expected Improvement and its gradient."""
import numpy

import testify as T

import moe
from moe.optimal_learning.python.data_containers import HistoricalData, SamplePoint
from moe.optimal_learning.python.geometry_utils import ClosedInterval
from moe.optimal_learning.python.python_version.covariance import SquareExponential
from moe.optimal_learning.python.python_version.domain import TensorProductDomain
from moe.optimal_learning.python.python_version.expected_improvement import multistart_expected_improvement_optimization, ExpectedImprovement
from moe.optimal_learning.python.python_version.gaussian_process import GaussianProcess
from moe.optimal_learning.python.python_version.optimization import GradientDescentParameters, GradientDescentOptimizer, LBFGSBParameters, LBFGSBOptimizer
from moe.optimal_learning.python.repeated_domain import RepeatedDomain
from moe.tests.optimal_learning.python.gaussian_process_test_case import GaussianProcessTestCase, GaussianProcessTestEnvironmentInput


class ExpectedImprovementTest(GaussianProcessTestCase):

    """Verify that the "naive" and "vectorized" EI implementations in Python return the same result.

    The code for the naive implementation of EI is straightforward to read whereas the vectorized version is a lot more
    opaque. So we verify one against the other.

    Fully verifying the monte carlo implemetation (e.g., conducting convergence tests, comparing against analytic results)
    is expensive and already a part of the C++ unit test suite.

    """

    precompute_gaussian_process_data = True

    noise_variance_base = 0.002
    dim = 3
    num_hyperparameters = dim + 1

    gp_test_environment_input = GaussianProcessTestEnvironmentInput(
        dim,
        num_hyperparameters,
        0,
        noise_variance_base=noise_variance_base,
        hyperparameter_interval=ClosedInterval(3.0, 5.0),
        lower_bound_interval=ClosedInterval(-2.0, 0.5),
        upper_bound_interval=ClosedInterval(2.0, 3.5),
        covariance_class=SquareExponential,
        spatial_domain_class=TensorProductDomain,
        hyperparameter_domain_class=TensorProductDomain,
        gaussian_process_class=GaussianProcess,
    )

    num_sampled_list = (1, 2, 5, 10, 16, 20, 50)

    num_mc_iterations = 747
    rng_seed = 314

    approx_grad = True
    max_func_evals = 150000
    max_metric_correc = 10
    factr = 10.0
    pgtol = 1e-10
    epsilon = 1e-8
    BFGS_parameters = LBFGSBParameters(
        approx_grad,
        max_func_evals,
        max_metric_correc,
        factr,
        pgtol,
        epsilon,
    )

    @T.class_setup
    def base_setup(self):
        """Run the standard setup but seed the RNG first (for repeatability).

        It is easy to stumble into test cases where EI is very small (e.g., < 1.e-20),
        which makes it difficult to set meaningful tolerances for the checks.

        """
        numpy.random.seed(7859)
        super(ExpectedImprovementTest, self).base_setup()

    def test_expected_improvement_and_gradient(self):
        """Test EI by comparing the vectorized and "naive" versions.

        With the same RNG state, these two functions should return identical output.
        We use a fairly low number of monte-carlo iterations since we are not
        trying to converge; just check for consistency.

        .. Note:: this is not a particularly good test. It relies on the "naive"
          version being easier to verify manually and only checks for consistency
          between the naive and vectorized versions.

        """
        num_points_p_q_list = ((1, 0), (1, 1), (2, 1), (1, 4), (5, 3))
        ei_tolerance = numpy.finfo(numpy.float64).eps
        grad_ei_tolerance = 1.0e-13
        numpy.random.seed(78532)

        for test_case in self.gp_test_environments:
            domain, gaussian_process = test_case

            for num_to_sample, num_being_sampled in num_points_p_q_list:
                points_to_sample = domain.generate_uniform_random_points_in_domain(num_to_sample)
                points_being_sampled = domain.generate_uniform_random_points_in_domain(num_being_sampled)

                union_of_points = numpy.reshape(numpy.append(points_to_sample, points_being_sampled), (num_to_sample + num_being_sampled, self.dim))
                ei_eval = ExpectedImprovement(
                    gaussian_process,
                    points_to_sample,
                    points_being_sampled=points_being_sampled,
                    num_mc_iterations=self.num_mc_iterations,
                )

                # Compute quantities required for EI
                mu_star = ei_eval._gaussian_process.compute_mean_of_points(union_of_points)
                var_star = ei_eval._gaussian_process.compute_variance_of_points(union_of_points)

                # Check EI
                # Save state first to restore at the end (o/w other "random" events will get screwed up)
                rng_state = numpy.random.get_state()
                numpy.random.seed(self.rng_seed)
                ei_vectorized = ei_eval._compute_expected_improvement_monte_carlo(mu_star, var_star)
                numpy.random.seed(self.rng_seed)
                ei_naive = ei_eval._compute_expected_improvement_monte_carlo_naive(mu_star, var_star)
                self.assert_scalar_within_relative(ei_vectorized, ei_naive, ei_tolerance)

                # Compute quantities required for grad EI
                grad_mu = ei_eval._gaussian_process.compute_grad_mean_of_points(
                    union_of_points,
                    num_derivatives=num_to_sample,
                )
                grad_chol_decomp = ei_eval._gaussian_process.compute_grad_cholesky_variance_of_points(
                    union_of_points,
                    num_derivatives=num_to_sample,
                )

                # Check grad EI
                numpy.random.seed(self.rng_seed)
                grad_ei_vectorized = ei_eval._compute_grad_expected_improvement_monte_carlo(
                    mu_star,
                    var_star,
                    grad_mu,
                    grad_chol_decomp,
                )
                numpy.random.seed(self.rng_seed)
                grad_ei_naive = ei_eval._compute_grad_expected_improvement_monte_carlo_naive(
                    mu_star,
                    var_star,
                    grad_mu,
                    grad_chol_decomp,
                )
                self.assert_vector_within_relative(grad_ei_vectorized, grad_ei_naive, grad_ei_tolerance)

                # Restore state
                numpy.random.set_state(rng_state)

    def _check_ei_symmetry(self, ei_eval, point_to_sample, shifts):
        """Compute ei at each ``[point_to_sample +/- shift for shift in shifts]`` and check for equality.

        :param ei_eval: properly configured ExpectedImprovementEvaluator object
        :type ei_eval: ExpectedImprovementInterface subclass
        :param point_to_sample: point at which to center the shifts
        :type point_to_sample: array of float64 with shape (1, )
        :param shifts: shifts to use in the symmetry check
        :type shifts: tuple of float64
        :return: None; assertions fail if test conditions are not met

        """
        for shift in shifts:
            ei_eval.current_point = point_to_sample - shift
            left_ei = ei_eval.compute_expected_improvement()
            left_grad_ei = ei_eval.compute_grad_expected_improvement()

            ei_eval.current_point = point_to_sample + shift
            right_ei = ei_eval.compute_expected_improvement()
            right_grad_ei = ei_eval.compute_grad_expected_improvement()

            self.assert_scalar_within_relative(left_ei, right_ei, 0.0)
            self.assert_vector_within_relative(left_grad_ei, -right_grad_ei, 0.0)

    def test_1d_analytic_ei_edge_cases(self):
        """Test cases where analytic EI would attempt to compute 0/0 without variance lower bounds."""
        base_coord = numpy.array([0.5])
        point1 = SamplePoint(base_coord, -1.809342, 0)
        point2 = SamplePoint(base_coord * 2.0, -1.09342, 0)

        # First a symmetric case: only one historical point
        data = HistoricalData(base_coord.size, [point1])

        hyperparameters = numpy.array([0.2, 0.3])
        covariance = SquareExponential(hyperparameters)
        gaussian_process = GaussianProcess(covariance, data)

        point_to_sample = base_coord
        ei_eval = ExpectedImprovement(gaussian_process, point_to_sample)

        ei = ei_eval.compute_expected_improvement()
        grad_ei = ei_eval.compute_grad_expected_improvement()
        self.assert_scalar_within_relative(ei, 0.0, 1.0e-15)
        self.assert_vector_within_relative(grad_ei, numpy.zeros(grad_ei.shape), 1.0e-15)

        shifts = (1.0e-15, 4.0e-11, 3.14e-6, 8.89e-1, 2.71)
        self._check_ei_symmetry(ei_eval, point_to_sample, shifts)

        # Now introduce some asymmetry with a second point
        # Right side has a larger objetive value, so the EI minimum
        # is shifted *slightly* to the left of best_so_far.
        gaussian_process.add_sampled_points([point2])
        shift = 3.0e-12
        ei_eval = ExpectedImprovement(gaussian_process, point_to_sample - shift)
        ei = ei_eval.compute_expected_improvement()
        grad_ei = ei_eval.compute_grad_expected_improvement()
        self.assert_scalar_within_relative(ei, 0.0, 1.0e-15)
        self.assert_vector_within_relative(grad_ei, numpy.zeros(grad_ei.shape), 1.0e-15)

    def test_evaluate_ei_at_points(self):
        """Check that ``evaluate_expected_improvement_at_point_list`` computes and orders results correctly (using 1D analytic EI)."""
        index = numpy.argmax(numpy.greater_equal(self.num_sampled_list, 5))
        domain, gaussian_process = self.gp_test_environments[index]

        points_to_sample = domain.generate_random_point_in_domain()
        ei_eval = ExpectedImprovement(gaussian_process, points_to_sample)

        num_to_eval = 10
        # Add in a newaxis to make num_to_sample explicitly 1
        points_to_evaluate = domain.generate_uniform_random_points_in_domain(num_to_eval)[:, numpy.newaxis, :]

        test_values = ei_eval.evaluate_at_point_list(points_to_evaluate)

        for i, value in enumerate(test_values):
            ei_eval.current_point = points_to_evaluate[i, ...]
            truth = ei_eval.compute_expected_improvement()
            T.assert_equal(value, truth)

    def test_multistart_analytic_expected_improvement_optimization(self):
        """Check that multistart optimization (gradient descent) can find the optimum point to sample (using 1D analytic EI)."""
        numpy.random.seed(3148)
        index = numpy.argmax(numpy.greater_equal(self.num_sampled_list, 20))
        domain, gaussian_process = self.gp_test_environments[index]

        max_num_steps = 200  # this is generally *too few* steps; we configure it this way so the test will run quickly
        max_num_restarts = 5
        num_steps_averaged = 0
        gamma = 0.2
        pre_mult = 1.5
        max_relative_change = 1.0
        tolerance = 1.0e-7
        gd_parameters = GradientDescentParameters(
            max_num_steps,
            max_num_restarts,
            num_steps_averaged,
            gamma,
            pre_mult,
            max_relative_change,
            tolerance,
        )
        num_multistarts = 3

        points_to_sample = domain.generate_random_point_in_domain()
        ei_eval = ExpectedImprovement(gaussian_process, points_to_sample)

        # expand the domain so that we are definitely not doing constrained optimization
        expanded_domain = TensorProductDomain([ClosedInterval(-4.0, 2.0)] * self.dim)

        num_to_sample = 1
        repeated_domain = RepeatedDomain(ei_eval.num_to_sample, expanded_domain)
        ei_optimizer = GradientDescentOptimizer(repeated_domain, ei_eval, gd_parameters)
        best_point = multistart_expected_improvement_optimization(ei_optimizer, num_multistarts, num_to_sample)

        # Check that gradients are small
        ei_eval.current_point = best_point
        gradient = ei_eval.compute_grad_expected_improvement()
        self.assert_vector_within_relative(gradient, numpy.zeros(gradient.shape), tolerance)

        # Check that output is in the domain
        T.assert_equal(repeated_domain.check_point_inside(best_point), True)

    def test_multistart_monte_carlo_expected_improvement_optimization(self):
        """Check that multistart optimization (gradient descent) can find the optimum point to sample (using 2-EI)."""
        numpy.random.seed(7858)  # TODO(271): Monte Carlo only works for this seed
        index = numpy.argmax(numpy.greater_equal(self.num_sampled_list, 20))
        domain, gaussian_process = self.gp_test_environments[index]

        max_num_steps = 75  # this is *too few* steps; we configure it this way so the test will run quickly
        max_num_restarts = 5
        num_steps_averaged = 50
        gamma = 0.2
        pre_mult = 1.5
        max_relative_change = 1.0
        tolerance = 3.0e-2  # really large tolerance b/c converging with monte-carlo (esp in Python) is expensive
        gd_parameters = GradientDescentParameters(
            max_num_steps,
            max_num_restarts,
            num_steps_averaged,
            gamma,
            pre_mult,
            max_relative_change,
            tolerance,
        )
        num_multistarts = 2

        # Expand the domain so that we are definitely not doing constrained optimization
        expanded_domain = TensorProductDomain([ClosedInterval(-4.0, 2.0)] * self.dim)
        num_to_sample = 2
        repeated_domain = RepeatedDomain(num_to_sample, expanded_domain)

        num_mc_iterations = 10000
        # Just any random point that won't be optimal
        points_to_sample = repeated_domain.generate_random_point_in_domain()
        ei_eval = ExpectedImprovement(gaussian_process, points_to_sample, num_mc_iterations=num_mc_iterations)
        # Compute EI and its gradient for the sake of comparison
        ei_initial = ei_eval.compute_expected_improvement(force_monte_carlo=True)  # TODO(271) Monte Carlo only works for this seed
        grad_ei_initial = ei_eval.compute_grad_expected_improvement()

        ei_optimizer = GradientDescentOptimizer(repeated_domain, ei_eval, gd_parameters)
        best_point = multistart_expected_improvement_optimization(ei_optimizer, num_multistarts, num_to_sample)

        # Check that gradients are "small"
        ei_eval.current_point = best_point
        ei_final = ei_eval.compute_expected_improvement(force_monte_carlo=True)  # TODO(271) Monte Carlo only works for this seed
        grad_ei_final = ei_eval.compute_grad_expected_improvement()
        self.assert_vector_within_relative(grad_ei_final, numpy.zeros(grad_ei_final.shape), tolerance)

        # Check that output is in the domain
        T.assert_equal(repeated_domain.check_point_inside(best_point), True)

        # Since we didn't really converge to the optimal EI (too costly), do some other sanity checks
        # EI should have improved
        T.assert_gt(ei_final, ei_initial)

        # grad EI should have improved
        for index in numpy.ndindex(grad_ei_final.shape):
            T.assert_lt(numpy.fabs(grad_ei_final[index]), numpy.fabs(grad_ei_initial[index]))

    def test_multistart_qei_expected_improvement_dfo(self):
        """Check that multistart optimization (BFGS) can find the optimum point to sample (using 2-EI)."""
        numpy.random.seed(7860)
        index = numpy.argmax(numpy.greater_equal(self.num_sampled_list, 20))
        domain, gaussian_process = self.gp_test_environments[index]

        tolerance = 6.0e-5
        num_multistarts = 3

        # Expand the domain so that we are definitely not doing constrained optimization
        expanded_domain = TensorProductDomain([ClosedInterval(-4.0, 3.0)] * self.dim)
        num_to_sample = 2
        repeated_domain = RepeatedDomain(num_to_sample, expanded_domain)

        num_mc_iterations = 100000
        # Just any random point that won't be optimal
        points_to_sample = repeated_domain.generate_random_point_in_domain()
        ei_eval = ExpectedImprovement(gaussian_process, points_to_sample, num_mc_iterations=num_mc_iterations)
        # Compute EI and its gradient for the sake of comparison
        ei_initial = ei_eval.compute_expected_improvement()

        ei_optimizer = LBFGSBOptimizer(repeated_domain, ei_eval, self.BFGS_parameters)
        best_point = multistart_expected_improvement_optimization(ei_optimizer, num_multistarts, num_to_sample)

        # Check that gradients are "small" or on border. MC is very inaccurate near 0, so use finite difference
        # gradient instead.
        ei_eval.current_point = best_point
        ei_final = ei_eval.compute_expected_improvement()

        finite_diff_grad = numpy.zeros(best_point.shape)
        h_value = 0.00001
        for i in range(best_point.shape[0]):
            for j in range(best_point.shape[1]):
                best_point[i, j] += h_value
                ei_eval.current_point = best_point
                ei_upper = ei_eval.compute_expected_improvement()
                best_point[i, j] -= 2 * h_value
                ei_eval.current_point = best_point
                ei_lower = ei_eval.compute_expected_improvement()
                best_point[i, j] += h_value
                finite_diff_grad[i, j] = (ei_upper - ei_lower) / (2 * h_value)

        self.assert_vector_within_relative(finite_diff_grad, numpy.zeros(finite_diff_grad.shape), tolerance)

        # Check that output is in the domain
        T.assert_true(repeated_domain.check_point_inside(best_point))

        # Since we didn't really converge to the optimal EI (too costly), do some other sanity checks
        # EI should have improved
        T.assert_gt(ei_final, ei_initial)

    def test_qd_ei_with_self(self):
        """Compare the 1D analytic EI results to the qD analytic EI results, checking several random points per test case.

        This test case (unfortunately) suffers from a lot of random variation in the qEI parameters. The tolerance is high because
        changing the number of iterations or the maximum relative error allowed in the mvndst function leads to different answers.

        These precomputed answers were calculated from:
        maxpts = 20,000 * q
        releps = 1e-5

        These values are a tradeoff between accuracy / speed.
        """
        ei_tolerance = 6.0e-6
        numpy.random.seed(8790)

        precomputed_answers = [
            5.83583593191e-08,
            5.83583593328e-08,
            2.40176803674e-07,
            0.349595041008,
            0.350524674435,
            0.350524720501,
            0.350524799491,
            0.409585935352,
        ]

        for test_case in self.gp_test_environments[2:3]:
            domain, python_gp = test_case
            all_points = domain.generate_uniform_random_points_in_domain(9)

            for i in range(2, 10):
                points_to_sample = all_points[0:i]
                mvndst_accurate_parameters = moe.optimal_learning.python.python_version.expected_improvement.MVNDSTParameters(
                        releps=1.0e-6,
                        abseps=1.0e-6,
                        maxpts_per_dim=200000,
                        )
                python_ei_eval = moe.optimal_learning.python.python_version.expected_improvement.ExpectedImprovement(
                        python_gp,
                        points_to_sample,
                        mvndst_parameters=mvndst_accurate_parameters
                        )

                python_ei_eval.current_point = points_to_sample
                python_qd_ei = python_ei_eval.compute_expected_improvement()
                self.assert_scalar_within_relative(python_qd_ei, precomputed_answers[i - 2], ei_tolerance)

    def test_qd_and_1d_return_same_analytic_ei(self):
        """Compare the 1D analytic EI results to the qD analytic EI results, checking several random points per test case."""
        num_tests_per_case = 10
        ei_tolerance = 1.0e-9

        for test_case in self.gp_test_environments:
            domain, python_gp = test_case
            points_to_sample = domain.generate_random_point_in_domain()
            python_ei_eval = moe.optimal_learning.python.python_version.expected_improvement.ExpectedImprovement(python_gp, points_to_sample)

            for _ in xrange(num_tests_per_case):
                points_to_sample = domain.generate_random_point_in_domain()
                python_ei_eval.current_point = points_to_sample

                python_1d_ei = python_ei_eval.compute_expected_improvement(force_1d_ei=True)
                python_qd_ei = python_ei_eval.compute_expected_improvement()
                self.assert_scalar_within_relative(python_1d_ei, python_qd_ei, ei_tolerance)


if __name__ == "__main__":
    T.run()
