# -*- coding: utf-8 -*-
"""Test the Python implementation of Expected Improvement and its gradient."""
import numpy

import testify as T

from moe.optimal_learning.python.geometry_utils import ClosedInterval
from moe.optimal_learning.python.python_version.covariance import SquareExponential
from moe.optimal_learning.python.python_version.domain import TensorProductDomain
from moe.optimal_learning.python.python_version.expected_improvement import multistart_expected_improvement_optimization, ExpectedImprovement
from moe.optimal_learning.python.python_version.gaussian_process import GaussianProcess
from moe.optimal_learning.python.python_version.optimization import GradientDescentParameters, GradientDescentOptimizer
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

    num_sampled_list = [1, 2, 5, 10, 16, 20, 50]

    num_mc_iterations = 747
    rng_seed = 314

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
        num_points_p_q_list = [(1, 0), (1, 1), (2, 1), (1, 4), (5, 3)]
        ei_tolerance = numpy.finfo('float64').eps
        grad_ei_tolerance = 1.0e-13
        numpy.random.seed(78532)

        for test_case in self.gp_test_environments:
            domain, _, gaussian_process = test_case

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

    def test_evaluate_ei_at_points(self):
        """Check that ``evaluate_expected_improvement_at_point_list`` computes and orders results correctly (using 1D analytic EI)."""
        index = numpy.argmax(numpy.greater_equal(self.num_sampled_list, 5))
        domain, _, gaussian_process = self.gp_test_environments[index]

        points_to_sample = domain.generate_random_point_in_domain()
        ei_eval = ExpectedImprovement(gaussian_process, points_to_sample)

        num_to_eval = 10
        # Add in a newaxis to make num_to_sample explicitly 1
        points_to_evaluate = domain.generate_uniform_random_points_in_domain(num_to_eval)[:, numpy.newaxis, :]

        test_values = ei_eval.evaluate_at_point_list(points_to_evaluate)

        for i, value in enumerate(test_values):
            ei_eval.set_current_point(points_to_evaluate[i, ...])
            truth = ei_eval.compute_expected_improvement()
            T.assert_equal(value, truth)

    def test_multistart_analytic_expected_improvement_optimization(self):
        """Check that multistart optimization (gradient descent) can find the optimum point to sample (using 1D analytic EI)."""
        numpy.random.seed(3148)
        index = numpy.argmax(numpy.greater_equal(self.num_sampled_list, 20))
        domain, _, gaussian_process = self.gp_test_environments[index]

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
        ei_eval.set_current_point(best_point)
        gradient = ei_eval.compute_grad_expected_improvement()
        self.assert_vector_within_relative(gradient, numpy.zeros(gradient.shape), tolerance)

        # Check that output is in the domain
        T.assert_equal(repeated_domain.check_point_inside(best_point), True)

    def test_multistart_mmonte_carlo_expected_improvement_optimization(self):
        """Check that multistart optimization (gradient descent) can find the optimum point to sample (using 2-EI)."""
        numpy.random.seed(7858)
        index = numpy.argmax(numpy.greater_equal(self.num_sampled_list, 20))
        domain, _, gaussian_process = self.gp_test_environments[index]

        max_num_steps = 75  # this is *too few* steps; we configure it this way so the test will run quickly
        max_num_restarts = 5
        num_steps_averaged = 50
        gamma = 0.2
        pre_mult = 1.5
        max_relative_change = 1.0
        tolerance = 1.0e-2  # really large tolerance b/c converging with monte-carlo (esp in Python) is expensive
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
        ei_initial = ei_eval.compute_expected_improvement()
        grad_ei_initial = ei_eval.compute_grad_expected_improvement()

        ei_optimizer = GradientDescentOptimizer(repeated_domain, ei_eval, gd_parameters)
        best_point = multistart_expected_improvement_optimization(ei_optimizer, num_multistarts, num_to_sample)

        # Check that gradients are "small"
        ei_eval.set_current_point(best_point)
        ei_final = ei_eval.compute_expected_improvement()
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


if __name__ == "__main__":
    T.run()
