# -*- coding: utf-8 -*-
"""Test the Python implementation of Expected Improvement and its gradient."""
import numpy
import testify as T

from moe.optimal_learning.python.geometry_utils import ClosedInterval
from moe.optimal_learning.python.python_version.covariance import SquareExponential
from moe.optimal_learning.python.python_version.domain import TensorProductDomain
from moe.optimal_learning.python.python_version.gaussian_process import GaussianProcess
from moe.optimal_learning.python.python_version.expected_improvement import multistart_expected_improvement_optimization, evaluate_expected_improvement_at_point_list, ExpectedImprovement
from moe.optimal_learning.python.python_version.optimization import GradientDescentParameters, GradientDescentOptimizer
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

    num_to_sample = 4
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
        num_tests_per_case = 4
        ei_tolerance = numpy.finfo('float64').eps
        grad_ei_tolerance = 1.0e-13

        for test_case in self.gp_test_environments:
            domain, _, gaussian_process = test_case
            points_to_sample = domain.generate_uniform_random_points_in_domain(self.num_to_sample)
            current_point = domain.generate_random_point_in_domain()
            ei_eval = ExpectedImprovement(gaussian_process, current_point, points_to_sample=points_to_sample, num_mc_iterations=self.num_mc_iterations)

            for _ in xrange(num_tests_per_case):
                num_points = 1 + points_to_sample.shape[0]
                current_point = domain.generate_random_point_in_domain()
                union_of_points = numpy.reshape(numpy.append(current_point, points_to_sample), (num_points, ei_eval.dim))
                ei_eval.set_current_point(current_point)

                # Compute quantities required for EI
                mu_star = ei_eval._gaussian_process.compute_mean_of_points(union_of_points)
                var_star = ei_eval._gaussian_process.compute_variance_of_points(union_of_points)

                # Check EI
                numpy.random.seed(self.rng_seed)
                ei_vectorized = ei_eval._compute_expected_improvement_monte_carlo(num_points, mu_star, var_star)
                numpy.random.seed(self.rng_seed)
                ei_naive = ei_eval._compute_expected_improvement_monte_carlo_naive(num_points, mu_star, var_star)
                self.assert_scalar_within_relative(ei_vectorized, ei_naive, ei_tolerance)

                # Compute quantities required for grad EI
                mu_star = ei_eval._gaussian_process.compute_mean_of_points(union_of_points)
                var_star = ei_eval._gaussian_process.compute_variance_of_points(union_of_points)
                grad_mu = ei_eval._gaussian_process.compute_grad_mean_of_points(union_of_points)
                grad_chol_decomp = ei_eval._gaussian_process.compute_grad_cholesky_variance_of_points(union_of_points, 0)

                # Check grad EI
                numpy.random.seed(self.rng_seed)
                grad_ei_vectorized = ei_eval._compute_grad_expected_improvement_monte_carlo(num_points, mu_star, var_star, grad_mu, grad_chol_decomp)
                numpy.random.seed(self.rng_seed)
                grad_ei_naive = ei_eval._compute_grad_expected_improvement_monte_carlo_naive(num_points, mu_star, var_star, grad_mu, grad_chol_decomp)
                self.assert_vector_within_relative(grad_ei_vectorized, grad_ei_naive, grad_ei_tolerance)

    def test_evaluate_ei_at_points(self):
        """Check that ``evaluate_expected_improvement_at_point_list`` computes and orders results correctly (using 1D analytic EI)."""
        index = numpy.argmax(numpy.greater_equal(self.num_sampled_list, 5))
        domain, _, gaussian_process = self.gp_test_environments[index]

        current_point = domain.generate_random_point_in_domain()
        ei_eval = ExpectedImprovement(gaussian_process, current_point)

        num_to_eval = 10
        points_to_evaluate = domain.generate_uniform_random_points_in_domain(num_to_eval)

        test_values = evaluate_expected_improvement_at_point_list(ei_eval, points_to_evaluate)

        for i, value in enumerate(test_values):
            ei_eval.set_current_point(points_to_evaluate[i, ...])
            truth = ei_eval.compute_expected_improvement()
            T.assert_equal(value, truth)

    def test_multistart_expected_improvement_optimization(self):
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

        current_point = domain.generate_random_point_in_domain()
        ei_eval = ExpectedImprovement(gaussian_process, current_point)

        # expand the domain so that we are definitely not doing constrained optimization
        expanded_domain = TensorProductDomain([ClosedInterval(-4.0, 2.0)] * self.dim)

        num_samples_to_generate = 1
        ei_optimizer = GradientDescentOptimizer(expanded_domain, ei_eval, gd_parameters)
        best_point = multistart_expected_improvement_optimization(ei_optimizer, num_multistarts, num_samples_to_generate)

        # Check that gradients are small
        ei_eval.set_current_point(best_point)
        gradient = ei_eval.compute_grad_expected_improvement()
        self.assert_vector_within_relative(gradient, numpy.zeros(self.num_hyperparameters), tolerance)

        # Check that output is in the domain
        T.assert_equal(expanded_domain.check_point_inside(best_point), True)


if __name__ == "__main__":
    T.run()
