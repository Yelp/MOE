# -*- coding: utf-8 -*-
"""Test cases for the Log Marginal Likelihood metric for model fit.

Testing is sparse at the moment. The C++ implementations are tested thoroughly (gpp_covariance_test.hpp/cpp) and
we rely more on cpp_wrappers/covariance_test.py's comparison with C++ for verification of the Python code.

"""
import numpy

import testify as T

from moe.optimal_learning.python.geometry_utils import ClosedInterval
from moe.optimal_learning.python.python_version.covariance import SquareExponential
from moe.optimal_learning.python.python_version.domain import TensorProductDomain
from moe.optimal_learning.python.python_version.gaussian_process import GaussianProcess
from moe.optimal_learning.python.python_version.log_likelihood import multistart_hyperparameter_optimization, evaluate_log_likelihood_at_hyperparameter_list, GaussianProcessLogMarginalLikelihood
from moe.optimal_learning.python.python_version.optimization import GradientDescentParameters, GradientDescentOptimizer
from moe.tests.optimal_learning.python.gaussian_process_test_case import GaussianProcessTestCase, GaussianProcessTestEnvironmentInput


class GaussianProcessLogMarginalLikelihoodTest(GaussianProcessTestCase):

    """Test cases for the Log Marginal Likelihood metric for model fit.

    Tests check that the gradients ping properly and that computed log likelihood values are < 0.0.

    """

    precompute_gaussian_process_data = False

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

    num_sampled_list = (1, 2, 5, 10, 16, 20, 42)

    def test_grad_log_likelihood_pings(self):
        """Ping test (compare analytic result to finite difference) the log likelihood gradient wrt hyperparameters."""
        h = 2.0e-4
        tolerance = 5.0e-6

        for num_sampled in self.num_sampled_list:
            self.gp_test_environment_input.num_sampled = num_sampled
            _, covariance, gaussian_process = self._build_gaussian_process_test_data(self.gp_test_environment_input)
            lml = GaussianProcessLogMarginalLikelihood(covariance, gaussian_process._historical_data)

            analytic_grad = lml.compute_grad_log_likelihood()
            for k in xrange(lml.num_hyperparameters):
                hyperparameters_old = lml.hyperparameters

                # hyperparamter + h
                hyperparameters_p = numpy.copy(hyperparameters_old)
                hyperparameters_p[k] += h
                lml.hyperparameters = (hyperparameters_p)
                cov_p = lml.compute_log_likelihood()
                lml.hyperparameters = (hyperparameters_old)

                # hyperparamter - h
                hyperparameters_m = numpy.copy(hyperparameters_old)
                hyperparameters_m[k] -= h
                lml.hyperparameters = (hyperparameters_m)
                cov_m = lml.compute_log_likelihood()
                lml.hyperparameters = (hyperparameters_old)

                # calculate finite diff
                fd_grad = (cov_p - cov_m) / (2.0 * h)

                self.assert_scalar_within_relative(fd_grad, analytic_grad[k], tolerance)

    def test_evaluate_log_likelihood_at_points(self):
        """Check that ``evaluate_log_likelihood_at_hyperparameter_list`` computes and orders results correctly."""
        num_sampled = 5

        self.gp_test_environment_input.num_sampled = num_sampled
        _, covariance, gaussian_process = self._build_gaussian_process_test_data(self.gp_test_environment_input)
        lml = GaussianProcessLogMarginalLikelihood(covariance, gaussian_process._historical_data)

        num_to_eval = 10
        domain_bounds = [self.gp_test_environment_input.hyperparameter_interval] * self.gp_test_environment_input.num_hyperparameters
        domain = TensorProductDomain(domain_bounds)
        hyperparameters_to_evaluate = domain.generate_uniform_random_points_in_domain(num_to_eval)

        test_values = evaluate_log_likelihood_at_hyperparameter_list(lml, hyperparameters_to_evaluate)

        for i, value in enumerate(test_values):
            lml.hyperparameters = (hyperparameters_to_evaluate[i, ...])
            truth = lml.compute_log_likelihood()
            T.assert_equal(value, truth)

    def test_multistart_hyperparameter_optimization(self):
        """Check that multistart optimization (gradient descent) can find the optimum hyperparameters."""
        random_state = numpy.random.get_state()
        numpy.random.seed(87612)

        max_num_steps = 200  # this is generally *too few* steps; we configure it this way so the test will run quickly
        max_num_restarts = 5
        num_steps_averaged = 0
        gamma = 0.2
        pre_mult = 1.0
        max_relative_change = 0.3
        tolerance = 1.0e-11
        gd_parameters = GradientDescentParameters(
            max_num_steps,
            max_num_restarts,
            num_steps_averaged,
            gamma,
            pre_mult,
            max_relative_change,
            tolerance,
        )
        num_multistarts = 3  # again, too few multistarts; but we want the test to run reasonably quickly

        num_sampled = 10
        self.gp_test_environment_input.num_sampled = num_sampled
        _, covariance, gaussian_process = self._build_gaussian_process_test_data(self.gp_test_environment_input)
        lml = GaussianProcessLogMarginalLikelihood(covariance, gaussian_process._historical_data)

        domain = TensorProductDomain([ClosedInterval(1.0, 4.0)] * self.gp_test_environment_input.num_hyperparameters)

        hyperparameter_optimizer = GradientDescentOptimizer(domain, lml, gd_parameters)
        best_hyperparameters = multistart_hyperparameter_optimization(hyperparameter_optimizer, num_multistarts)

        # Check that gradients are small
        lml.hyperparameters = (best_hyperparameters)
        gradient = lml.compute_grad_log_likelihood()
        self.assert_vector_within_relative(gradient, numpy.zeros(self.num_hyperparameters), tolerance)

        # Check that output is in the domain
        T.assert_equal(domain.check_point_inside(best_hyperparameters), True)

        numpy.random.set_state(random_state)


if __name__ == "__main__":
    T.run()
