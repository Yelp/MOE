# -*- coding: utf-8 -*-
"""Test the C++ implementation of Gaussian Process properties (mean, var, gradients thereof) against the Python version."""
import numpy

import testify as T

from moe.optimal_learning.python import cpp_wrappers
from moe.optimal_learning.python import python_version
import moe.optimal_learning.python.cpp_wrappers.covariance
import moe.optimal_learning.python.cpp_wrappers.gaussian_process
from moe.optimal_learning.python.geometry_utils import ClosedInterval
import moe.optimal_learning.python.python_version.covariance
from moe.optimal_learning.python.python_version.domain import TensorProductDomain
import moe.optimal_learning.python.python_version.gaussian_process
from moe.tests.optimal_learning.python.gaussian_process_test_case import GaussianProcessTestCase, GaussianProcessTestEnvironmentInput


class GaussianProcessTest(GaussianProcessTestCase):

    """Test C++ vs Python implementations of Gaussian Process properties (mean, variance, cholesky variance, and their gradients).

    TODO(eliu): check several points_to_sample per computation
    TODO(eliu): check grad var & grad chol var with each possible value ovar_of_grad

    """

    precompute_gaussian_process_data = True

    noise_variance_base = 0.0002
    dim = 3
    num_hyperparameters = dim + 1

    gp_test_environment_input = GaussianProcessTestEnvironmentInput(
        dim,
        num_hyperparameters,
        0,
        noise_variance_base=noise_variance_base,
        hyperparameter_interval=ClosedInterval(0.1, 1.3),
        lower_bound_interval=ClosedInterval(-2.0, 0.5),
        upper_bound_interval=ClosedInterval(2.0, 3.5),
        covariance_class=python_version.covariance.SquareExponential,
        spatial_domain_class=TensorProductDomain,
        hyperparameter_domain_class=TensorProductDomain,
        gaussian_process_class=python_version.gaussian_process.GaussianProcess,
    )

    num_sampled_list = [1, 2, 3, 5, 10, 16, 20, 42]
    num_to_sample_list = [1, 2, 3, 8]

    @T.class_setup
    def base_setup(self):
        """Run the standard setup but seed the RNG first (for repeatability).

        It is easy to stumble into test cases where mean, var terms are very small (e.g., < 1.e-20),
        which makes it difficult to set meaningful tolerances for the checks.

        """
        numpy.random.seed(8794)
        super(GaussianProcessTest, self).base_setup()

    def test_python_and_cpp_return_same_mu_and_gradient(self):
        """Compare mu/grad mu results from Python & C++, checking seeral random points per test case."""
        num_tests_per_case = 4
        mu_tolerance = 3.0e-13
        grad_mu_tolerance = 3.0e-12

        for test_case in self.gp_test_environments:
            domain, python_cov, python_gp = test_case

            cpp_cov = cpp_wrappers.covariance.SquareExponential(python_cov.get_hyperparameters())
            cpp_gp = cpp_wrappers.gaussian_process.GaussianProcess(cpp_cov, python_gp._historical_data)

            for num_to_sample in self.num_to_sample_list:
                for _ in xrange(num_tests_per_case):
                    current_points = domain.generate_uniform_random_points_in_domain(num_to_sample)

                    cpp_mu = cpp_gp.compute_mean_of_points(current_points)
                    python_mu = python_gp.compute_mean_of_points(current_points)
                    self.assert_vector_within_relative(python_mu, cpp_mu, mu_tolerance)

                    cpp_grad_mu = cpp_gp.compute_grad_mean_of_points(current_points)
                    python_grad_mu = python_gp.compute_grad_mean_of_points(current_points)
                    self.assert_vector_within_relative(python_grad_mu, cpp_grad_mu, grad_mu_tolerance)

    def test_python_and_cpp_return_same_variance_and_gradient(self):
        """Compare var/grad var results from Python & C++, checking seeral random points per test case."""
        num_tests_per_case = 2
        var_tolerance = 3.0e-13
        grad_var_tolerance = 3.0e-12

        for test_case in self.gp_test_environments:
            domain, python_cov, python_gp = test_case

            cpp_cov = cpp_wrappers.covariance.SquareExponential(python_cov.get_hyperparameters())
            cpp_gp = cpp_wrappers.gaussian_process.GaussianProcess(cpp_cov, python_gp._historical_data)

            for num_to_sample in self.num_to_sample_list:
                for _ in xrange(num_tests_per_case):
                    current_points = domain.generate_uniform_random_points_in_domain(num_to_sample)

                    cpp_var = cpp_gp.compute_variance_of_points(current_points)
                    python_var = python_gp.compute_variance_of_points(current_points)
                    self.assert_vector_within_relative(python_var, cpp_var, var_tolerance)

                    for i in xrange(num_to_sample):
                        cpp_grad_var = cpp_gp.compute_grad_variance_of_points(current_points, i)
                        python_grad_var = python_gp.compute_grad_variance_of_points(current_points, i)
                        self.assert_vector_within_relative(python_grad_var, cpp_grad_var, grad_var_tolerance)

    def test_python_and_cpp_return_same_cholesky_variance_and_gradient(self):
        """Compare chol_var/grad chol_var results from Python & C++, checking seeral random points per test case."""
        num_tests_per_case = 2
        var_tolerance = 3.0e-12
        grad_var_tolerance = 3.0e-12

        for test_case in self.gp_test_environments:
            domain, python_cov, python_gp = test_case

            cpp_cov = cpp_wrappers.covariance.SquareExponential(python_cov.get_hyperparameters())
            cpp_gp = cpp_wrappers.gaussian_process.GaussianProcess(cpp_cov, python_gp._historical_data)

            for num_to_sample in self.num_to_sample_list:
                for _ in xrange(num_tests_per_case):
                    current_points = domain.generate_uniform_random_points_in_domain(num_to_sample)

                    cpp_var = cpp_gp.compute_cholesky_variance_of_points(current_points)
                    python_var = python_gp.compute_cholesky_variance_of_points(current_points)
                    self.assert_vector_within_relative(python_var, cpp_var, var_tolerance)

                    for i in xrange(num_to_sample):
                        cpp_grad_var = cpp_gp.compute_grad_cholesky_variance_of_points(current_points, i)
                        python_grad_var = python_gp.compute_grad_cholesky_variance_of_points(current_points, i)
                        self.assert_vector_within_relative(python_grad_var, cpp_grad_var, grad_var_tolerance)


if __name__ == "__main__":
    T.run()
