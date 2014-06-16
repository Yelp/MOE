# -*- coding: utf-8 -*-
"""Test the C++ implementation of Gaussian Process properties (mean, var, gradients thereof) against the Python version."""
import copy

import numpy

import testify as T

import moe.build.GPP as C_GP
from moe.optimal_learning.python.cpp_wrappers.covariance import SquareExponential
from moe.optimal_learning.python.cpp_wrappers.gaussian_process import GaussianProcess
from moe.optimal_learning.python.data_containers import HistoricalData, SamplePoint
from moe.tests.optimal_learning.python.gaussian_process_test_case import GaussianProcessTestCase


class GaussianProcessTest(GaussianProcessTestCase):

    """Test C++ vs Python implementations of Gaussian Process properties (mean, variance, cholesky variance, and their gradients)."""

    precompute_gaussian_process_data = True

    @T.class_setup
    def base_setup(self):
        """Run the standard setup but seed the RNG first (for repeatability).

        It is easy to stumble into test cases where mean, var terms are very small (e.g., < 1.e-20),
        which makes it difficult to set meaningful tolerances for the checks.

        """
        numpy.random.seed(8794)
        super(GaussianProcessTest, self).base_setup()

    def test_gp_construction_singular_covariance_matrix(self):
        """Test that the GaussianProcess ctor indicates a singular covariance matrix on bad input."""
        index = numpy.argmax(numpy.greater_equal(self.num_sampled_list, 1))
        domain, covariance, _ = self.gp_test_environments[index]
        point_one = SamplePoint([0.0] * domain.dim, 1.0, 0.0)
        point_two = SamplePoint([1.0] * domain.dim, 1.0, 0.0)
        point_three = point_two

        historical_data = HistoricalData(len(point_one.point), [point_one, point_two, point_three])
        T.assert_raises(C_GP.SingularMatrixException, GaussianProcess, covariance, historical_data)

    def test_gp_add_sampled_points_singular_covariance_matrix(self):
        """Test that GaussianProcess.add_sampled_points indicates a singular covariance matrix on bad input."""
        test_environment_input = copy.copy(self.gp_test_environment_input)
        test_environment_input.num_sampled = 1
        test_environment_input.gaussian_process_class = GaussianProcess
        _, _, gaussian_process = self._build_gaussian_process_test_data(test_environment_input)
        point_one = SamplePoint([0.5] * gaussian_process.dim, 1.0, 0.0)
        point_two = SamplePoint([1.0] * gaussian_process.dim, -1.0, 0.0)
        point_three = point_one

        # points one and two are different, so this is safe
        gaussian_process.add_sampled_points([point_one, point_two])
        # point_three is identical to point_one; this will produce a singular covariance matrix
        T.assert_raises(C_GP.SingularMatrixException, gaussian_process.add_sampled_points, [point_three])

    def test_python_and_cpp_return_same_mu_and_gradient(self):
        """Compare mu/grad mu results from Python & C++, checking seeral random points per test case."""
        num_tests_per_case = 4
        mu_tolerance = 3.0e-13
        grad_mu_tolerance = 3.0e-12

        for test_case in self.gp_test_environments:
            domain, python_cov, python_gp = test_case

            cpp_cov = SquareExponential(python_cov.get_hyperparameters())
            cpp_gp = GaussianProcess(cpp_cov, python_gp._historical_data)

            for num_to_sample in self.num_to_sample_list:
                for _ in xrange(num_tests_per_case):
                    points_to_sample = domain.generate_uniform_random_points_in_domain(num_to_sample)

                    cpp_mu = cpp_gp.compute_mean_of_points(points_to_sample)
                    python_mu = python_gp.compute_mean_of_points(points_to_sample)
                    self.assert_vector_within_relative(python_mu, cpp_mu, mu_tolerance)

                    cpp_grad_mu = cpp_gp.compute_grad_mean_of_points(points_to_sample)
                    python_grad_mu = python_gp.compute_grad_mean_of_points(points_to_sample)
                    self.assert_vector_within_relative(python_grad_mu, cpp_grad_mu, grad_mu_tolerance)

    def test_python_and_cpp_return_same_variance_and_gradient(self):
        """Compare var/grad var results from Python & C++, checking seeral random points per test case."""
        num_tests_per_case = 2
        var_tolerance = 3.0e-13
        grad_var_tolerance = 3.0e-12

        for test_case in self.gp_test_environments:
            domain, python_cov, python_gp = test_case

            cpp_cov = SquareExponential(python_cov.get_hyperparameters())
            cpp_gp = GaussianProcess(cpp_cov, python_gp._historical_data)

            for num_to_sample in self.num_to_sample_list:
                for _ in xrange(num_tests_per_case):
                    points_to_sample = domain.generate_uniform_random_points_in_domain(num_to_sample)

                    cpp_var = cpp_gp.compute_variance_of_points(points_to_sample)
                    python_var = python_gp.compute_variance_of_points(points_to_sample)
                    self.assert_vector_within_relative(python_var, cpp_var, var_tolerance)

                    cpp_grad_var = cpp_gp.compute_grad_variance_of_points(points_to_sample)
                    python_grad_var = python_gp.compute_grad_variance_of_points(points_to_sample)
                    self.assert_vector_within_relative(python_grad_var, cpp_grad_var, grad_var_tolerance)

    def test_python_and_cpp_return_same_cholesky_variance_and_gradient(self):
        """Compare chol_var/grad chol_var results from Python & C++, checking seeral random points per test case."""
        num_tests_per_case = 2
        var_tolerance = 3.0e-12
        grad_var_tolerance = 3.0e-12

        for test_case in self.gp_test_environments:
            domain, python_cov, python_gp = test_case

            cpp_cov = SquareExponential(python_cov.get_hyperparameters())
            cpp_gp = GaussianProcess(cpp_cov, python_gp._historical_data)

            for num_to_sample in self.num_to_sample_list:
                for _ in xrange(num_tests_per_case):
                    points_to_sample = domain.generate_uniform_random_points_in_domain(num_to_sample)

                    cpp_var = cpp_gp.compute_cholesky_variance_of_points(points_to_sample)
                    python_var = python_gp.compute_cholesky_variance_of_points(points_to_sample)
                    self.assert_vector_within_relative(python_var, cpp_var, var_tolerance)

                    cpp_grad_var = cpp_gp.compute_grad_cholesky_variance_of_points(points_to_sample)
                    python_grad_var = python_gp.compute_grad_cholesky_variance_of_points(points_to_sample)
                    self.assert_vector_within_relative(python_grad_var, cpp_grad_var, grad_var_tolerance)


if __name__ == "__main__":
    T.run()
