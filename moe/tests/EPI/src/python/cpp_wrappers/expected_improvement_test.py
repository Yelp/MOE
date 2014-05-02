# -*- coding: utf-8 -*-
"""Test the C++ implementation of expected improvement against the Python implementation."""
import numpy
import testify as T

import moe.optimal_learning.EPI.src.python.cpp_wrappers.covariance as cpp_covariance
import moe.optimal_learning.EPI.src.python.cpp_wrappers.expected_improvement as cpp_expected_improvement
import moe.optimal_learning.EPI.src.python.cpp_wrappers.gaussian_process as cpp_gaussian_process
import moe.optimal_learning.EPI.src.python.python_version.covariance as python_covariance
import moe.optimal_learning.EPI.src.python.python_version.expected_improvement as python_expected_improvement
import moe.optimal_learning.EPI.src.python.python_version.gaussian_process as python_gaussian_process
from moe.optimal_learning.EPI.src.python.geometry_utils import ClosedInterval
from moe.optimal_learning.EPI.src.python.python_version.domain import TensorProductDomain
from moe.tests.EPI.src.python.gaussian_process_test_case import GaussianProcessTestCase, GaussianProcessTestEnvironmentInput


class ExpectedImprovementTest(GaussianProcessTestCase):

    """Test C++ vs Python implementations of Expected Improvement.

    Currently only checks that the 1D, analytic EI & gradient match.
    Checking monte carlo would be very expensive (b/c of the need to converge the MC) or very difficult
    (to make python & C++ use the exact same sequence of random numbers).

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
        hyperparameter_interval=ClosedInterval(0.1, 0.3),
        lower_bound_interval=ClosedInterval(-1.0, 0.5),
        upper_bound_interval=ClosedInterval(2.0, 3.5),
        covariance_class=python_covariance.SquareExponential,
        spatial_domain_class=TensorProductDomain,
        hyperparameter_domain_class=TensorProductDomain,
        gaussian_process_class=python_gaussian_process.GaussianProcess,
    )

    num_sampled_list = [1, 2, 5, 10, 16, 20, 42, 50]

    @T.class_setup
    def base_setup(self):
        """Run the standard setup but seed the RNG first (for repeatability).

        It is easy to stumble into test cases where EI is very small (e.g., < 1.e-20),
        which makes it difficult to set meaningful tolerances for the checks.

        """
        numpy.random.seed(8794)
        super(ExpectedImprovementTest, self).base_setup()

    def test_python_and_cpp_return_same_1D_analytic_ei_and_gradient(self):
        """Compare the 1D analytic EI/grad EI results from Python & C++, checking several random points per test case."""
        num_tests_per_case = 10
        ei_tolerance = 6.0e-14
        grad_ei_tolerance = 6.0e-14

        for test_case in self.gp_test_environments:
            domain, python_cov, python_gp = test_case
            current_point = domain.generate_random_point_in_domain()
            python_ei_eval = python_expected_improvement.ExpectedImprovement(python_gp, current_point)

            cpp_cov = cpp_covariance.SquareExponential(python_cov.get_hyperparameters())
            cpp_gp = cpp_gaussian_process.GaussianProcess(cpp_cov, python_gp._historical_data)
            cpp_ei_eval = cpp_expected_improvement.ExpectedImprovement(cpp_gp, current_point)

            for _ in xrange(num_tests_per_case):
                current_point = domain.generate_random_point_in_domain()
                cpp_ei_eval.set_current_point(current_point)
                python_ei_eval.set_current_point(current_point)

                cpp_ei = cpp_ei_eval.compute_expected_improvement()
                python_ei = python_ei_eval.compute_expected_improvement()
                self.assert_scalar_within_relative(python_ei, cpp_ei, ei_tolerance)

                cpp_grad_ei = cpp_ei_eval.compute_grad_expected_improvement()
                python_grad_ei = python_ei_eval.compute_grad_expected_improvement()
                self.assert_vector_within_relative(python_grad_ei, cpp_grad_ei, grad_ei_tolerance)


if __name__ == "__main__":
    T.run()
