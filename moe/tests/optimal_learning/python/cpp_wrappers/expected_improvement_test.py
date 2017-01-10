# -*- coding: utf-8 -*-
"""Test the C++ implementation of expected improvement against the Python implementation."""
import numpy

import pytest

import moe.optimal_learning.python.cpp_wrappers.covariance
import moe.optimal_learning.python.cpp_wrappers.expected_improvement
import moe.optimal_learning.python.cpp_wrappers.gaussian_process
from moe.optimal_learning.python.geometry_utils import ClosedInterval
import moe.optimal_learning.python.python_version.covariance
import moe.optimal_learning.python.python_version.domain
import moe.optimal_learning.python.python_version.expected_improvement
import moe.optimal_learning.python.python_version.gaussian_process
from moe.tests.optimal_learning.python.gaussian_process_test_case import GaussianProcessTestCase, GaussianProcessTestEnvironmentInput


class TestExpectedImprovement(GaussianProcessTestCase):

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
        covariance_class=moe.optimal_learning.python.python_version.covariance.SquareExponential,
        spatial_domain_class=moe.optimal_learning.python.python_version.domain.TensorProductDomain,
        hyperparameter_domain_class=moe.optimal_learning.python.python_version.domain.TensorProductDomain,
        gaussian_process_class=moe.optimal_learning.python.python_version.gaussian_process.GaussianProcess,
    )

    num_sampled_list = (1, 2, 5, 10, 16, 20, 42, 50)

    @classmethod
    @pytest.fixture(autouse=True, scope='class')
    def base_setup(cls):
        """Run the standard setup but seed the RNG first (for repeatability).

        It is easy to stumble into test cases where EI is very small (e.g., < 1.e-20),
        which makes it difficult to set meaningful tolerances for the checks.

        """
        numpy.random.seed(8794)
        super(TestExpectedImprovement, cls).base_setup()

    def test_python_and_cpp_return_same_1d_analytic_ei_and_gradient(self):
        """Compare the 1D analytic EI/grad EI results from Python & C++, checking several random points per test case."""
        num_tests_per_case = 10
        ei_tolerance = 6.0e-14
        # TODO(GH-240): set RNG seed for this case and restore toleranace to 6.0e-14 or better
        grad_ei_tolerance = 6.0e-13

        for test_case in self.gp_test_environments:
            domain, python_gp = test_case
            python_cov, historical_data = python_gp.get_core_data_copy()

            points_to_sample = domain.generate_random_point_in_domain()
            python_ei_eval = moe.optimal_learning.python.python_version.expected_improvement.ExpectedImprovement(python_gp, points_to_sample)

            cpp_cov = moe.optimal_learning.python.cpp_wrappers.covariance.SquareExponential(python_cov.hyperparameters)
            cpp_gp = moe.optimal_learning.python.cpp_wrappers.gaussian_process.GaussianProcess(cpp_cov, historical_data)
            cpp_ei_eval = moe.optimal_learning.python.cpp_wrappers.expected_improvement.ExpectedImprovement(cpp_gp, points_to_sample)

            for _ in range(num_tests_per_case):
                points_to_sample = domain.generate_random_point_in_domain()
                cpp_ei_eval.current_point = points_to_sample
                python_ei_eval.current_point = points_to_sample

                cpp_ei = cpp_ei_eval.compute_expected_improvement()
                python_ei = python_ei_eval.compute_expected_improvement(force_1d_ei=True)
                self.assert_scalar_within_relative(python_ei, cpp_ei, ei_tolerance)

                cpp_grad_ei = cpp_ei_eval.compute_grad_expected_improvement()
                python_grad_ei = python_ei_eval.compute_grad_expected_improvement()
                self.assert_vector_within_relative(python_grad_ei, cpp_grad_ei, grad_ei_tolerance)
