# -*- coding: utf-8 -*-
"""Test the C++ implementation of expected improvement against the Python implementation."""
import numpy

import testify as T

import moe.optimal_learning.python.cpp_wrappers.covariance
import moe.optimal_learning.python.cpp_wrappers.expected_improvement
import moe.optimal_learning.python.cpp_wrappers.gaussian_process
from moe.optimal_learning.python.geometry_utils import ClosedInterval
import moe.optimal_learning.python.python_version.covariance
import moe.optimal_learning.python.python_version.domain
import moe.optimal_learning.python.python_version.expected_improvement
import moe.optimal_learning.python.python_version.gaussian_process
from moe.tests.optimal_learning.python.gaussian_process_test_case import GaussianProcessTestCase, GaussianProcessTestEnvironmentInput


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
        covariance_class=moe.optimal_learning.python.python_version.covariance.SquareExponential,
        spatial_domain_class=moe.optimal_learning.python.python_version.domain.TensorProductDomain,
        hyperparameter_domain_class=moe.optimal_learning.python.python_version.domain.TensorProductDomain,
        gaussian_process_class=moe.optimal_learning.python.python_version.gaussian_process.GaussianProcess,
    )

    num_sampled_list = (1, 2, 5, 10, 16, 20, 42, 50)

    @T.class_setup
    def base_setup(self):
        """Run the standard setup but seed the RNG first (for repeatability).

        It is easy to stumble into test cases where EI is very small (e.g., < 1.e-20),
        which makes it difficult to set meaningful tolerances for the checks.

        """
        numpy.random.seed(8794)
        super(ExpectedImprovementTest, self).base_setup()

    def test_python_and_cpp_return_same_1d_analytic_ei_and_gradient(self):
        """Compare the 1D analytic EI/grad EI results from Python & C++, checking several random points per test case."""
        num_tests_per_case = 10
        ei_tolerance = 6.0e-14
        # TODO(GH-240): set RNG seed for this case and restore toleranace to 6.0e-14 or better
        grad_ei_tolerance = 6.0e-13

        for test_case in self.gp_test_environments:
            domain, python_gp = test_case
            points_to_sample = domain.generate_random_point_in_domain()
            python_ei_eval = moe.optimal_learning.python.python_version.expected_improvement.ExpectedImprovement(python_gp, points_to_sample)

            cpp_cov = moe.optimal_learning.python.cpp_wrappers.covariance.SquareExponential(python_gp._covariance.hyperparameters)
            cpp_gp = moe.optimal_learning.python.cpp_wrappers.gaussian_process.GaussianProcess(cpp_cov, python_gp._historical_data)
            cpp_ei_eval = moe.optimal_learning.python.cpp_wrappers.expected_improvement.ExpectedImprovement(cpp_gp, points_to_sample)

            for _ in xrange(num_tests_per_case):
                points_to_sample = domain.generate_random_point_in_domain()
                cpp_ei_eval.current_point = points_to_sample
                python_ei_eval.current_point = points_to_sample

                cpp_ei = cpp_ei_eval.compute_expected_improvement()
                python_ei = python_ei_eval.compute_expected_improvement(force_1d_ei=True)
                self.assert_scalar_within_relative(python_ei, cpp_ei, ei_tolerance)

                cpp_grad_ei = cpp_ei_eval.compute_grad_expected_improvement()
                python_grad_ei = python_ei_eval.compute_grad_expected_improvement()
                self.assert_vector_within_relative(python_grad_ei, cpp_grad_ei, grad_ei_tolerance)

    def test_qd_and_1d_return_same_analytic_ei(self):
        """Compare the 1D analytic EI results to the qD analytic EI results, checking several random points per test case."""
        num_tests_per_case = 10
        ei_tolerance = 6.0e-14

        for test_case in self.gp_test_environments:
            domain, python_gp = test_case
            points_to_sample = domain.generate_random_point_in_domain()
            python_ei_eval = moe.optimal_learning.python.python_version.expected_improvement.ExpectedImprovement(python_gp, points_to_sample)

            for _ in xrange(num_tests_per_case):
                points_to_sample = domain.generate_random_point_in_domain()
                python_ei_eval.set_current_point(points_to_sample)

                python_1d_ei = python_ei_eval.compute_expected_improvement(force_1d_ei=True)
                python_qd_ei = python_ei_eval.compute_expected_improvement()
                self.assert_scalar_within_relative(python_1d_ei, python_qd_ei, ei_tolerance)

    def test_qd_ei_with_self(self):
        """Compare the 1D analytic EI results to the qD analytic EI results, checking several random points per test case.

        This test case (unfortunately) suffers from a lot of random variation in the qEI parameters. The tolerance is high because
        changing the number of iterations or the maximum relative error allowed in the mvndst function leads to different answers.

        These precomputed answers were calculated from:
        maxpts = 20,000 * q
        releps = 1e-5

        These values are a tradeoff between accuracy / speed.
        """
        ei_tolerance = 6.0e-8
        numpy.random.seed(8790)

        precomputed_answers = [
            0.0447239829581,
            0.0844060926313,
            0.119761506674,
            0.151392981174,
            0.179810984207,
            0.20532396992,
            0.228557714942,
            0.249686586053,
            0.268972737603,
        ]

        for test_case in self.gp_test_environments[:1]:
            domain, python_gp = test_case
            all_points = domain.generate_uniform_random_points_in_domain(14)

            for i in range(1, 10):
                points_to_sample = all_points[0:i]
                python_ei_eval = moe.optimal_learning.python.python_version.expected_improvement.ExpectedImprovement(python_gp, points_to_sample, num_mc_iterations=10000000)

                python_ei_eval.current_point = points_to_sample
                python_qd_ei = python_ei_eval.compute_expected_improvement()
                self.assert_scalar_within_relative(python_qd_ei, precomputed_answers[i - 1], ei_tolerance)


if __name__ == "__main__":
    T.run()
