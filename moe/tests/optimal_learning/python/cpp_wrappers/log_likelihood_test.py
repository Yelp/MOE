# -*- coding: utf-8 -*-
"""Test cases to check that C++ and Python implementations of interfaces/log_likelihood.py match."""
import testify as T

import moe.optimal_learning.python.cpp_wrappers.covariance
import moe.optimal_learning.python.cpp_wrappers.log_likelihood
from moe.optimal_learning.python.geometry_utils import ClosedInterval
import moe.optimal_learning.python.python_version.covariance
import moe.optimal_learning.python.python_version.domain
import moe.optimal_learning.python.python_version.log_likelihood
from moe.tests.optimal_learning.python.gaussian_process_test_case import GaussianProcessTestCase, GaussianProcessTestEnvironmentInput


class LogLikelihoodTest(GaussianProcessTestCase):

    """Test that the C++ and Python implementations of the Log Marginal Likelihood match (value and gradient)."""

    precompute_gaussian_process_data = False

    noise_variance_base = 0.0002
    dim = 3
    num_hyperparameters = dim + 1

    gp_test_environment_input = GaussianProcessTestEnvironmentInput(
        dim,
        num_hyperparameters,
        0,
        noise_variance_base=noise_variance_base,
        hyperparameter_interval=ClosedInterval(0.2, 1.5),
        lower_bound_interval=ClosedInterval(-2.0, 0.5),
        upper_bound_interval=ClosedInterval(2.0, 3.5),
        covariance_class=moe.optimal_learning.python.python_version.covariance.SquareExponential,
        spatial_domain_class=moe.optimal_learning.python.python_version.domain.TensorProductDomain,
        hyperparameter_domain_class=moe.optimal_learning.python.python_version.domain.TensorProductDomain,
    )

    num_sampled_list = (1, 2, 5, 10, 16, 20, 42)

    def test_python_and_cpp_return_same_log_likelihood_and_gradient(self):
        """Check that the C++ and Python log likelihood + gradients match over a series of randomly built data sets."""
        tolerance_log_like = 5.0e-11
        tolerance_grad_log_like = 4.0e-12

        for num_sampled in self.num_sampled_list:
            self.gp_test_environment_input.num_sampled = num_sampled
            _, python_gp = self._build_gaussian_process_test_data(self.gp_test_environment_input)
            python_lml = moe.optimal_learning.python.python_version.log_likelihood.GaussianProcessLogMarginalLikelihood(python_gp._covariance, python_gp._historical_data)
            cpp_cov = moe.optimal_learning.python.cpp_wrappers.covariance.SquareExponential(python_gp._covariance.hyperparameters)
            cpp_lml = moe.optimal_learning.python.cpp_wrappers.log_likelihood.GaussianProcessLogMarginalLikelihood(cpp_cov, python_gp._historical_data)

            python_log_like = python_lml.compute_log_likelihood()
            cpp_log_like = cpp_lml.compute_log_likelihood()
            self.assert_scalar_within_relative(python_log_like, cpp_log_like, tolerance_log_like)

            python_grad_log_like = python_lml.compute_grad_log_likelihood()
            cpp_grad_log_like = cpp_lml.compute_grad_log_likelihood()
            self.assert_vector_within_relative(python_grad_log_like, cpp_grad_log_like, tolerance_grad_log_like)


if __name__ == "__main__":
    T.run()
