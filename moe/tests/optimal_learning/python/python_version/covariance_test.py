# -*- coding: utf-8 -*-
"""Test cases for the Square Exponential covariance function and its spatial gradient.

Testing is sparse at the moment. The C++ implementations are tested thoroughly (gpp_covariance_test.hpp/cpp) and
we rely more on cpp_wrappers/covariance_test.py's comparison with C++ for verification of the Python code.
x
TODO(eliu): test hyperparameter gradient
TODO(eliu): ping testing for spatial gradients
TODO(eliu): make test structure general enough to support other covariance functions automatically

"""
import numpy

import testify as T

import moe.optimal_learning.python.gaussian_process_test_utils as gp_utils
from moe.optimal_learning.python.geometry_utils import ClosedInterval
from moe.optimal_learning.python.python_version.covariance import SquareExponential
from moe.optimal_learning.python.python_version.domain import TensorProductDomain
from moe.tests.optimal_learning.python.optimal_learning_test_case import OptimalLearningTestCase


class SquareExponentialTest(OptimalLearningTestCase):

    """Tests for the computation of the SquareExponential covariance and spatial gradient of covariance.

    Tests cases are against manually verified results in various spatial dimensions and some ping tests.

    """

    @T.class_setup
    def base_setup(self):
        """Set up parameters for test cases."""
        self.epsilon = 2.0 * numpy.finfo('float64').eps
        self.CovarianceClass = SquareExponential

        self.one_dim_test_sets = numpy.array([
            [1.0, 0.1],
            [2.0, 0.1],
            [1.0, 1.0],
            [0.1, 10.0],
            [1.0, 1.0],
            [0.1, 10.0],
        ])

        self.three_dim_test_sets = numpy.array([
            [1.0, 0.1, 0.1, 0.1],
            [1.0, 0.1, 0.2, 0.1],
            [1.0, 0.1, 0.2, 0.3],
            [2.0, 0.1, 0.1, 0.1],
            [2.0, 0.1, 0.2, 0.1],
            [2.0, 0.1, 0.2, 0.3],
            [0.1, 10.0, 1.0, 0.1],
            [1.0, 10.0, 1.0, 0.1],
            [10.0, 10.0, 1.0, 0.1],
            [0.1, 10.0, 1.0, 0.1],
            [1.0, 10.0, 1.0, 0.1],
            [10.0, 10.0, 1.0, 0.1],
        ])

    def test_square_exponential_covariance_one_dim(self):
        """Test the SquareExponential covariance function against correct values for different sets of hyperparameters in 1D."""
        for hyperparameters in self.one_dim_test_sets:
            signal_variance = hyperparameters[0]
            length = hyperparameters[1]
            covariance = self.CovarianceClass(hyperparameters)

            # One length away
            truth = signal_variance * numpy.exp(-0.5)
            self.assert_scalar_within_relative(
                covariance.covariance(numpy.array([0.0]), numpy.array(length)),
                truth,
                self.epsilon,
            )
            # Sym
            self.assert_scalar_within_relative(
                covariance.covariance(numpy.array(length), numpy.array([0.0])),
                truth,
                self.epsilon,
            )

            # One length * sqrt 2 away
            truth = signal_variance * numpy.exp(-1.0)
            self.assert_scalar_within_relative(
                covariance.covariance(numpy.array([0.0]), numpy.array([length * numpy.sqrt(2)])),
                truth,
                self.epsilon,
            )

    def test_square_exponential_covariance_three_dim(self):
        """Test the SquareExponential covariance function against correct values for different sets of hyperparameters in 3D."""
        for hyperparameters in self.three_dim_test_sets:
            signal_variance = hyperparameters[0]
            length = hyperparameters[1:]
            covariance = self.CovarianceClass(hyperparameters)

            self.assert_scalar_within_relative(
                covariance.covariance(numpy.array([0.0, 0.0, 0.0]), numpy.array([0.0, 0.0, length[2]])),
                signal_variance * numpy.exp(-0.5),
                self.epsilon,
            )
            self.assert_scalar_within_relative(
                covariance.covariance(numpy.array([0.0, 0.0, 0.0]), numpy.array([0.0, length[1], 0.0])),
                signal_variance * numpy.exp(-0.5),
                self.epsilon,
            )
            self.assert_scalar_within_relative(
                covariance.covariance(numpy.array([0.0, 0.0, 0.0]), numpy.array([length[0], 0.0, 0.0])),
                signal_variance * numpy.exp(-0.5),
                self.epsilon,
            )

            self.assert_scalar_within_relative(
                covariance.covariance(numpy.array([0.0, 0.0, 0.0]), numpy.array([numpy.sqrt(3) / 3.0 * length[0], numpy.sqrt(3) / 3.0 * length[1], numpy.sqrt(3) / 3.0 * length[2]])),
                signal_variance * numpy.exp(-0.5),
                self.epsilon,
            )
            # Sym
            self.assert_scalar_within_relative(
                covariance.covariance(numpy.array([numpy.sqrt(3) / 3.0 * length[0], numpy.sqrt(3) / 3.0 * length[1], numpy.sqrt(3) / 3.0 * length[2]]), numpy.array([0.0, 0.0, 0.0])),
                signal_variance * numpy.exp(-0.5),
                self.epsilon,
            )

    def test_square_exponential_grad_covariance_three_dim(self):
        """Test the SquareExponential grad_covariance function against correct values for different sets of hyperparameters in 3D."""
        for hyperparameters in self.three_dim_test_sets:
            length = hyperparameters[1:]
            covariance = self.CovarianceClass(hyperparameters)

            # Same point
            truth = numpy.array([0.0, 0.0, 0.0])
            grad_cov = covariance.grad_covariance(numpy.array([0.0, 0.0, 0.0]), numpy.array([0.0, 0.0, 0.0]))
            self.assert_vector_within_relative(grad_cov, truth, 0.0)

            # One length away
            truth1 = numpy.array([0.0, 0.0, 1.0 / length[2] * covariance.covariance(numpy.array([0.0, 0.0, 0.0]), numpy.array([0.0, 0.0, length[2]]))])
            grad_cov1 = covariance.grad_covariance(numpy.array([0.0, 0.0, 0.0]), numpy.array([0.0, 0.0, length[2]]))
            self.assert_vector_within_relative(grad_cov1, truth1, self.epsilon)

            # Sym is opposite
            truth2 = truth1.copy()
            truth2[2] *= -1.0
            grad_cov2 = covariance.grad_covariance(numpy.array([0.0, 0.0, length[2]]), numpy.array([0.0, 0.0, 0.0]))
            self.assert_vector_within_relative(grad_cov2, truth2, self.epsilon)

            T.assert_equal(grad_cov1[2], -grad_cov2[2])

    def test_hyperparameter_gradient_pings(self):
        """Ping test (compare analytic result to finite difference) the gradient wrt hyperparameters."""
        h = 2.0e-3
        tolerance = 1.0e-5
        num_tests = 10

        dim = 3
        num_hyperparameters = dim + 1
        hyperparameter_interval = ClosedInterval(3.0, 5.0)

        domain = TensorProductDomain(ClosedInterval.build_closed_intervals_from_list([[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]]))

        points1 = domain.generate_uniform_random_points_in_domain(num_tests)
        points2 = domain.generate_uniform_random_points_in_domain(num_tests)

        for i in xrange(num_tests):
            point_one = points1[i, ...]
            point_two = points2[i, ...]

            covariance = gp_utils.fill_random_covariance_hyperparameters(
                hyperparameter_interval,
                num_hyperparameters,
                covariance_type=self.CovarianceClass,
            )

            analytic_grad = covariance.hyperparameter_grad_covariance(point_one, point_two)
            for k in xrange(covariance.num_hyperparameters):
                hyperparameters_old = covariance.get_hyperparameters()

                # hyperparamter + h
                hyperparameters_p = numpy.copy(hyperparameters_old)
                hyperparameters_p[k] += h
                covariance.set_hyperparameters(hyperparameters_p)
                cov_p = covariance.covariance(point_one, point_two)
                covariance.set_hyperparameters(hyperparameters_old)

                # hyperparamter - h
                hyperparameters_m = numpy.copy(hyperparameters_old)
                hyperparameters_m[k] -= h
                covariance.set_hyperparameters(hyperparameters_m)
                cov_m = covariance.covariance(point_one, point_two)
                covariance.set_hyperparameters(hyperparameters_old)

                # calculate finite diff
                fd_grad = (cov_p - cov_m) / (2.0 * h)

                self.assert_scalar_within_relative(fd_grad, analytic_grad[k], tolerance)


if __name__ == "__main__":
    T.run()
