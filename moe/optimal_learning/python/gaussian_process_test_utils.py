# -*- coding: utf-8 -*-
"""Utilities for generating domains, hyperparameters of covariance, and gaussian processes; useful primarily for testing.

By default, the functions in this file use the Python optimal_learning library (python_version package). Users
can override this behavior with any implementation of the ABCs in the interfaces package.

"""
import numpy

from moe.optimal_learning.python.data_containers import HistoricalData, SamplePoint
from moe.optimal_learning.python.geometry_utils import ClosedInterval
from moe.optimal_learning.python.python_version.covariance import SquareExponential
from moe.optimal_learning.python.python_version.gaussian_process import GaussianProcess


def fill_random_covariance_hyperparameters(hyperparameter_interval, num_hyperparameters, covariance_type=SquareExponential):
    """Generate random hyperparameters and returns a covariance object with those hyperparameters.

    :param hyperparameter_interval: range, [min, max], from which to draw the hyperparameters
    :type hyperparameter_interval: ClosedInterval
    :param num_hyperparameters: number of hyperparameters
    :type num_hyperparameters: int > 0
    :param covariance_type: covariance function whose hyperparameters are being set
    :type covariance_type: interfaces.covariance_interface.CovarianceInterface subclass
    :return: covariance_type instantiated with the generated hyperparameters
    :rtype: covariance_type object

    """
    hyper = [numpy.random.uniform(hyperparameter_interval.min, hyperparameter_interval.max) for _ in xrange(num_hyperparameters)]
    return covariance_type(hyper)


def fill_random_domain_bounds(lower_bound_interval, upper_bound_interval, dim):
    r"""Generate a random list of dim ``[min_i, max_i]`` pairs.

    The data is organized such that:
    ``min_i \in [uniform_double_lower_bound.a(), uniform_double_lower_bound.b()]``
    ``max_i \in [uniform_double_upper_bound.a(), uniform_double_upper_bound.b()]``

    :param lower_bound_interval: an uniform range, ``[min, max]``, from which to draw the domain lower bounds, ``min_i``
    :type lower_bound_interval: ClosedInterval
    :param upper_bound_interval: an uniform range, ``[min, max]``, from which to draw the domain upper bounds, ``max_i``
    :type upper_bound_interval: ClosedInterval
    :param dim: the spatial dimension of a point (i.e., number of independent params in experiment)
    :type dim: int > 0
    :return: ClosedInterval objects with their min, max members initialized as described
    :rtype: list of ClosedInterval

    """
    temp = numpy.empty((dim, 2))
    temp[..., 0] = numpy.random.uniform(lower_bound_interval.min, lower_bound_interval.max)
    temp[..., 1] = numpy.random.uniform(upper_bound_interval.min, upper_bound_interval.max)
    return ClosedInterval.build_closed_intervals_from_list(temp)


def build_random_gaussian_process(points_sampled, covariance, noise_variance=None, gaussian_process_type=GaussianProcess):
    r"""Utility to draw ``points_sampled.shape[0]`` points from a GaussianProcess and add those values to the prior.

    :param points_sampled: points at which to draw from the GP
    :type points_sampled: array of float64 with shape (num_sampled, dim)
    :param covariance: covariance function backing the GP
    :type covariance: interfaces.covariance_interface.CovarianceInterface subclass composable with gaussian_process_type
    :param noise_variance: the ``\sigma_n^2`` (noise variance) associated w/the new observations, ``points_sampled_value``
    :type noise_variance: array of float64 with shape (num_sampled)
    :param gaussian_process_type: gaussian process whose historical data is being set
    :type gaussian_process_type: interfaces.gaussian_process_interface.GaussianProcessInterface subclass
    :return: a gaussian process with the generated prior data
    :rtype: gaussian_process_type object

    """
    if noise_variance is None:
        noise_variance = numpy.zeros(points_sampled.shape[0])

    gaussian_process = gaussian_process_type(covariance, HistoricalData(points_sampled.shape[1]))
    for i, point in enumerate(points_sampled):
        # Draw function value from the GP
        function_value = gaussian_process.sample_point_from_gp(point, noise_variance=noise_variance[i])
        # Add function value back into the GP
        sample_point = [SamplePoint(point, function_value, noise_variance[i])]
        gaussian_process.add_sampled_points(sample_point)

    return gaussian_process
