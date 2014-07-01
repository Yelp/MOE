"""An example for accessing the gp_mean_var simple endpoint.

:func:`moe.easy_interface.simple_endpoint.gp_mean_var`

The function requires some historical information to inform the Gaussian Process

The optimal hyperparameters are returned.
"""
import random

import numpy

from moe.easy_interface.simple_endpoint import gp_hyper_opt
from moe.optimal_learning.python.data_containers import SamplePoint

# Randomly generate some historical data
# points_sampled is an iterable of iterables of the form [point_as_a_list, objective_function_value, value_variance]
points_sampled = [
        SamplePoint(numpy.array([x]), random.uniform(-1, 1), 0.01) for x in numpy.arange(0, 1, 0.1)
        ]

if __name__ == '__main__':
    covariance_info = gp_hyper_opt(
            points_sampled,
            )
    print covariance_info
