"""An example for accessing the gp_mean_var simple endpoint.

:func:`moe.easy_interface.simple_endpoint.gp_mean_var`

The function requires some historical information to inform the Gaussian Process
and a set of points to calculate the posterior mean and variance at.

The posterior mean and variance is then printed for every point.
"""
import random

import numpy

from moe.easy_interface.simple_endpoint import gp_mean_var

# Randomly generate some historical data
# points_sampled is an iterable of iterables of the form [point_as_a_list, objective_function_value, value_variance]
points_sampled = [
        [[x], random.uniform(-1, 1), 0.01] for x in numpy.arange(0, 1, 0.1)
        ]

if __name__ == '__main__':
    points_to_evaluate = [[x] for x in numpy.arange(0, 1, 0.05)]  # uniform grid of points
    mean, var = gp_mean_var(
            points_sampled,  # Historical data to inform Gaussian Process
            points_to_evaluate,  # We will calculate the mean and variance of the GP at these points
            )
    # Print out the mean and variance of the GP at each point_to_evaluate
    for i, point in enumerate(points_to_evaluate):
        print "GP(%s) ~ N(%f, %f)" % (str(point), mean[i], var[i][i])
