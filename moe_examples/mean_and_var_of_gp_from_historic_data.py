# -*- coding: utf-8 -*-
"""An example for accessing the gp_mean_var simple endpoint.

:func:`moe.easy_interface.simple_endpoint.gp_mean_var`

The function requires some historical information to inform the Gaussian Process
and a set of points to calculate the posterior mean and variance at.

The posterior mean and variance is then printed for every point.
"""
from __future__ import print_function
import numpy

from moe.easy_interface.simple_endpoint import gp_mean_var

# Randomly generate some historical data
# points_sampled is an iterable of iterables of the form [point_as_a_list, objective_function_value, value_variance]
points_sampled = [
        [[x], numpy.random.uniform(-1, 1), 0.01] for x in numpy.arange(0, 1, 0.1)
        ]


def run_example(verbose=True, testapp=None, **kwargs):
    """Run the example, finding the posterior mean and variance for various poinst from a random GP."""
    points_to_evaluate = [[x] for x in numpy.arange(0, 1, 0.05)]  # uniform grid of points
    mean, var = gp_mean_var(
            points_sampled,  # Historical data to inform Gaussian Process
            points_to_evaluate,  # We will calculate the mean and variance of the GP at these points
            testapp=testapp,
            **kwargs
            )

    if verbose:
        # Print out the mean and variance of the GP at each point_to_evaluate
        for i, point in enumerate(points_to_evaluate):
            print("GP({0:s}) ~ N({1:.18E}, {2:.18E})".format(str(point), mean[i], var[i][i]))


if __name__ == '__main__':
    run_example()
