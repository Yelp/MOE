# -*- coding: utf-8 -*-
"""An example for accessing the gp_next_points_* endpoints.

:func:`moe.easy_interface.simple_endpoint.gp_next_points`

The function requires some historical information to inform the Gaussian Process

The result is the next best set of point(s) to sample

.. Note::

    This file is copied in ``README.md`` and ``docs/index.rst``. Any changes here should be made there as well.
"""
import math
import random

from moe.easy_interface.experiment import Experiment
from moe.easy_interface.simple_endpoint import gp_next_points
from moe.optimal_learning.python.data_containers import SamplePoint
from moe.optimal_learning.python.constant import CPP_COMPONENT_INSTALLED


# Note: this function can be anything, the output of a batch, results of an A/B experiment, the value of a physical experiment etc.
def function_to_minimize(x):
    """Calculate an aribitrary 2-d function with some noise with minimum near [1, 2.6]."""
    return math.sin(x[0]) * math.cos(x[1]) + math.cos(x[0] + x[1]) + random.uniform(-0.02, 0.02)


def run_example(num_points_to_sample=20, verbose=True, **kwargs):
    """Run the example, aksing MOE for ``num_points_to_sample`` optimal points to sample."""
    exp = Experiment([[0, 2], [0, 4]])  # 2D experiment, we build a tensor product domain
    # Bootstrap with some known or already sampled point(s)
    exp.historical_data.append_sample_points([
        SamplePoint([0, 0], function_to_minimize([0, 0]), 0.05),  # Iterables of the form [point, f_val, f_var] are also allowed
        ])

    # Sample num_points_to_sample points
    for _ in range(num_points_to_sample):
        # Use MOE to determine what is the point with highest Expected Improvement to use next
        next_point_to_sample = gp_next_points(exp, **kwargs)[0]  # By default we only ask for one point
        # Sample the point from our objective function, we can replace this with any function
        value_of_next_point = function_to_minimize(next_point_to_sample)

        if verbose:
            print "Sampled f({0:s}) = {1:.18E}".format(str(next_point_to_sample), value_of_next_point)

        # Add the information about the point to the experiment historical data to inform the GP
        exp.historical_data.append_sample_points([SamplePoint(next_point_to_sample, value_of_next_point, 0.01)])  # We can add some noise


if __name__ == '__main__':
    if CPP_COMPONENT_INSTALLED:
        run_example()
