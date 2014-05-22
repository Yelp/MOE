import random

import numpy

from moe.easy_interface.simple_endpoint import gp_mean_var

points_sampled = [
        [[x], random.uniform(-1, 1), 0.01] for x in numpy.arange(0.03, 0.97, 0.1)
        ]

print gp_mean_var(
        points_sampled,
        [[x] for x in numpy.arange(0, 1, 0.1)],
        )
