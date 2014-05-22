from moe.easy_interface.experiment import Experiment
from moe.easy_interface.simple_endpoint import gp_next_points

import math, random
def function_to_minimize(x):
    """This function has a minimum near [1, 2.6]."""
    return math.sin(x[0]) * math.cos(x[1]) + math.cos(x[0] + x[1]) + random.uniform(-0.02, 0.02)

exp = Experiment([[0, 2], [0, 4]])
exp.historical_data.append_sample_points([[[0, 0], 1.0, 0.01]]) # Bootstrap with some known or already sampled point

# Sample 20 points
for i in range(20):
    next_point_to_sample = gp_next_points(exp)[0] # By default we only ask for one point
    value_of_next_point = function_to_minimize(next_point_to_sample)
    exp.historical_data.append_sample_points([[next_point_to_sample, value_of_next_point, 0.01]]) # We can add some noise

print exp
