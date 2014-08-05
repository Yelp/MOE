# -*- coding: utf-8 -*-
"""An example for accessing the all the functions in moe/easy_interface/simple_endpoint.

The following functions are used:

    1. :func:`moe.easy_interface.simple_endpoint.gp_next_points`
    2. :func:`moe.easy_interface.simple_endpoint.gp_hyper_opt`
    3. :func:`moe.easy_interface.simple_endpoint.gp_mean_var`

The function requires some historical information to inform the Gaussian Process
we use an arbitrary function (with noise) function_to_minimize

We first sample [0,0] from the function and then generate and sample 5 optimal points from moe sequentially
We then update the hyperparameters of the GP (model selection)
This process is repeated until we have sampled 20 points in total
We then calculate the posterior mean and variance of the GP at several points
"""
import numpy

from moe.easy_interface.experiment import Experiment
from moe.easy_interface.simple_endpoint import gp_next_points, gp_hyper_opt, gp_mean_var


def function_to_minimize(x):
    """Calculate an aribitrary 2-d function with some noise.

    This function has a minimum near [1, 2.6].
    """
    return numpy.sin(x[0]) * numpy.cos(x[1]) + numpy.cos(x[0] + x[1]) + numpy.random.uniform(-0.02, 0.02)


def run_example(num_to_sample=20, verbose=True, testapp=None, gp_next_points_kwargs=None, gp_hyper_opt_kwargs=None, gp_mean_var_kwargs=None, **kwargs):
    """Run the combined example.

    :param num_to_sample: Number of points for MOE to suggest and then sample [20]
    :type num_to_sample: int > 0
    :param verbose: Whether to print information to the screen [True]
    :type verbose: bool
    :param testapp: Whether to use a supplied test pyramid application or a rest server [None]
    :type testapp: Pyramid test application
    :param gp_next_points_kwargs: Optional kwargs to pass to gp_next_points endpoint
    :type gp_next_points_kwargs: dict
    :param gp_hyper_opt_kwargs: Optional kwargs to pass to gp_hyper_opt_kwargs endpoint
    :type gp_hyper_opt_kwargs: dict
    :param gp_mean_var_kwargs: Optional kwargs to pass to gp_mean_var_kwargs endpoint
    :type gp_mean_var_kwargs: dict
    :param kwargs: Optional kwargs to pass to all endpoints
    :type kwargs: dict

    """
    # Set and combine all optional kwargs
    # Note that the more specific kwargs take precedence (and will override general kwargs)
    if gp_next_points_kwargs is None:
        gp_next_points_kwargs = {}
    else:
        gp_next_points_kwargs = dict(kwargs.items() + gp_next_points_kwargs.items())

    if gp_hyper_opt_kwargs is None:
        gp_hyper_opt_kwargs = {}
    else:
        gp_hyper_opt_kwargs = dict(kwargs.items() + gp_hyper_opt_kwargs.items())

    if gp_mean_var_kwargs is None:
        gp_mean_var_kwargs = {}
    else:
        gp_mean_var_kwargs = dict(kwargs.items() + gp_mean_var_kwargs.items())

    exp = Experiment([[0, 2], [0, 4]])
    # Bootstrap with some known or already sampled point(s)
    exp.historical_data.append_sample_points([
        [[0, 0], function_to_minimize([0, 0]), 0.01],  # sampled points have the form [point_as_a_list, objective_function_value, value_variance]
        ])

    # Sample points
    for i in range(num_to_sample):
        covariance_info = {}
        if i > 0 and i % 5 == 0:
            covariance_info = gp_hyper_opt(exp.historical_data.to_list_of_sample_points(), testapp=testapp, **gp_hyper_opt_kwargs)

            if verbose:
                print "Updated covariance_info with {0:s}".format(str(covariance_info))
        # Use MOE to determine what is the point with highest Expected Improvement to use next
        next_point_to_sample = gp_next_points(
                exp,
                covariance_info=covariance_info,
                testapp=testapp,
                **gp_next_points_kwargs
                )[0]  # By default we only ask for one point
        # Sample the point from our objective function, we can replace this with any function
        value_of_next_point = function_to_minimize(next_point_to_sample)

        if verbose:
            print "Sampled f({0:s}) = {1:.18E}".format(str(next_point_to_sample), value_of_next_point)

        # Add the information about the point to the experiment historical data to inform the GP
        exp.historical_data.append_sample_points([[next_point_to_sample, value_of_next_point, 0.01]])  # We can add some noise

    points_to_evaluate = [[x, x] for x in numpy.arange(0, 1, 0.1)]  # uniform grid of points
    mean, var = gp_mean_var(
            exp.historical_data.to_list_of_sample_points(),  # Historical data to inform Gaussian Process
            points_to_evaluate,  # We will calculate the mean and variance of the GP at these points
            testapp=testapp,
            **gp_mean_var_kwargs
            )

    if verbose:
        print "GP mean at (0, 0), (0.1, 0.1), ...: {0:s}".format(str(mean))


if __name__ == '__main__':
    run_example()
