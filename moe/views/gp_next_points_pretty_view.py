# -*- coding: utf-8 -*-
"""A class to encapsulate 'pretty' views for ``gp_next_points_*`` endpoints.

Include:
    1. Request and response schemas
    2. Class that extends GpPrettyView for next_points optimizers
"""
import numpy

import moe.optimal_learning.python.cpp_wrappers.expected_improvement
from moe.optimal_learning.python.cpp_wrappers.expected_improvement import ExpectedImprovement
from moe.views.gp_pretty_view import GpPrettyView
from moe.views.optimizable_gp_pretty_view import OptimizableGpPrettyView
from moe.views.schemas import GpNextPointsRequest, GpNextPointsResponse
from moe.views.utils import _make_gp_from_params, _make_domain_from_params, _make_optimizer_parameters_from_params


class GpNextPointsPrettyView(OptimizableGpPrettyView):

    """A class to encapsulate 'pretty' ``gp_next_points_*`` views.

    Extends GpPrettyView with:
        1. gaussian_process generation from params
        2. Converting params into a C++ consumable set of optimizer parameters
        3. A method (compute_next_points_to_sample_response) for computing the next best points to sample from a gaussian_process

    """

    request_schema = GpNextPointsRequest()
    response_schema = GpNextPointsResponse()

    _pretty_default_request = {
            "num_to_sample": 1,
            "gp_historical_info": GpPrettyView._pretty_default_gp_historical_info,
            "domain_info": {
                "dim": 1,
                "domain_bounds": [
                    {
                        "min": 0.0,
                        "max": 1.0,
                    },
                    ],
                },
            }

    def compute_next_points_to_sample_response(self, params, optimizer_method_name, route_name, *args, **kwargs):
        """Compute the next points to sample (and their expected improvement) using optimizer_method_name from params in the request.

        :param request_params: the deserialized REST request, containing ei_optimizer_parameters and gp_historical_info
        :type request_params: a deserialized self.request_schema object as a dict
        :param optimizer_method_name: the optimization method to use
        :type optimizer_method_name: string in ``moe.views.constant.OPTIMIZER_METHOD_NAMES``
        :param route_name: name of the route being called
        :type route_name: string in ``moe.views.constant.ALL_REST_ROUTES_ROUTE_NAME_TO_ENDPOINT.keys()``
        :param ``*args``: extra args to be passed to optimization method
        :param ``**kwargs``: extra kwargs to be passed to optimization method

        """
        points_being_sampled = numpy.array(params.get('points_being_sampled'))
        num_to_sample = params.get('num_to_sample')
        num_mc_iterations = params.get('mc_iterations')
        max_num_threads = params.get('max_num_threads')

        gaussian_process = _make_gp_from_params(params)

        if gaussian_process.num_sampled < num_to_sample:
            self.log.warning("Attempting to find {0:d} optimal points with only {1:d} (< {0:d}) historical points sampled. This can cause matrix issues under some conditions. Try requesting < {0:d} points for better performance. To bootstrap more points try sampling at random, or from a grid.".format(num_to_sample, gaussian_process.num_sampled))

        expected_improvement_evaluator = ExpectedImprovement(
                gaussian_process,
                points_being_sampled=points_being_sampled,
                num_mc_iterations=num_mc_iterations,
                )

        # TODO(GH-89): Make the optimal_learning library handle this case 'organically' with
        # reasonable default behavior and remove hacks like this one.
        if gaussian_process.num_sampled == 0:
            # If there is no initial data we bootstrap with random points
            py_domain = _make_domain_from_params(params, python_version=True)
            next_points = py_domain.generate_uniform_random_points_in_domain(num_to_sample)
        else:
            # Calculate the next best points to sample given the historical data
            domain = _make_domain_from_params(params)

            optimizer_class, optimizer_parameters, num_random_samples = _make_optimizer_parameters_from_params(params)

            expected_improvement_optimizer = optimizer_class(
                    domain,
                    expected_improvement_evaluator,
                    optimizer_parameters,
                    num_random_samples=num_random_samples,
                    )

            opt_method = getattr(moe.optimal_learning.python.cpp_wrappers.expected_improvement, optimizer_method_name)

            next_points = opt_method(
                    expected_improvement_optimizer,
                    optimizer_parameters.num_multistarts,
                    num_to_sample,
                    max_num_threads=max_num_threads,
                    *args,
                    **kwargs
                    )

        # TODO(GH-285): Use analytic q-EI here
        expected_improvement_evaluator.current_point = next_points
        expected_improvement = expected_improvement_evaluator.compute_expected_improvement()

        return self.form_response({
                'endpoint': route_name,
                'points_to_sample': next_points.tolist(),
                'expected_improvement': expected_improvement,
                })
