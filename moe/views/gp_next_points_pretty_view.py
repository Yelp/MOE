# -*- coding: utf-8 -*-
"""A class to encapsulate 'pretty' views for ``gp_next_points_*`` endpoints.

Include:
    1. Request and response schemas
    2. Class that extends GpPrettyView for next_points optimizers
"""
import colander

import numpy

from moe.optimal_learning.python.constant import DEFAULT_EXPECTED_IMPROVEMENT_MC_ITERATIONS
import moe.optimal_learning.python.cpp_wrappers.expected_improvement
from moe.optimal_learning.python.cpp_wrappers.expected_improvement import ExpectedImprovement
from moe.views.gp_pretty_view import GpPrettyView
from moe.views.optimizable_gp_pretty_view import OptimizableGpPrettyView
from moe.views.schemas import GpHistoricalInfo, ListOfPointsInDomain, CovarianceInfo, BoundedDomainInfo, OptimizationInfo
from moe.views.utils import _make_gp_from_params, _make_domain_from_params, _make_optimization_parameters_from_params


class GpNextPointsRequest(colander.MappingSchema):

    """A ``gp_next_points_*`` request colander schema.

    **Required fields**

        :gp_historical_info: a :class:`moe.views.schemas.GpHistoricalInfo` dict of historical data
        :domain_info: a :class:`moe.views.schemas.BoundedDomainInfo` dict of domain information

    **Optional fields**

        :num_to_sample: number of next points to generate (default: 1)
        :mc_iterations: number of Monte Carlo (MC) iterations to perform in numerical integration to calculate EI
        :covariance_info: a :class:`moe.views.schemas.CovarianceInfo` dict of covariance information
        :optimization_info: a :class:`moe.views.schemas.OptimizationInfo` dict of optimization information
        :points_being_sampled: list of points in domain being sampled in concurrent experiments (default: [])

    **Example Minimal Request**

    .. sourcecode:: http

        Content-Type: text/javascript

        {
            "num_to_sample": 1,
            "gp_historical_info": {
                "points_sampled": [
                        {"value_var": 0.01, "value": 0.1, "point": [0.0]},
                        {"value_var": 0.01, "value": 0.2, "point": [1.0]}
                    ],
                },
            "domain_info": {
                "dim": 1,
                "domain_bounds": [
                    {"min": 0.0, "max": 1.0},
                    ],
                },
        }

    **Example Full Request**

    .. sourcecode:: http

        Content-Type: text/javascript

        {
            "num_to_sample": 1,
            "points_being_sampled": [[0.2], [0.7]],
            "mc_iterations": 10000,
            "gp_historical_info": {
                "points_sampled": [
                        {"value_var": 0.01, "value": 0.1, "point": [0.0]},
                        {"value_var": 0.01, "value": 0.2, "point": [1.0]}
                    ],
                },
            "domain_info": {
                "domain_type": "tensor_product"
                "dim": 1,
                "domain_bounds": [
                    {"min": 0.0, "max": 1.0},
                    ],
                },
            "covariance_info": {
                "covariance_type": "square_exponential",
                "hyperparameters": [1.0, 1.0],
                },
            "optimization_info": {
                "optimization_type": "gradient_descent_optimizer",
                "num_multistarts": 200,
                "num_random_samples": 4000,
                "optimization_parameters": {
                    "gamma": 0.5,
                    ...
                    },
                },
        }

    """

    num_to_sample = colander.SchemaNode(
            colander.Int(),
            missing=1,
            validator=colander.Range(min=1),
            )
    mc_iterations = colander.SchemaNode(
            colander.Int(),
            validator=colander.Range(min=1),
            missing=DEFAULT_EXPECTED_IMPROVEMENT_MC_ITERATIONS,
            )
    gp_historical_info = GpHistoricalInfo()
    domain_info = BoundedDomainInfo()
    covariance_info = CovarianceInfo(
            missing=CovarianceInfo().deserialize({}),
            )
    optimization_info = OptimizationInfo(
            missing=OptimizationInfo().deserialize({}),
            )
    points_being_sampled = ListOfPointsInDomain(
            missing=[],
            )


class GpNextPointsResponse(colander.MappingSchema):

    """A ``gp_next_points_*`` response colander schema.

    **Output fields**

        :endpoint: the endpoint that was called
        :points_to_sample: list of points in the domain to sample next (:class:`moe.views.schemas.ListOfPointsInDomain`)
        :expected_improvement: list of EI of points in points_to_sample (:class:`moe.views.schemas.ListOfExpectedImprovements`)

    **Example Response**

    .. sourcecode:: http

        {
            "endpoint":"gp_ei",
            "points_to_sample": [["0.478332304526"]],
            "expected_improvement": "0.443478498868",
        }

    """

    endpoint = colander.SchemaNode(colander.String())
    points_to_sample = ListOfPointsInDomain()
    expected_improvement = colander.SchemaNode(
            colander.Float(),
            validator=colander.Range(min=0.0),
            )


class GpNextPointsPrettyView(OptimizableGpPrettyView):

    """A class to encapsulate 'pretty' ``gp_next_points_*`` views.

    Extends GpPrettyView with:
        1. gaussian_process generation from params
        2. Converting params into a C++ consumable set of optimization parameters
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

    def compute_next_points_to_sample_response(self, params, optimization_method_name, route_name, *args, **kwargs):
        """Compute the next points to sample (and their expected improvement) using optimization_method_name from params in the request.

        :param request_params: the deserialized REST request, containing ei_optimization_parameters and gp_historical_info
        :type request_params: a deserialized self.request_schema object as a dict
        :param optimization_method_name: the optimization method to use
        :type optimization_method_name: string in ``moe.views.constant.OPTIMIZATION_METHOD_NAMES``
        :param route_name: name of the route being called
        :type route_name: string in ``moe.views.constant.ALL_REST_ROUTES_ROUTE_NAME_TO_ENDPOINT.keys()``
        :param ``*args``: extra args to be passed to optimization method
        :param ``**kwargs``: extra kwargs to be passed to optimization method

        """
        points_being_sampled = numpy.array(params.get('points_being_sampled'))
        num_to_sample = params.get('num_to_sample')
        num_mc_iterations = params.get('mc_iterations')

        gaussian_process = _make_gp_from_params(params)

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

            optimizer_class, optimization_parameters, num_random_samples = _make_optimization_parameters_from_params(params)

            expected_improvement_optimizer = optimizer_class(
                    domain,
                    expected_improvement_evaluator,
                    optimization_parameters,
                    num_random_samples=num_random_samples,
                    )

            opt_method = getattr(moe.optimal_learning.python.cpp_wrappers.expected_improvement, optimization_method_name)

            next_points = opt_method(
                    expected_improvement_optimizer,
                    optimization_parameters.num_multistarts,
                    num_to_sample,
                    *args,
                    **kwargs
                    )

        expected_improvement_evaluator.current_point = next_points
        expected_improvement = expected_improvement_evaluator.compute_expected_improvement()

        return self.form_response({
                'endpoint': route_name,
                'points_to_sample': next_points.tolist(),
                'expected_improvement': expected_improvement,
                })
