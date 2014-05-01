# -*- coding: utf-8 -*-
"""A class to encapsulate 'pretty' views for gp_next_points_* endpoints.

Include:
    1. Request and response schemas
    2. Class that extends GpPrettyView for next_points optimizers
"""
import colander

import moe.build.GPP as C_GP
from moe.optimal_learning.python.cpp_wrappers.optimization_parameters import ExpectedImprovementOptimizationParameters
from moe.views.gp_pretty_view import GpPrettyView
from moe.views.schemas import GpInfo, EiOptimizationParameters, ListOfPointsInDomain, ListOfExpectedImprovements
from moe.views.utils import _make_gp_from_gp_info


class GpNextPointsRequest(colander.MappingSchema):

    """A gp_next_points_* request colander schema.

    **Required fields**

        :gp_info: a moe.views.schemas.GpInfo object of historical data

    **Optional fields**

        :num_samples_to_generate: number of next points to generate (default: 1)
        :ei_optimization_parameters: moe.views.schemas.EiOptimizationParameters() object containing optimization parameters (default: moe.optimal_learning.EPI.src.python.constant.default_ei_optimization_parameters)

    **Example Request**

    .. sourcecode:: http

        Content-Type: text/javascript

        {
            'num_samples_to_generate': 1,
            'gp_info': {
                'points_sampled': [
                        {'value_var': 0.01, 'value': 0.1, 'point': [0.0]},
                        {'value_var': 0.01, 'value': 0.2, 'point': [1.0]}
                    ],
                'domain': [
                    [0, 1],
                    ]
                },
            },
        }

    """

    num_samples_to_generate = colander.SchemaNode(
            colander.Int(),
            validator=colander.Range(min=1),
            )
    gp_info = GpInfo()
    ei_optimization_parameters = EiOptimizationParameters(
            missing=EiOptimizationParameters().deserialize({})
            )


class GpNextPointsResponse(colander.MappingSchema):

    """A gp_next_points_* response colander schema.

    **Output fields**

        :endpoint: the endpoint that was called
        :points_to_sample: list of points in the domain to sample next (moe.views.schemas.ListOfPointsInDomain)
        :expected_improvement: list of EI of points in points_to_sample (moe.views.schemas.ListOfExpectedImprovements)

    **Example Response**

    .. sourcecode:: http

        {
            "endpoint":"gp_ei",
            "points_to_sample": [["0.478332304526"]],
            "expected_improvement": ["0.443478498868"],
        }

    """

    endpoint = colander.SchemaNode(colander.String())
    points_to_sample = ListOfPointsInDomain()
    expected_improvement = ListOfExpectedImprovements()


class GpNextPointsPrettyView(GpPrettyView):

    """A class to encapsulate 'pretty' gp_next_points_* views.

    Extends GpPrettyView with:
        1. GP generation from params
        2. Converting params into a C++ consumable set of optimization parameters
        3. A method (compute_next_points_to_sample_response) for computing the next best points to sample from a GP

    """

    request_schema = GpNextPointsRequest()
    response_schema = GpNextPointsResponse()

    _pretty_default_request = {
            "num_samples_to_generate": 1,
            "gp_info": GpPrettyView._pretty_default_gp_info,
            }

    def compute_next_points_to_sample_response(self, params, optimization_method_name, route_name, *args, **kwargs):
        """Compute the next points to sample (and their expected improvement) using optimization_method_name from params in the request.

        :param deserialized_request_params: the deserialized REST request, containing ei_optimization_parameters and gp_info
        :type deserialized_request_params: a deserialized self.request_schema object
        :param optimization_method_name: the optimization method to use
        :type optimization_method_name: string in moe.views.constant.OPTIMIZATION_METHOD_NAMES
        :param route_name: name of the route being called
        :type route_name: string in moe.views.constant.ALL_REST_ROUTES_ROUTE_NAME_TO_ENDPOINT.keys()
        :param *args: extra args to be passed to optimization method
        :param **kwargs: extra kwargs to be passed to optimization method

        """
        num_samples_to_generate = params.get('num_samples_to_generate')

        GP = self.make_gp(params)
        ei_optimization_parameters_cpp = self.get_optimization_parameters_cpp(params)

        optimization_method = getattr(GP, optimization_method_name)

        next_points = optimization_method(
                ei_optimization_parameters_cpp,
                num_samples_to_generate,
                *args,
                **kwargs
                )
        expected_improvement = GP.evaluate_expected_improvement_at_point_list(next_points)

        return self.form_response({
                'endpoint': route_name,
                'points_to_sample': next_points.tolist(),
                'expected_improvement': expected_improvement.tolist(),
                })

    @staticmethod
    def make_gp(deserialized_request_params):
        """Create a GP object from deserialized request params.

        :param deserialized_request_params: the deserialized params of a REST request, containing gp_info
        :type deserialized_request_params: a dictionary with a key 'gp_info' containing a deserialized moe.views.schemas.GpInfo object of historical data.

        """
        gp_info = deserialized_request_params.get('gp_info')
        return _make_gp_from_gp_info(gp_info)

    @staticmethod
    def get_optimization_parameters_cpp(deserialized_request_params):
        """Form a C++ consumable ExpectedImprovementOptimizationParameters object from deserialized request params.

        :param deserialized_request_params: the deserialized REST request, containing ei_optimization_parameters
        :type deserialized_request_params: a dictionary with a key ei_optimization_parameters containing a moe.views.schemas.EiOptimizationParameters() object with optimization parameters

        """
        ei_optimization_parameters = deserialized_request_params.get('ei_optimization_parameters')

        # Note: num_random_samples only has meaning when computing more than 1 points_to_sample simultaneously
        new_params = ExpectedImprovementOptimizationParameters(
            optimizer_type=getattr(
                C_GP.OptimizerTypes,
                ei_optimization_parameters.get('optimizer_type')
                ),
            num_random_samples=ei_optimization_parameters.get('num_random_samples'),
            optimizer_parameters=C_GP.GradientDescentParameters(
                ei_optimization_parameters.get('num_multistarts'),
                ei_optimization_parameters.get('gd_iterations'),
                ei_optimization_parameters.get('max_num_restarts'),
                ei_optimization_parameters.get('gamma'),
                ei_optimization_parameters.get('pre_mult'),
                ei_optimization_parameters.get('max_relative_change'),
                ei_optimization_parameters.get('tolerance'),
            ),
        )
        # TODO(eliu): domain_type should passed as part of the domain; this is a hack until I
        # refactor these calls to use the new interface
        new_params.domain_type = C_GP.DomainTypes.tensor_product
        return new_params
