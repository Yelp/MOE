# -*- coding: utf-8 -*-
"""Classes for gp_next_points_kriging endpoints.

Includes:
    1. pretty and backend views
"""
import colander
from pyramid.view import view_config

from moe.views.gp_next_points_pretty_view import GpNextPointsPrettyView, GpNextPointsRequest
from moe.views.constant import GP_NEXT_POINTS_KRIGING_ROUTE_NAME, GP_NEXT_POINTS_KRIGING_PRETTY_ROUTE_NAME


class GpNextPointsKrigingRequest(GpNextPointsRequest):

    """Extends the standard request moe.views.gp_next_points_pretty_view.GpNextPointsRequest() with kriging parameters.

    **Required fields**

        :gp_info: a moe.views.schemas.GpInfo object of historical data

    **Optional fields**

        :num_samples_to_generate: number of next points to generate (default: 1)
        :ei_optimization_parameters: moe.views.schemas.EiOptimizationParameters() object containing optimization parameters (default: moe.optimal_learning.EPI.src.python.constant.default_ei_optimization_parameters)
        :std_deviation_coef: a float used in Kriging, see Kriging implementation docs (default: 0.0)
        :kriging_noise_variance: a positive (>= 0) float used in Kriging, see Kriging implementation docs (default: 0.0)

    **Example Request**

    .. sourcecode:: http

        Content-Type: text/javascrip

        {
            'num_samples_to_generate': 1,
            'std_deviation_coef': 0.0,
            'kriging_noise_variance': 0.0,
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

    std_deviation_coef = colander.SchemaNode(
            colander.Float(),
            missing=0.0,
            )
    kriging_noise_variance = colander.SchemaNode(
            colander.Float(),
            missing=0.0,
            validator=colander.Range(min=0.0),
            )


class GpNextPointsKriging(GpNextPointsPrettyView):

    """Views for gp_next_points_kriging endpoints."""

    route_name = GP_NEXT_POINTS_KRIGING_ROUTE_NAME
    pretty_route_name = GP_NEXT_POINTS_KRIGING_PRETTY_ROUTE_NAME

    request_schema = GpNextPointsKrigingRequest()

    pretty_default_request = GpNextPointsPrettyView.pretty_default_request.copy()
    pretty_default_request['std_deviation_coef'] = 0.0
    pretty_default_request['kriging_noise_variance'] = 0.0

    @view_config(route_name=pretty_route_name, renderer=GpNextPointsPrettyView.pretty_renderer)
    def pretty_view(self):
        """A pretty, browser interactive view for the interface. Includes form request and response."""
        return self.pretty_response()

    @view_config(route_name=route_name, renderer='json', request_method='POST')
    def gp_next_points_kriging_view(self):
        """Endpoint for gp_next_points_kriging POST requests."""
        params = self.get_params_from_request()
        return self.compute_next_points_to_sample_response(
                params,
                'kriging_believer_expected_improvement_optimization',
                self.route_name,
                )
