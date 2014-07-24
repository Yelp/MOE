# -*- coding: utf-8 -*-
"""Classes for gp_next_points_kriging endpoints.

Includes:
    1. pretty and backend views
"""
from pyramid.view import view_config

from moe.views.constant import GP_NEXT_POINTS_KRIGING_ROUTE_NAME, GP_NEXT_POINTS_KRIGING_PRETTY_ROUTE_NAME, GP_NEXT_POINTS_KRIGING_OPTIMIZER_METHOD_NAME
from moe.views.gp_next_points_pretty_view import GpNextPointsPrettyView
from moe.views.pretty_view import PRETTY_RENDERER
from moe.views.schemas import GpNextPointsKrigingRequest


class GpNextPointsKriging(GpNextPointsPrettyView):

    """Views for gp_next_points_kriging endpoints."""

    _route_name = GP_NEXT_POINTS_KRIGING_ROUTE_NAME
    _pretty_route_name = GP_NEXT_POINTS_KRIGING_PRETTY_ROUTE_NAME

    request_schema = GpNextPointsKrigingRequest()

    _pretty_default_request = GpNextPointsPrettyView._pretty_default_request.copy()
    _pretty_default_request['std_deviation_coef'] = 0.0
    _pretty_default_request['kriging_noise_variance'] = 1e-8

    @view_config(route_name=_pretty_route_name, renderer=PRETTY_RENDERER)
    def pretty_view(self):
        """A pretty, browser interactive view for the interface. Includes form request and response."""
        return self.pretty_response()

    @view_config(route_name=_route_name, renderer='json', request_method='POST')
    def gp_next_points_kriging_view(self):
        """Endpoint for gp_next_points_kriging POST requests.

        .. http:post:: /gp/next_points/kriging

           Calculates the next best points to sample, given historical data, using Kriging.

           :input: :class:`moe.views.schemas.GpNextPointsKrigingRequest`
           :output: :class:`moe.views.schemas.GpNextPointsResponse`

           :status 200: returns a response
           :status 500: server error

        """
        params = self.get_params_from_request()
        return self.compute_next_points_to_sample_response(
                params,
                GP_NEXT_POINTS_KRIGING_OPTIMIZER_METHOD_NAME,
                self._route_name,
                std_deviation_coef=params.get('std_deviation_coef'),
                kriging_noise_variance=params.get('kriging_noise_variance'),
                )
