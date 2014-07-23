# -*- coding: utf-8 -*-
"""Classes for gp_next_points_epi endpoints.

Includes:
    1. pretty and backend views
"""
from pyramid.view import view_config

from moe.views.constant import GP_NEXT_POINTS_EPI_ROUTE_NAME, GP_NEXT_POINTS_EPI_PRETTY_ROUTE_NAME, GP_NEXT_POINTS_EPI_OPTIMIZER_METHOD_NAME
from moe.views.gp_next_points_pretty_view import GpNextPointsPrettyView
from moe.views.pretty_view import PRETTY_RENDERER


class GpNextPointsEpi(GpNextPointsPrettyView):

    """Views for gp_next_points_epi endpoints."""

    _route_name = GP_NEXT_POINTS_EPI_ROUTE_NAME
    _pretty_route_name = GP_NEXT_POINTS_EPI_PRETTY_ROUTE_NAME

    @view_config(route_name=_pretty_route_name, renderer=PRETTY_RENDERER)
    def pretty_view(self):
        """A pretty, browser interactive view for the interface. Includes form request and response.

        .. http:get:: /gp/next_points/epi/pretty

        """
        return self.pretty_response()

    @view_config(route_name=_route_name, renderer='json', request_method='POST')
    def gp_next_points_epi_view(self):
        """Endpoint for gp_next_points_epi POST requests.

        .. http:post:: /gp/next_points/epi

           Calculates the next best points to sample, given historical data, using Expected Parallel Improvement (EPI).

           :input: :class:`moe.views.gp_next_points_pretty_view.GpNextPointsRequest`
           :output: :class:`moe.views.gp_next_points_pretty_view.GpNextPointsResponse`

           :status 200: returns a response
           :status 500: server error

        """
        params = self.get_params_from_request()
        return self.compute_next_points_to_sample_response(
                params,
                GP_NEXT_POINTS_EPI_OPTIMIZER_METHOD_NAME,
                self._route_name,
                )
