# -*- coding: utf-8 -*-
"""Classes for gp_next_points_epi endpoints.

Includes:
    1. pretty and backend views
"""
from pyramid.view import view_config

from moe.views.gp_next_points_pretty_view import GpNextPointsPrettyView
from moe.views.constant import GP_NEXT_POINTS_EPI_ROUTE_NAME, GP_NEXT_POINTS_EPI_PRETTY_ROUTE_NAME


class GpNextPointsEpi(GpNextPointsPrettyView):

    """Views for gp_next_points_epi endpoints."""

    route_name = GP_NEXT_POINTS_EPI_ROUTE_NAME
    pretty_route_name = GP_NEXT_POINTS_EPI_PRETTY_ROUTE_NAME

    @view_config(route_name=pretty_route_name, renderer=GpNextPointsPrettyView.pretty_renderer)
    def pretty_view(self):
        """A pretty, browser interactive view for the interface. Includes form request and response.

        .. http:get:: /gp/next_points/epi/pretty

        """
        return self.pretty_response()

    @view_config(route_name=route_name, renderer='json', request_method='POST')
    def gp_next_points_epi_view(self):
        """Endpoint for gp_next_points_epi POST requests.

        .. http:post:: /gp/next_points/epi

           Calculates the next best points to sample, given historical data, using Expected Parallel Improvement (EPI).

           :input: moe.views.gp_next_points_pretty_view.GpNextPointsRequest()
           :output: moe.views.gp_next_points_pretty_view.GpNextPointsResponse()

           :status 200: returns a response
           :status 500: server error

        """
        params = self.get_params_from_request()
        return self.compute_next_points_to_sample_response(
                params,
                'multistart_expected_improvement_optimization',
                self.route_name,
                )
