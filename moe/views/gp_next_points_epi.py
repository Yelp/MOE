# -*- coding: utf-8 -*-
"""Classes for gp_next_points_epi endpoints.

Includes:
    1. pretty and backend views
"""
from pyramid.view import view_config

from moe.views.gp_next_points_pretty_view import GpNextPointsPrettyView


class GpNextPointsEpi(GpNextPointsPrettyView):

    """Views for gp_next_points_epi endpoints."""

    route_name = 'gp_next_points_epi'
    pretty_route_name = 'gp_next_points_epi_pretty'

    @view_config(route_name=pretty_route_name, renderer=GpNextPointsPrettyView.pretty_renderer)
    def pretty_view(self):
        """A pretty, browser interactive view for the interface. Includes form request and response."""
        return self.pretty_response()

    @view_config(route_name=route_name, renderer='json', request_method='POST')
    def gp_next_points_epi_view(self):
        """Endpoint for gp_next_points_epi POST requests."""
        params = self.get_params_from_request()
        return self.compute_next_points_to_sample_response(
                params,
                'multistart_expected_improvement_optimization',
                self.route_name,
                )
