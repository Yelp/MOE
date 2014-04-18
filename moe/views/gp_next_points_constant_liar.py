# -*- coding: utf-8 -*-
"""Classes for gp_next_points_constant_liar endpoints.

Includes:
    1. pretty and backend views
"""
import colander
from pyramid.view import view_config

from moe.views.gp_next_points_pretty_view import GpNextPointsPrettyView, GpNextPointsRequest


class GpNextPointsConstantLiarRequest(GpNextPointsRequest):

    """Extends the standard request with a lie value."""

    lie_value = colander.SchemaNode(
            colander.Float(),
            missing=0.0,
            )
    lie_noise_variance = colander.SchemaNode(
            colander.Float(),
            missing=0.0,
            validator=colander.Range(min=0.0),
            )


class GpNextPointsconstant_liar(GpNextPointsPrettyView):

    """Views for gp_next_points_constant_liar endpoints."""

    route_name = 'gp_next_points_constant_liar'
    pretty_route_name = 'gp_next_points_constant_liar_pretty'

    request_schema = GpNextPointsConstantLiarRequest()

    pretty_default_request = GpNextPointsPrettyView.pretty_default_request.copy()
    pretty_default_request['lie_value'] = 0.0
    pretty_default_request['lie_noise_variance'] = 0.0

    @view_config(route_name=pretty_route_name, renderer=GpNextPointsPrettyView.pretty_renderer)
    def pretty_view(self):
        """A pretty, browser interactive view for the interface. Includes form request and response."""
        return self.pretty_response()

    @view_config(route_name=route_name, renderer='json', request_method='POST')
    def gp_next_points_constant_liar_view(self):
        """Endpoint for gp_next_points_constant_liar POST requests."""
        params = self.get_params_from_request()
        return self.compute_next_points_to_sample_response(
                params,
                'constant_liar_expected_improvement_optimization',
                self.route_name,
                params.get('lie_value'),
                )
