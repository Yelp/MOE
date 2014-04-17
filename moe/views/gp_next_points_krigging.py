"""Classes for gp_next_points_kriging endpoints.

Includes:
    1. pretty and backend views
"""

import colander
from pyramid.view import view_config

from moe.views.gp_next_points_pretty_view import GpNextPointsPrettyView, GpNextPointsRequest


class GpNextPointsKrigingRequest(GpNextPointsRequest):

    """Extends the standard request with a lie value."""

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

    route_name = 'gp_next_points_kriging'
    pretty_route_name = 'gp_next_points_kriging_pretty'

    request_schema = GpNextPointsKrigingRequest()

    pretty_default_request = GpNextPointsPrettyView.pretty_default_request.copy()
    pretty_default_request['std_deviation_coef'] = 0.0
    pretty_default_request['kriging_noise_variance'] = 0.0

    @view_config(route_name=pretty_route_name, renderer=GpNextPointsPrettyView.pretty_renderer)
    def pretty_view(self):
        """A pretty, browser interactable view for the interface. Includes form request and response."""
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
