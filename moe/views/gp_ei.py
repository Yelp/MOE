"""Classes for gp_ei endpoints.

Includes:
    1. request and response schemas
    2. pretty and backend views
"""
from pyramid.view import view_config

from moe.views import _make_gp_from_gp_info
from moe.views.gp_pretty_view import GpPrettyView

import colander
from moe.views.schemas import ListOfPointsInDomain, GpInfo, ListOfExpectedImprovements
from moe.optimal_learning.EPI.src.python.constant import default_expected_improvement_parameters


class GpEiRequest(colander.MappingSchema):

    """A gp_ei request colander schema."""

    points_to_evaluate = ListOfPointsInDomain()
    points_being_sampled = ListOfPointsInDomain(
            missing=[],
            )
    mc_iterations = colander.SchemaNode(
            colander.Int(),
            validator=colander.Range(min=1),
            missing=default_expected_improvement_parameters.mc_iterations,
            )
    gp_info = GpInfo()


class GpEiResponse(colander.MappingSchema):

    """A gp_ei response colander schema."""

    endpoint = colander.SchemaNode(colander.String())
    expected_improvement = ListOfExpectedImprovements()


class GpEiView(GpPrettyView):

    """Views for gp_ei endpoints."""

    route_name = 'gp_ei'
    pretty_route_name = 'gp_ei_pretty'

    request_schema = GpEiRequest()
    response_schema = GpEiResponse()

    pretty_default_request = {
            "points_to_evaluate": [
                [0.1], [0.5], [0.9],
                ],
            "gp_info": GpPrettyView.pretty_default_gp_info,
            }

    @view_config(route_name=pretty_route_name, renderer=GpPrettyView.pretty_renderer)
    def pretty_view(self):
        """A pretty, browser interactive view for the interface. Includes form request and response."""
        return self.pretty_response()

    @view_config(route_name=route_name, renderer='json', request_method='POST')
    def gp_ei_view(self):
        """Endpoint for gp_ei POST requests."""
        params = self.get_params_from_request()

        points_to_evaluate = params.get('points_to_evaluate')
        points_being_sampled = params.get('points_being_sampled')
        gp_info = params.get('gp_info')

        GP = _make_gp_from_gp_info(gp_info)

        expected_improvement = GP.evaluate_expected_improvement_at_point_list(
                points_to_evaluate,
                points_being_sampled=points_being_sampled,
                )

        return self.form_response({
                'endpoint': 'gp_ei',
                'expected_improvement': expected_improvement,
                })
