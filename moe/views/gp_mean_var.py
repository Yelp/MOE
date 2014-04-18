# -*- coding: utf-8 -*-
"""Classes for gp_mean_var endpoints.

Includes:
    1. request and response schemas
    2. pretty and backend views
"""
import numpy

from pyramid.view import view_config

from moe.views.utils import _make_gp_from_gp_info
from moe.views.gp_pretty_view import GpPrettyView

import colander
from moe.views.schemas import ListOfFloats, MatrixOfFloats, GpInfo, ListOfPointsInDomain


class GpMeanVarRequest(colander.MappingSchema):

    """A gp_mean_var request colander schema."""

    points_to_sample = ListOfPointsInDomain()
    gp_info = GpInfo()


class GpMeanVarResponse(colander.MappingSchema):

    """A gp_mean_var response colander schema."""

    endpoint = colander.SchemaNode(colander.String())
    mean = ListOfFloats()
    var = MatrixOfFloats()


class GpMeanVarView(GpPrettyView):

    """Views for gp_mean_var endpoints."""

    route_name = 'gp_mean_var'
    pretty_route_name = 'gp_mean_var_pretty'

    request_schema = GpMeanVarRequest()
    response_schema = GpMeanVarResponse()

    pretty_default_request = {
            "points_to_sample": [
                [0.1], [0.5], [0.9],
                ],
            "gp_info": GpPrettyView.pretty_default_gp_info,
            }

    @view_config(route_name=pretty_route_name, renderer=GpPrettyView.pretty_renderer)
    def pretty_view(self):
        """A pretty, browser interactive view for the interface. Includes form request and response."""
        return self.pretty_response()

    @view_config(route_name=route_name, renderer='json', request_method='POST')
    def gp_mean_var_view(self):
        """Endpoint for gp_mean_var POST requests."""
        params = self.get_params_from_request()

        points_to_sample = numpy.array(params.get('points_to_sample'))
        gp_info = params.get('gp_info')

        GP = _make_gp_from_gp_info(gp_info)

        mean, var = GP.get_mean_and_var_of_points(points_to_sample)

        return self.form_response({
                'endpoint': 'gp_mean_var',
                'mean': mean.tolist(),
                'var': var.tolist(),
                })
