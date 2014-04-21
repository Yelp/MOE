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
from moe.views.constant import GP_MEAN_VAR_ROUTE_NAME, GP_MEAN_VAR_PRETTY_ROUTE_NAME


class GpMeanVarRequest(colander.MappingSchema):

    """A gp_mean_var request colander schema.

    **Required fields**

        :points_to_sample: list of points in domain to calculate the Gaussian Process (GP) mean and covariance a
        :gp_info: a GpInfo object of historical data

    **Example Request**

    .. sourcecode:: http

        Content-Type: text/javascrip

        {
            'points_to_sample': [[0.1], [0.5], [0.9]],
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

    points_to_sample = ListOfPointsInDomain()
    gp_info = GpInfo()


class GpMeanVarResponse(colander.MappingSchema):

    """A gp_mean_var response colander schema.

    **Output fields**

        :endpoint: the endpoint that was called
        :mean: list of the means of the GP at the points sampled
        :variance: matrix of covariance of the GP at the points sampled

    **Example Response**

    .. sourcecode:: http

        {
            "endpoint":"gp_mean_var",
            "mean": ["0.0873832198661","0.0130505261903","0.174755506336"],
            "var": [
                    ["0.228910114429","0.0969433771923","0.000268292907969"],
                    ["0.0969433771923","0.996177332647","0.0969433771923"],
                    ["0.000268292907969","0.0969433771923","0.228910114429"]
                ],
        }

    """

    endpoint = colander.SchemaNode(colander.String())
    mean = ListOfFloats()
    var = MatrixOfFloats()


class GpMeanVarView(GpPrettyView):

    """Views for gp_mean_var endpoints."""

    route_name = GP_MEAN_VAR_ROUTE_NAME
    pretty_route_name = GP_MEAN_VAR_PRETTY_ROUTE_NAME

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
        """A pretty, browser interactive view for the interface. Includes form request and response.

        .. http:get:: /gp/mean_var/pretty

        """
        return self.pretty_response()

    @view_config(route_name=route_name, renderer='json', request_method='POST')
    def gp_mean_var_view(self):
        """Endpoint for gp_mean_var POST requests.

        .. http:post:: /gp/mean_var

           Calculates the GP mean and covariance of a set of points, given historical data.

           :input: moe.views.gp_ei.GpMeanVarRequest()
           :output: moe.views.gp_ei.GpMeanVarResponse()

           :status 200: returns a response
           :status 500: server error

        """
        params = self.get_params_from_request()

        points_to_sample = numpy.array(params.get('points_to_sample'))
        gp_info = params.get('gp_info')

        GP = _make_gp_from_gp_info(gp_info)

        mean, var = GP.get_mean_and_var_of_points(points_to_sample)

        return self.form_response({
                'endpoint': self.route_name,
                'mean': mean.tolist(),
                'var': var.tolist(),
                })
