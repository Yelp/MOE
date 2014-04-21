# -*- coding: utf-8 -*-
"""Classes for gp_ei endpoints.

Includes:
    1. request and response schemas
    2. pretty and backend views
"""
import colander
import numpy
from pyramid.view import view_config

from moe.optimal_learning.EPI.src.python.constant import default_expected_improvement_parameters
from moe.views.gp_pretty_view import GpPrettyView
from moe.views.schemas import ListOfPointsInDomain, GpInfo, ListOfExpectedImprovements
from moe.views.utils import _make_gp_from_gp_info
from moe.views.constant import GP_EI_ROUTE_NAME, GP_EI_PRETTY_ROUTE_NAME


class GpEiRequest(colander.MappingSchema):

    """A gp_ei request colander schema.

    **Required fields**

        :points_to_evaluate: list of points in domain to calculate Expected Improvement (EI) a
        :gp_info: a GpInfo object of historical data

    **Optional fields**

        :points_being_sampled: list of points in domain being sampled (default: [])
        :mc_iterations: number of Monte Carlo (MC) iterations to perform in numerical integration to calculate EI (default: 1000)

    **Example Request**

    .. sourcecode:: http

        Content-Type: text/javascrip

        {
            'points_to_evaluate': [[0.1], [0.5], [0.9]],
            'points_being_sampled': [],
            'mc_iterations': 1000,
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

    """A gp_ei response colander schema.

    **Output fields**

        :endpoint: the endpoint that was called
        :expected_improvement: list of calculated expected improvements

    **Example Response**

    .. sourcecode:: http

        {
            "endpoint":"gp_ei",
            "expected_improvement":["0.197246898375","0.443163755117","0.155819546878"]
        }

    """

    endpoint = colander.SchemaNode(colander.String())
    expected_improvement = ListOfExpectedImprovements()


class GpEiView(GpPrettyView):

    """Views for gp_ei endpoints."""

    route_name = GP_EI_ROUTE_NAME
    pretty_route_name = GP_EI_PRETTY_ROUTE_NAME

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
        """A pretty, browser interactive view for the interface. Includes form request and response.

        .. http:get:: /gp/ei/pretty

        """
        return self.pretty_response()

    @view_config(route_name=route_name, renderer='json', request_method='POST')
    def gp_ei_view(self):
        """Endpoint for gp_ei POST requests.

        .. http:post:: /gp/ei

           Calculates the Expected Improvement (EI) of a set of points, given historical data.

           :input: moe.views.gp_ei.GpEiRequest()
           :output: moe.views.gp_ei.GpEiResponse()

           :status 200: returns a response
           :status 500: server error

        """
        params = self.get_params_from_request()

        points_to_evaluate = numpy.array(params.get('points_to_evaluate'))
        points_being_sampled = numpy.array(params.get('points_being_sampled'))
        gp_info = params.get('gp_info')

        GP = _make_gp_from_gp_info(gp_info)

        expected_improvement = GP.evaluate_expected_improvement_at_point_list(
                points_to_evaluate,
                points_being_sampled=points_being_sampled,
                )

        return self.form_response({
                'endpoint': self.route_name,
                'expected_improvement': expected_improvement.tolist(),
                })
