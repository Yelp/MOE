# -*- coding: utf-8 -*-
"""Classes for gp_ei endpoints.

Includes:
    1. request and response schemas
    2. pretty and backend views
"""
import colander

import numpy

from pyramid.view import view_config

from moe.optimal_learning.python.constant import DEFAULT_EXPECTED_IMPROVEMENT_MC_ITERATIONS
from moe.optimal_learning.python.cpp_wrappers.expected_improvement import ExpectedImprovement
from moe.views.constant import GP_EI_ROUTE_NAME, GP_EI_PRETTY_ROUTE_NAME
from moe.views.gp_pretty_view import GpPrettyView, PRETTY_RENDERER
from moe.views.schemas import ListOfPointsInDomain, GpHistoricalInfo, ListOfExpectedImprovements, CovarianceInfo, DomainInfo
from moe.views.utils import _make_gp_from_params


class GpEiRequest(colander.MappingSchema):

    """A gp_ei request colander schema.

    **Required fields**

        :points_to_evaluate: list of points in domain to calculate Expected Improvement (EI) at (:class:`moe.views.schemas.ListOfPointsInDomain`)
        :gp_historical_info: a :class:`moe.views.schemas.GpHistoricalInfo` object of historical data

    **Optional fields**

        :points_being_sampled: list of points in domain being sampled in concurrent experiments (default: []) (:class:`moe.views.schemas.ListOfPointsInDomain`)
        :mc_iterations: number of Monte Carlo (MC) iterations to perform in numerical integration to calculate EI
        :covariance_info: a :class:`moe.views.schemas.CovarianceInfo` dict of covariance information

    **Example Minimal Request**

    .. sourcecode:: http

        Content-Type: text/javascript

        {
            'points_to_evaluate': [[0.1], [0.5], [0.9]],
            'gp_historical_info': {
                'points_sampled': [
                        {'value_var': 0.01, 'value': 0.1, 'point': [0.0]},
                        {'value_var': 0.01, 'value': 0.2, 'point': [1.0]}
                    ],
                },
            'domain_info': {
                'dim': 1,
                },
        }

    **Example Full Request**

    .. sourcecode:: http

        Content-Type: text/javascript

        {
            'points_to_evaluate': [[0.1], [0.5], [0.9]],
            'points_being_sampled': [[0.2], [0.7]],
            'mc_iterations': 10000,
            'gp_historical_info': {
                'points_sampled': [
                        {'value_var': 0.01, 'value': 0.1, 'point': [0.0]},
                        {'value_var': 0.01, 'value': 0.2, 'point': [1.0]}
                    ],
                },
            'domain_info': {
                'domain_type': 'tensor_product'
                'dim': 1,
                },
            'covariance_info': {
                'covariance_type': 'square_exponential',
                'hyperparameters': [1.0, 1.0],
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
            missing=DEFAULT_EXPECTED_IMPROVEMENT_MC_ITERATIONS,
            )
    gp_historical_info = GpHistoricalInfo()
    domain_info = DomainInfo()
    covariance_info = CovarianceInfo(
            missing=CovarianceInfo().deserialize({}),
            )


class GpEiResponse(colander.MappingSchema):

    """A gp_ei response colander schema.

    **Output fields**

        :endpoint: the endpoint that was called
        :expected_improvement: list of calculated expected improvements (:class:`moe.views.schemas.ListOfExpectedImprovements`)

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

    _route_name = GP_EI_ROUTE_NAME
    _pretty_route_name = GP_EI_PRETTY_ROUTE_NAME

    request_schema = GpEiRequest()
    response_schema = GpEiResponse()

    _pretty_default_request = {
            "points_to_evaluate": [
                [0.1], [0.5], [0.9],
                ],
            "gp_historical_info": GpPrettyView._pretty_default_gp_historical_info,
            "covariance_info": GpPrettyView._pretty_default_covariance_info,
            "domain_info": GpPrettyView._pretty_default_domain_info,
            }

    @view_config(route_name=_pretty_route_name, renderer=PRETTY_RENDERER)
    def pretty_view(self):
        """A pretty, browser interactive view for the interface. Includes form request and response.

        .. http:get:: /gp/ei/pretty

        """
        return self.pretty_response()

    @view_config(route_name=_route_name, renderer='json', request_method='POST')
    def gp_ei_view(self):
        """Endpoint for gp_ei POST requests.

        .. http:post:: /gp/ei

           Calculates the Expected Improvement (EI) of a set of points, given historical data.

           :input: :class:`moe.views.gp_ei.GpEiRequest`
           :output: :class:`moe.views.gp_ei.GpEiResponse`

           :status 201: returns a response
           :status 500: server error

        """
        params = self.get_params_from_request()

        # TODO(GH-99): Change REST interface to give points_to_evaluate with shape
        # (num_to_evaluate, num_to_sample, dim)
        # Here we assume the shape is (num_to_evaluate, dim) so we insert an axis, making num_to_sample = 1.
        points_to_evaluate = numpy.array(params.get('points_to_evaluate'))[:, numpy.newaxis, :]
        points_being_sampled = numpy.array(params.get('points_being_sampled'))
        gaussian_process = _make_gp_from_params(params)

        expected_improvement_evaluator = ExpectedImprovement(
                gaussian_process,
                points_being_sampled=points_being_sampled,
                )

        expected_improvement = expected_improvement_evaluator.evaluate_at_point_list(
                points_to_evaluate,
                )

        return self.form_response({
                'endpoint': self._route_name,
                'expected_improvement': expected_improvement.tolist(),
                })
