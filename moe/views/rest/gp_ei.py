# -*- coding: utf-8 -*-
"""Classes for gp_ei endpoints.

Includes:
    1. request and response schemas
    2. pretty and backend views
"""
import numpy

from pyramid.view import view_config

from moe.optimal_learning.python.cpp_wrappers.expected_improvement import ExpectedImprovement
from moe.views.constant import GP_EI_ROUTE_NAME, GP_EI_PRETTY_ROUTE_NAME
from moe.views.gp_pretty_view import GpPrettyView, PRETTY_RENDERER
from moe.views.schemas import GpEiRequest, GpEiResponse
from moe.views.utils import _make_gp_from_params


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
        num_mc_iterations = params.get('mc_iterations')
        max_num_threads = params.get('max_num_threads')
        gaussian_process = _make_gp_from_params(params)

        expected_improvement_evaluator = ExpectedImprovement(
                gaussian_process,
                points_being_sampled=points_being_sampled,
                num_mc_iterations=num_mc_iterations,
                )

        expected_improvement = expected_improvement_evaluator.evaluate_at_point_list(
                points_to_evaluate,
                max_num_threads=max_num_threads,
                )

        return self.form_response({
                'endpoint': self._route_name,
                'expected_improvement': expected_improvement.tolist(),
                })
