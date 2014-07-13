# -*- coding: utf-8 -*-
"""Classes for gp_next_points_kriging endpoints.

Includes:
    1. pretty and backend views
"""
import colander

from pyramid.view import view_config

from moe.views.constant import GP_NEXT_POINTS_KRIGING_ROUTE_NAME, GP_NEXT_POINTS_KRIGING_PRETTY_ROUTE_NAME, GP_NEXT_POINTS_KRIGING_OPTIMIZATION_METHOD_NAME
from moe.views.gp_next_points_pretty_view import GpNextPointsPrettyView, GpNextPointsRequest
from moe.views.gp_pretty_view import PRETTY_RENDERER


class GpNextPointsKrigingRequest(GpNextPointsRequest):

    """Extends the standard request :class:`moe.views.gp_next_points_pretty_view.GpNextPointsRequest` with kriging parameters.

    **Required fields**

        :gp_historical_info: a :class:`moe.views.schemas.GpHistoricalInfo` dict of historical data
        :domain_info: a :class:`moe.views.schemas.BoundedDomainInfo` dict of domain information

    **Optional fields**

        :num_to_sample: number of next points to generate (default: 1)
        :std_deviation_coef: a float used in Kriging, see Kriging implementation docs (default: 0.0)
        :kriging_noise_variance: a positive (>= 0) float used in Kriging, see Kriging implementation docs (default: 0.0)
        :covariance_info: a :class:`moe.views.schemas.CovarianceInfo` dict of covariance information
        :optimiaztion_info: a :class:`moe.views.schemas.OptimizationInfo` dict of optimization information

    **Example Request**

    .. sourcecode:: http

        Content-Type: text/javascript

        {
            "num_to_sample": 1,
            "std_deviation_coef": 0.0,
            "kriging_noise_variance": 0.0,
            "gp_historical_info": {
                "points_sampled": [
                        {"value_var": 0.01, "value": 0.1, "point": [0.0]},
                        {"value_var": 0.01, "value": 0.2, "point": [1.0]}
                    ],
                },
            "domain_info": {
                "dim": 1,
                "domain_bounds": [
                    {"min": 0.0, "max": 1.0},
                    ],
                },
        }

    """

    std_deviation_coef = colander.SchemaNode(
            colander.Float(),
            missing=0.0,
            )
    # TODO(GH-257): Find a better value for missing here.
    kriging_noise_variance = colander.SchemaNode(
            colander.Float(),
            missing=1e-8,
            validator=colander.Range(min=0.0),
            )


class GpNextPointsKriging(GpNextPointsPrettyView):

    """Views for gp_next_points_kriging endpoints."""

    _route_name = GP_NEXT_POINTS_KRIGING_ROUTE_NAME
    _pretty_route_name = GP_NEXT_POINTS_KRIGING_PRETTY_ROUTE_NAME

    request_schema = GpNextPointsKrigingRequest()

    _pretty_default_request = GpNextPointsPrettyView._pretty_default_request.copy()
    _pretty_default_request['std_deviation_coef'] = 0.0
    _pretty_default_request['kriging_noise_variance'] = 1e-8

    @view_config(route_name=_pretty_route_name, renderer=PRETTY_RENDERER)
    def pretty_view(self):
        """A pretty, browser interactive view for the interface. Includes form request and response."""
        return self.pretty_response()

    @view_config(route_name=_route_name, renderer='json', request_method='POST')
    def gp_next_points_kriging_view(self):
        """Endpoint for gp_next_points_kriging POST requests.

        .. http:post:: /gp/next_points/kriging

           Calculates the next best points to sample, given historical data, using Kriging.

           :input: :class:`moe.views.rest.gp_next_points_kriging.GpNextPointsKrigingRequest`
           :output: :class:`moe.views.gp_next_points_pretty_view.GpNextPointsResponse`

           :status 200: returns a response
           :status 500: server error

        """
        params = self.get_params_from_request()
        return self.compute_next_points_to_sample_response(
                params,
                GP_NEXT_POINTS_KRIGING_OPTIMIZATION_METHOD_NAME,
                self._route_name,
                std_deviation_coef=params.get('std_deviation_coef'),
                kriging_noise_variance=params.get('kriging_noise_variance'),
                )
