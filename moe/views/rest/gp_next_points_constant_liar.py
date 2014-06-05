# -*- coding: utf-8 -*-
"""Classes for gp_next_points_constant_liar endpoints.

Includes:

    1. pretty and backend views

Support for:

    * `'constant_liar_min'` - `lie_value` is equal to the *min* of all points sampled so far
    * `'constant_liar_max'` - `lie_value` is equal to the *max* of all points sampled so far
    * `'constant_liar_mean'` - `lie_value` is equal to the *mean* of all points sampled so far
"""
import colander

import numpy

from pyramid.view import view_config

from moe.views.constant import GP_NEXT_POINTS_CONSTANT_LIAR_ROUTE_NAME, GP_NEXT_POINTS_CONSTANT_LIAR_PRETTY_ROUTE_NAME, GP_NEXT_POINTS_CONSTANT_LIAR_OPTIMIZATION_METHOD_NAME
from moe.views.gp_next_points_pretty_view import GpNextPointsPrettyView, GpNextPointsRequest
from moe.views.gp_pretty_view import PRETTY_RENDERER
from moe.views.utils import _make_gp_from_params


CONSTANT_LIAR_MIN = 'constant_liar_min'
CONSTANT_LIAR_MAX = 'constant_liar_max'
CONSTANT_LIAR_MEAN = 'constant_liar_mean'

CONSTANT_LIAR_METHODS = [
        CONSTANT_LIAR_MIN,
        CONSTANT_LIAR_MAX,
        CONSTANT_LIAR_MEAN,
        ]


class GpNextPointsConstantLiarRequest(GpNextPointsRequest):

    """Extends the standard request :class:`moe.views.gp_next_points_pretty_view.GpNextPointsRequest` with a lie value.

    **Required fields**

        :gp_historical_info: a :class:`moe.views.schemas.GpHistoricalInfo` dict of historical data
        :domain_info: a :class:`moe.views.schemas.BoundedDomainInfo` dict of domain information

    **Optional fields**

        :num_to_sample: number of next points to generate (default: 1)
        :lie_method: a string from `CONSTANT_LIAR_METHODS` representing the liar method to use (default: 'constant_liar_min')
        :lie_value: a float representing the 'lie' the Constant Liar heuristic will use (default: None). If `lie_value` is not None the algorithm will use this value instead of one calculated using `lie_method`.
        :lie_noise_variance: a positive (>= 0) float representing the noise variance of the 'lie' value (default: 0.0)
        :covariance_info: a :class:`moe.views.schemas.CovarianceInfo` dict of covariance information
        :optimiaztion_info: a :class:`moe.views.schemas.OptimizationInfo` dict of optimization information

    **Example Request**

    .. sourcecode:: http

        Content-Type: text/javascript

        {
            'num_to_sample': 1,
            'lie_value': 0.0,
            'lie_noise_variance': 0.0,
            'gp_historical_info': {
                'points_sampled': [
                        {'value_var': 0.01, 'value': 0.1, 'point': [0.0]},
                        {'value_var': 0.01, 'value': 0.2, 'point': [1.0]}
                    ],
                },
            'domain_info': {
                'dim': 1,
                'domain_bounds': [
                    {'min': 0.0, 'max': 1.0},
                    ],
                },
        }

    """

    lie_method = colander.SchemaNode(
            colander.String(),
            missing=CONSTANT_LIAR_MIN,
            validator=colander.OneOf(CONSTANT_LIAR_METHODS),
            )
    lie_value = colander.SchemaNode(
            colander.Float(),
            missing=None,
            )
    lie_noise_variance = colander.SchemaNode(
            colander.Float(),
            missing=0.0,
            validator=colander.Range(min=0.0),
            )


class GpNextPointsConstantLiar(GpNextPointsPrettyView):

    """Views for gp_next_points_constant_liar endpoints."""

    _route_name = GP_NEXT_POINTS_CONSTANT_LIAR_ROUTE_NAME
    _pretty_route_name = GP_NEXT_POINTS_CONSTANT_LIAR_PRETTY_ROUTE_NAME

    request_schema = GpNextPointsConstantLiarRequest()

    _pretty_default_request = GpNextPointsPrettyView._pretty_default_request.copy()
    _pretty_default_request['lie_method'] = CONSTANT_LIAR_MIN
    _pretty_default_request['lie_value'] = None
    _pretty_default_request['lie_noise_variance'] = 0.0

    @view_config(route_name=_pretty_route_name, renderer=PRETTY_RENDERER)
    def pretty_view(self):
        """A pretty, browser interactive view for the interface. Includes form request and response.

        .. http:get:: /gp/next_points/constant_liar/pretty

        """
        return self.pretty_response()

    def get_lie_value(self, params):
        """Return the lie value associated with the lie_method, unless lie_value is explicitly given."""
        if params.get('lie_value') is not None:
            return params.get('lie_value')

        gaussian_process = _make_gp_from_params(params)
        points_sampled_values = gaussian_process._historical_data._points_sampled_value.to_list()

        if params.get('lie_method') == CONSTANT_LIAR_MIN:
            return numpy.amin(points_sampled_values)
        elif params.get('lie_method') == CONSTANT_LIAR_MAX:
            return numpy.amax(points_sampled_values)
        elif params.get('lie_method') == CONSTANT_LIAR_MEAN:
            return numpy.mean(points_sampled_values)
        else:
            raise(NotImplementedError, '%s is not implemented' % params.get('lie_method'))

    @view_config(route_name=_route_name, renderer='json', request_method='POST')
    def gp_next_points_constant_liar_view(self):
        """Endpoint for gp_next_points_constant_liar POST requests.

        .. http:post:: /gp/next_points/constant_liar

           Calculates the next best points to sample, given historical data, using Constant Liar (CL).

           :input: :class:`moe.views.rest.gp_next_points_constant_liar.GpNextPointsConstantLiarRequest`
           :output: :class:`moe.views.gp_next_points_pretty_view.GpNextPointsResponse`

           :status 200: returns a response
           :status 500: server error

        """
        params = self.get_params_from_request()
        return self.compute_next_points_to_sample_response(
                params,
                GP_NEXT_POINTS_CONSTANT_LIAR_OPTIMIZATION_METHOD_NAME,
                self._route_name,
                self.get_lie_value(params),
                lie_noise_variance=params.get('lie_noise_variance'),
                )
