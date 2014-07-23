# -*- coding: utf-8 -*-
"""Classes for gp_next_points_constant_liar endpoints.

Includes:

    1. pretty and backend views

Support for:

    * `'constant_liar_min'` - `lie_value` is equal to the *min* of all points sampled so far
    * `'constant_liar_max'` - `lie_value` is equal to the *max* of all points sampled so far
    * `'constant_liar_mean'` - `lie_value` is equal to the *mean* of all points sampled so far
"""
import numpy

from pyramid.view import view_config

from moe.optimal_learning.python.constant import CONSTANT_LIAR_MIN, CONSTANT_LIAR_MAX, CONSTANT_LIAR_MEAN, DEFAULT_CONSTANT_LIAR_METHOD
from moe.views.constant import GP_NEXT_POINTS_CONSTANT_LIAR_ROUTE_NAME, GP_NEXT_POINTS_CONSTANT_LIAR_PRETTY_ROUTE_NAME, GP_NEXT_POINTS_CONSTANT_LIAR_OPTIMIZER_METHOD_NAME
from moe.views.gp_next_points_pretty_view import GpNextPointsPrettyView
from moe.views.pretty_view import PRETTY_RENDERER
from moe.views.schemas import GpNextPointsConstantLiarRequest
from moe.views.utils import _make_gp_from_params


class GpNextPointsConstantLiar(GpNextPointsPrettyView):

    """Views for gp_next_points_constant_liar endpoints."""

    _route_name = GP_NEXT_POINTS_CONSTANT_LIAR_ROUTE_NAME
    _pretty_route_name = GP_NEXT_POINTS_CONSTANT_LIAR_PRETTY_ROUTE_NAME

    request_schema = GpNextPointsConstantLiarRequest()

    _pretty_default_request = GpNextPointsPrettyView._pretty_default_request.copy()
    _pretty_default_request['lie_method'] = DEFAULT_CONSTANT_LIAR_METHOD

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
        points_sampled_values = gaussian_process._historical_data._points_sampled_value.tolist()

        if params.get('lie_method') == CONSTANT_LIAR_MIN:
            return numpy.amin(points_sampled_values)
        elif params.get('lie_method') == CONSTANT_LIAR_MAX:
            return numpy.amax(points_sampled_values)
        elif params.get('lie_method') == CONSTANT_LIAR_MEAN:
            return numpy.mean(points_sampled_values)
        else:
            raise(NotImplementedError, '{0} is not implemented'.format(params.get('lie_method')))

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
                GP_NEXT_POINTS_CONSTANT_LIAR_OPTIMIZER_METHOD_NAME,
                self._route_name,
                self.get_lie_value(params),
                lie_noise_variance=params.get('lie_noise_variance'),
                )
