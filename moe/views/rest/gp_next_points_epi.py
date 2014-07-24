# -*- coding: utf-8 -*-
"""Classes for gp_next_points_epi endpoints.

Includes:

    1. pretty and backend views

"""
from pyramid.view import view_config

from moe.optimal_learning.python.constant import OPTIMIZER_TYPE_AND_OBJECTIVE_TO_DEFAULT_PARAMETERS, EI_COMPUTE_TYPE_ANALYTIC, EI_COMPUTE_TYPE_MONTE_CARLO
from moe.views.constant import GP_NEXT_POINTS_EPI_ROUTE_NAME, GP_NEXT_POINTS_EPI_PRETTY_ROUTE_NAME, GP_NEXT_POINTS_EPI_OPTIMIZER_METHOD_NAME
from moe.views.gp_next_points_pretty_view import GpNextPointsPrettyView
from moe.views.pretty_view import PRETTY_RENDERER


class GpNextPointsEpi(GpNextPointsPrettyView):

    """Views for gp_next_points_epi endpoints."""

    _route_name = GP_NEXT_POINTS_EPI_ROUTE_NAME
    _pretty_route_name = GP_NEXT_POINTS_EPI_PRETTY_ROUTE_NAME

    def _get_default_optimizer_params(self, params):
        """Get the default optimizer parameters associated with the desired ``optimizer_type``, REST endpoint, and analytic vs monte carlo computation.

        :param params: a (partially) deserialized REST request with everything except possibly
          ``params['optimizer_info']``
        :type params: dict
        :return: default multistart and optimizer parameters to use with this REST request
        :rtype: :class:`moe.optimal_learning.python.constant.DefaultOptimizerInfoTuple`

        """
        optimizer_type = params['optimizer_info']['optimizer_type']
        optimizer_parameters_lookup = (optimizer_type, self._route_name)

        num_to_sample = params.get('num_to_sample')
        if num_to_sample == 1:
            optimizer_parameters_lookup += (EI_COMPUTE_TYPE_ANALYTIC, )
        else:
            optimizer_parameters_lookup += (EI_COMPUTE_TYPE_MONTE_CARLO, )

        return OPTIMIZER_TYPE_AND_OBJECTIVE_TO_DEFAULT_PARAMETERS[optimizer_parameters_lookup]

    @view_config(route_name=_pretty_route_name, renderer=PRETTY_RENDERER)
    def pretty_view(self):
        """A pretty, browser interactive view for the interface. Includes form request and response.

        .. http:get:: /gp/next_points/epi/pretty

        """
        return self.pretty_response()

    @view_config(route_name=_route_name, renderer='json', request_method='POST')
    def gp_next_points_epi_view(self):
        """Endpoint for gp_next_points_epi POST requests.

        .. http:post:: /gp/next_points/epi

           Calculates the next best points to sample, given historical data, using Expected Parallel Improvement (EPI).

           :input: :class:`moe.views.schemas.GpNextPointsRequest`
           :output: :class:`moe.views.schemas.GpNextPointsResponse`

           :status 200: returns a response
           :status 500: server error

        """
        params = self.get_params_from_request()
        return self.compute_next_points_to_sample_response(
                params,
                GP_NEXT_POINTS_EPI_OPTIMIZER_METHOD_NAME,
                self._route_name,
                )
