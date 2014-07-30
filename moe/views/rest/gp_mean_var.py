# -*- coding: utf-8 -*-
"""Classes for gp_mean, gp_var, and gp_mean_var (and _diag) endpoints.

Includes:
    1. request and response schemas
    2. pretty and backend views
"""
import numpy

from pyramid.view import view_config

from moe.optimal_learning.python.timing import timing_context
from moe.views.constant import GP_MEAN_ROUTE_NAME, GP_MEAN_PRETTY_ROUTE_NAME, GP_VAR_ROUTE_NAME, GP_VAR_PRETTY_ROUTE_NAME, GP_VAR_DIAG_ROUTE_NAME, GP_VAR_DIAG_PRETTY_ROUTE_NAME, GP_MEAN_VAR_ROUTE_NAME, GP_MEAN_VAR_PRETTY_ROUTE_NAME, GP_MEAN_VAR_DIAG_ROUTE_NAME, GP_MEAN_VAR_DIAG_PRETTY_ROUTE_NAME
from moe.views.gp_pretty_view import GpPrettyView, PRETTY_RENDERER
from moe.views.schemas.rest.gp_mean_var import GpMeanVarRequest, GpMeanVarResponse, GpMeanVarDiagResponse, GpMeanResponse, GpVarResponse, GpVarDiagResponse
from moe.views.utils import _make_gp_from_params


MEAN_VAR_COMPUTATION_TIMING_LABEL = 'mean/var computation time'


class GpMeanVarBaseView(GpPrettyView):

    """Base View class for gp_mean, gp_var, gp_mean_var, and _diag endpoints."""

    request_schema = GpMeanVarRequest()

    _pretty_default_request = {
            "points_to_evaluate": [
                [0.1], [0.5], [0.9],
                ],
            "gp_historical_info": GpPrettyView._pretty_default_gp_historical_info,
            "covariance_info": GpPrettyView._pretty_default_covariance_info,
            "domain_info": GpPrettyView._pretty_default_domain_info,
            }

    def gp_mean_var_response_dict(self, compute_mean=False, compute_var=False, var_diag=False):
        """Produce a response dict (to be serialized) for gp_mean, gp_var, gp_mean_var, and _diag POST requests.

        :param compute_mean: whether to compute the GP mean
        :type compute_mean: bool
        :param compute_var: whether to compute the GP variance or covariance
        :type compute_var: bool
        :param var_diag: whether to compute the full GP covariance or just the variance terms
        :type var_diag: bool
        :return: dict with 'endpoint' and optionally 'mean' and 'var' keys depending on inputs
        :rtype: dict

        """
        params = self.get_params_from_request()

        points_to_evaluate = numpy.array(params.get('points_to_evaluate'))
        gaussian_process = _make_gp_from_params(params)

        response_dict = {}
        response_dict['endpoint'] = self._route_name

        with timing_context(MEAN_VAR_COMPUTATION_TIMING_LABEL):
            if compute_mean:
                response_dict['mean'] = gaussian_process.compute_mean_of_points(points_to_evaluate).tolist()

            if compute_var:
                if var_diag:
                    response_dict['var'] = numpy.diag(
                        gaussian_process.compute_variance_of_points(points_to_evaluate)
                    ).tolist()
                else:
                    response_dict['var'] = gaussian_process.compute_variance_of_points(points_to_evaluate).tolist()

        return response_dict


class GpMeanVarView(GpMeanVarBaseView):

    """Views for gp_mean_var endpoints."""

    _route_name = GP_MEAN_VAR_ROUTE_NAME
    _pretty_route_name = GP_MEAN_VAR_PRETTY_ROUTE_NAME

    response_schema = GpMeanVarResponse()

    @view_config(route_name=_pretty_route_name, renderer=PRETTY_RENDERER)
    def pretty_view(self):
        """A pretty, browser interactive view for the interface. Includes form request and response.

        .. http:get:: /gp/mean_var/pretty

        """
        return self.pretty_response()

    @view_config(route_name=_route_name, renderer='json', request_method='POST')
    def gp_mean_var_view(self):
        """Endpoint for gp_mean_var POST requests.

        .. http:post:: /gp/mean_var

           Calculates the GP mean and covariance of a set of points, given historical data.

           :input: :class:`moe.views.schemas.rest.gp_mean_var.GpMeanVarRequest`
           :output: :class:`moe.views.schemas.rest.gp_mean_var.GpMeanVarResponse`

           :status 200: returns a response
           :status 500: server error

        """
        return self.form_response(
            self.gp_mean_var_response_dict(compute_mean=True, compute_var=True, var_diag=False)
        )


class GpMeanVarDiagView(GpMeanVarBaseView):

    """Views for gp_mean_var_diag endpoints."""

    _route_name = GP_MEAN_VAR_DIAG_ROUTE_NAME
    _pretty_route_name = GP_MEAN_VAR_DIAG_PRETTY_ROUTE_NAME

    response_schema = GpMeanVarDiagResponse()

    @view_config(route_name=_pretty_route_name, renderer=PRETTY_RENDERER)
    def pretty_view(self):
        """A pretty, browser interactive view for the interface. Includes form request and response.

        .. http:get:: /gp/mean_var/diag/pretty

        """
        return self.pretty_response()

    @view_config(route_name=_route_name, renderer='json', request_method='POST')
    def gp_mean_var_diag_view(self):
        """Endpoint for gp_mean_var_diag POST requests.

        .. http:post:: /gp/mean_var/diag

           Calculates the GP mean and variance of a set of points, given historical data.

           :input: :class:`moe.views.schemas.rest.gp_mean_var.GpMeanVarRequest`
           :output: :class:`moe.views.schemas.rest.gp_mean_var.GpMeanVarDiagResponse`

           :status 200: returns a response
           :status 500: server error

        """
        return self.form_response(
            self.gp_mean_var_response_dict(compute_mean=True, compute_var=True, var_diag=True)
        )


class GpMeanView(GpMeanVarBaseView):

    """Views for gp_mean_var endpoints."""

    _route_name = GP_MEAN_ROUTE_NAME
    _pretty_route_name = GP_MEAN_PRETTY_ROUTE_NAME

    response_schema = GpMeanResponse()

    @view_config(route_name=_pretty_route_name, renderer=PRETTY_RENDERER)
    def pretty_view(self):
        """A pretty, browser interactive view for the interface. Includes form request and response.

        .. http:get:: /gp/mean/pretty

        """
        return self.pretty_response()

    @view_config(route_name=_route_name, renderer='json', request_method='POST')
    def gp_mean_view(self):
        """Endpoint for gp_mean POST requests.

        .. http:post:: /gp/mean

           Calculates the GP mean of a set of points, given historical data.

           :input: :class:`moe.views.schemas.rest.gp_mean_var.GpMeanVarRequest`
           :output: :class:`moe.views.schemas.rest.gp_mean_var.GpMeanResponse`

           :status 200: returns a response
           :status 500: server error

        """
        return self.form_response(
            self.gp_mean_var_response_dict(compute_mean=True, compute_var=False)
        )


class GpVarView(GpMeanVarBaseView):

    """Views for gp_var endpoints."""

    _route_name = GP_VAR_ROUTE_NAME
    _pretty_route_name = GP_VAR_PRETTY_ROUTE_NAME

    response_schema = GpVarResponse()

    @view_config(route_name=_pretty_route_name, renderer=PRETTY_RENDERER)
    def pretty_view(self):
        """A pretty, browser interactive view for the interface. Includes form request and response.

        .. http:get:: /gp/var/pretty

        """
        return self.pretty_response()

    @view_config(route_name=_route_name, renderer='json', request_method='POST')
    def gp_var_view(self):
        """Endpoint for gp_var POST requests.

        .. http:post:: /gp/var

           Calculates the GP covariance of a set of points, given historical data.

           :input: :class:`moe.views.schemas.rest.gp_mean_var.GpMeanVarRequest`
           :output: :class:`moe.views.schemas.rest.gp_mean_var.GpVarResponse`

           :status 200: returns a response
           :status 500: server error

        """
        return self.form_response(
            self.gp_mean_var_response_dict(compute_mean=False, compute_var=True, var_diag=False)
        )


class GpVarDiagView(GpMeanVarBaseView):

    """Views for gp_var_diag endpoints."""

    _route_name = GP_VAR_DIAG_ROUTE_NAME
    _pretty_route_name = GP_VAR_DIAG_PRETTY_ROUTE_NAME

    response_schema = GpVarDiagResponse()

    @view_config(route_name=_pretty_route_name, renderer=PRETTY_RENDERER)
    def pretty_view(self):
        """A pretty, browser interactive view for the interface. Includes form request and response.

        .. http:get:: /gp/var/diag/pretty

        """
        return self.pretty_response()

    @view_config(route_name=_route_name, renderer='json', request_method='POST')
    def gp_var_diag_view(self):
        """Endpoint for gp_var_diag POST requests.

        .. http:post:: /gp/var/diag

           Calculates the GP variance of a set of points, given historical data.

           :input: :class:`moe.views.schemas.rest.gp_mean_var.GpMeanVarRequest`
           :output: :class:`moe.views.schemas.rest.gp_mean_var.GpVarDiagResponse`

           :status 200: returns a response
           :status 500: server error

        """
        return self.form_response(
            self.gp_mean_var_response_dict(compute_mean=False, compute_var=True, var_diag=True)
        )
