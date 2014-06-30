# -*- coding: utf-8 -*-
"""Classes for gp_mean, gp_var, and gp_mean_var (and _diag) endpoints.

Includes:
    1. request and response schemas
    2. pretty and backend views
"""
import colander

import numpy

from pyramid.view import view_config

from moe.views.constant import GP_MEAN_ROUTE_NAME, GP_MEAN_PRETTY_ROUTE_NAME, GP_VAR_ROUTE_NAME, GP_VAR_PRETTY_ROUTE_NAME, GP_VAR_DIAG_ROUTE_NAME, GP_VAR_DIAG_PRETTY_ROUTE_NAME, GP_MEAN_VAR_ROUTE_NAME, GP_MEAN_VAR_PRETTY_ROUTE_NAME, GP_MEAN_VAR_DIAG_ROUTE_NAME, GP_MEAN_VAR_DIAG_PRETTY_ROUTE_NAME
from moe.views.gp_pretty_view import GpPrettyView, PRETTY_RENDERER
from moe.views.schemas import ListOfFloats, MatrixOfFloats, GpHistoricalInfo, ListOfPointsInDomain, CovarianceInfo, DomainInfo
from moe.views.utils import _make_gp_from_params


class GpMeanVarRequest(colander.MappingSchema):

    """A gp_mean_var request colander schema.

    **Required fields**

        :points_to_sample: list of points in domain to calculate the Gaussian Process (GP) mean and covariance at (:class:`moe.views.schemas.ListOfPointsInDomain`)
        :gp_historical_info: a :class:`moe.views.schemas.GpHistoricalInfo` object of historical data

    **Optional fields**

        :covariance_info: a :class:`moe.views.schemas.CovarianceInfo` dict of covariance information

    **Example Minimal Request**

    .. sourcecode:: http

        Content-Type: text/javascript

        {
            "points_to_sample": [[0.1], [0.5], [0.9]],
            "gp_historical_info": {
                "points_sampled": [
                        {"value_var": 0.01, "value": 0.1, "point": [0.0]},
                        {"value_var": 0.01, "value": 0.2, "point": [1.0]}
                    ],
                },
            "domain_info": {
                "dim": 1,
                },
        }

    **Example Full Request**

    .. sourcecode:: http

        Content-Type: text/javascript

        {
            "points_to_sample": [[0.1], [0.5], [0.9]],
            "gp_historical_info": {
                "points_sampled": [
                        {"value_var": 0.01, "value": 0.1, "point": [0.0]},
                        {"value_var": 0.01, "value": 0.2, "point": [1.0]}
                    ],
                },
            "domain_info": {
                "domain_type": "tensor_product"
                "dim": 1,
                },
            "covariance_info": {
                "covariance_type": "square_exponential",
                "hyperparameters": [1.0, 1.0],
                },
        }

    """

    points_to_sample = ListOfPointsInDomain()
    gp_historical_info = GpHistoricalInfo()
    domain_info = DomainInfo()
    covariance_info = CovarianceInfo(
            missing=CovarianceInfo().deserialize({}),
            )


class GpEndpointResponse(colander.MappingSchema):

    """A base schema for the endpoint name.

    **Output fields**

        :endpoint: the endpoint that was called

    **Example Response**

    .. sourcecode:: http

        {
            "endpoint":"gp_mean_var",
        }

    """

    endpoint = colander.SchemaNode(colander.String())


class GpMeanMixinResponse(colander.MappingSchema):

    """A mixin response colander schema for the mean of a gaussian process.

    **Output fields**

        :mean: list of the means of the GP at ``points_to_sample`` (:class:`moe.views.schemas.ListOfFloats`)

    **Example Response**

    .. sourcecode:: http

        {
            "mean": ["0.0873832198661","0.0130505261903","0.174755506336"],
        }

    """

    mean = ListOfFloats()


class GpVarMixinResponse(colander.MappingSchema):

    """A mixin response colander schema for the [co]variance of a gaussian process.

    **Output fields**

        :variance: matrix of covariance of the GP at ``points_to_sample`` (:class:`moe.views.schemas.MatrixOfFloats`)

    **Example Response**

    .. sourcecode:: http

        {
            "var": [
                    ["0.228910114429","0.0969433771923","0.000268292907969"],
                    ["0.0969433771923","0.996177332647","0.0969433771923"],
                    ["0.000268292907969","0.0969433771923","0.228910114429"]
                ],
        }

    """

    var = MatrixOfFloats()


class GpVarDiagMixinResponse(colander.MappingSchema):

    """A mixin response colander schema for the variance of a gaussian process.

    **Output fields**

        :variance: list of variances of the GP at ``points_to_sample``; i.e., diagonal of the ``variance`` response from gp_mean_var (:class:`moe.views.schemas.ListOfFloats`)

    **Example Response**

    .. sourcecode:: http

        {
            "var": ["0.228910114429","0.996177332647","0.228910114429"],
        }

    """

    var = ListOfFloats()


class GpMeanResponse(GpEndpointResponse, GpMeanMixinResponse):

    """A gp_mean response colander schema.

    **Output fields**

        :endpoint: the endpoint that was called
        :mean: list of the means of the GP at ``points_to_sample`` (:class:`moe.views.schemas.ListOfFloats`)

    **Example Response**

    See composing members' docstrings.

    """

    pass


class GpVarResponse(GpEndpointResponse, GpVarMixinResponse):

    """A gp_var response colander schema.

    **Output fields**

        :endpoint: the endpoint that was called
        :variance: matrix of covariance of the GP at ``points_to_sample`` (:class:`moe.views.schemas.MatrixOfFloats`)

    **Example Response**

    See composing members' docstrings.

    """

    pass


class GpVarDiagResponse(GpEndpointResponse, GpVarDiagMixinResponse):

    """A gp_var_diag response colander schema.

    **Output fields**

        :endpoint: the endpoint that was called
        :variance: list of variances of the GP at ``points_to_sample``; i.e., diagonal of the ``variance`` response from gp_mean_var (:class:`moe.views.schemas.ListOfFloats`)

    **Example Response**

    See composing members' docstrings.

    """

    pass


class GpMeanVarResponse(GpMeanResponse, GpVarMixinResponse):

    """A gp_mean_var response colander schema.

    **Output fields**

        :endpoint: the endpoint that was called
        :mean: list of the means of the GP at ``points_to_sample`` (:class:`moe.views.schemas.ListOfFloats`)
        :variance: matrix of covariance of the GP at ``points_to_sample`` (:class:`moe.views.schemas.MatrixOfFloats`)

    **Example Response**

    See composing members' docstrings.

    """

    pass


class GpMeanVarDiagResponse(GpMeanResponse, GpVarDiagMixinResponse):

    """A gp_mean_var_diag response colander schema.

    **Output fields**

        :endpoint: the endpoint that was called
        :mean: list of the means of the GP at ``points_to_sample`` (:class:`moe.views.schemas.ListOfFloats`)
        :variance: list of variances of the GP at ``points_to_sample``; i.e., diagonal of the ``variance`` response from gp_mean_var (:class:`moe.views.schemas.ListOfFloats`)

    **Example Response**

    .. sourcecode:: http

    See composing members' docstrings.

    """

    pass


class GpMeanVarBaseView(GpPrettyView):

    """Base View class for gp_mean, gp_var, gp_mean_var, and _diag endpoints."""

    request_schema = GpMeanVarRequest()

    _pretty_default_request = {
            "points_to_sample": [
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

        points_to_sample = numpy.array(params.get('points_to_sample'))
        gaussian_process = _make_gp_from_params(params)

        response_dict = {}
        response_dict['endpoint'] = self._route_name
        if compute_mean:
            response_dict['mean'] = gaussian_process.compute_mean_of_points(points_to_sample).tolist()

        if compute_var:
            if var_diag:
                response_dict['var'] = numpy.diag(
                    gaussian_process.compute_variance_of_points(points_to_sample)
                ).tolist()
            else:
                response_dict['var'] = gaussian_process.compute_variance_of_points(points_to_sample).tolist()

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

           :input: :class:`moe.views.gp_ei.GpMeanVarRequest`
           :output: :class:`moe.views.gp_ei.GpMeanVarResponse`

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

           :input: :class:`moe.views.gp_ei.GpMeanVarRequest`
           :output: :class:`moe.views.gp_ei.GpMeanVarDiagResponse`

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

           :input: :class:`moe.views.gp_ei.GpMeanVarRequest`
           :output: :class:`moe.views.gp_ei.GpMeanResponse`

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

           :input: :class:`moe.views.gp_ei.GpMeanVarRequest`
           :output: :class:`moe.views.gp_ei.GpVarResponse`

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

           :input: :class:`moe.views.gp_ei.GpMeanVarRequest`
           :output: :class:`moe.views.gp_ei.GpVarDiagResponse`

           :status 200: returns a response
           :status 500: server error

        """
        return self.form_response(
            self.gp_mean_var_response_dict(compute_mean=False, compute_var=True, var_diag=True)
        )
