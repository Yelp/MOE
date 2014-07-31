# -*- coding: utf-8 -*-
"""Request/response schemas for ``gp_mean``, ``gp_var``, and ``gp_mean_var`` (and ``_diag``) endpoints."""
import colander

from moe.views.schemas import base_schemas


class GpMeanVarRequest(base_schemas.StrictMappingSchema):

    """A request colander schema for the views in :mod:`moe.views.rest.gp_mean_var`.

    .. Note:: Requesting ``points_to_evaluate`` that are close to each or close to existing
      ``points_sampled`` may result in a [numerically] singular GP-variance matrix.

    See additional notes in :class:`moe.views.schemas.base_schemas.CovarianceInfo`,
    :class:`moe.views.schemas.base_schemas.GpHistoricalInfo`.

    **Required fields**

    :ivar points_to_evaluate: (*list of list of float64*) points in domain to calculate the Gaussian Process (GP) mean and covariance at (:class:`moe.views.schemas.base_schemas.ListOfPointsInDomain`)
    :ivar gp_historical_info: (:class:`moe.views.schemas.base_schemas.GpHistoricalInfo`) object of historical data

    **Optional fields**

    :ivar covariance_info: (:class:`moe.views.schemas.base_schemas.CovarianceInfo`) dict of covariance information

    **Example Minimal Request**

    .. sourcecode:: http

        Content-Type: text/javascript

        {
            "points_to_evaluate": [[0.1], [0.5], [0.9]],
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
            "points_to_evaluate": [[0.1], [0.5], [0.9]],
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

    points_to_evaluate = base_schemas.ListOfPointsInDomain()
    gp_historical_info = base_schemas.GpHistoricalInfo()
    domain_info = base_schemas.DomainInfo()
    covariance_info = base_schemas.CovarianceInfo(
            missing=base_schemas.CovarianceInfo().deserialize({}),
            )


class GpEndpointResponse(base_schemas.StrictMappingSchema):

    """A base schema for the endpoint name.

    **Output fields**

    :ivar endpoint: (*str*) the endpoint that was called

    **Example Response**

    .. sourcecode:: http

        {
            "endpoint":"gp_mean_var",
        }

    """

    endpoint = colander.SchemaNode(colander.String())


class GpMeanMixinResponse(base_schemas.StrictMappingSchema):

    """A mixin response colander schema for the mean of a gaussian process.

    **Output fields**

    :ivar mean: (*list of float64*) the means of the GP at ``points_to_evaluate`` (:class:`moe.views.schemas.base_schemas.ListOfFloats`)

    **Example Response**

    .. sourcecode:: http

        {
            "mean": ["0.0873832198661","0.0130505261903","0.174755506336"],
        }

    """

    mean = base_schemas.ListOfFloats()


class GpVarMixinResponse(base_schemas.StrictMappingSchema):

    """A mixin response colander schema for the [co]variance of a gaussian process.

    **Output fields**

    :ivar var: (:class:`moe.views.schemas.base_schemas.MatrixOfFloats`) matrix of covariance of the GP at ``points_to_evaluate``

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

    var = base_schemas.MatrixOfFloats()


class GpVarDiagMixinResponse(base_schemas.StrictMappingSchema):

    """A mixin response colander schema for the variance of a gaussian process.

    **Output fields**

    :ivar var: (*list of float64*) variances of the GP at ``points_to_evaluate``; i.e., diagonal of the ``var`` response from gp_mean_var (:class:`moe.views.schemas.base_schemas.ListOfFloats`)

    **Example Response**

    .. sourcecode:: http

        {
            "var": ["0.228910114429","0.996177332647","0.228910114429"],
        }

    """

    var = base_schemas.ListOfFloats()


class GpMeanResponse(GpEndpointResponse, GpMeanMixinResponse):

    """A :class:`moe.views.rest.gp_mean_var.GpMeanView` response colander schema.

    See composing members' docstrings.

    """

    pass


class GpVarResponse(GpEndpointResponse, GpVarMixinResponse):

    """A :class:`moe.views.rest.gp_mean_var.GpVarView` response colander schema.

    See composing members' docstrings.

    """

    pass


class GpVarDiagResponse(GpEndpointResponse, GpVarDiagMixinResponse):

    """A :class:`moe.views.rest.gp_mean_var.GpVarDiagView` response colander schema.

    See composing members' docstrings.

    """

    pass


class GpMeanVarResponse(GpMeanResponse, GpVarMixinResponse):

    """A :class:`moe.views.rest.gp_mean_var.GpMeanVarView` response colander schema.

    See composing members' docstrings.

    """

    pass


class GpMeanVarDiagResponse(GpMeanResponse, GpVarDiagMixinResponse):

    """A :class:`moe.views.rest.gp_mean_var.GpMeanVarDiagView` response colander schema.

    See composing members' docstrings.

    """

    pass
