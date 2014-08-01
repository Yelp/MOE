# -*- coding: utf-8 -*-
"""Request/response schemas for ``gp_ei`` endpoints."""
import colander

from moe.optimal_learning.python.constant import DEFAULT_MAX_NUM_THREADS, MAX_ALLOWED_NUM_THREADS, DEFAULT_EXPECTED_IMPROVEMENT_MC_ITERATIONS
from moe.views.schemas import base_schemas


class GpEiRequest(base_schemas.StrictMappingSchema):

    """A :class:`moe.views.rest.gp_ei.GpEiView` request colander schema.

    .. Note:: Requesting ``points_to_evaluate`` and ``points_being_sampled`` that are close
      to each or close to existing ``points_sampled`` may result in a [numerically] singular
      GP-variance matrix (which is required by EI).

    See additional notes in :class:`moe.views.schemas.base_schemas.CovarianceInfo`,
    :class:`moe.views.schemas.base_schemas.GpHistoricalInfo`.

    **Required fields**

    :ivar points_to_evaluate: (*list of list of float64*) points in domain to calculate Expected Improvement (EI) at (:class:`moe.views.schemas.base_schemas.ListOfPointsInDomain`)
    :ivar gp_historical_info: (:class:`moe.views.schemas.base_schemas.GpHistoricalInfo`) object of historical data

    **Optional fields**

    :ivar points_being_sampled: (*list of list of float64*) points in domain being sampled in concurrent experiments (default: []) (:class:`moe.views.schemas.base_schemas.ListOfPointsInDomain`)
    :ivar mc_iterations: (*int*) number of Monte Carlo (MC) iterations to perform in numerical integration to calculate EI
    :ivar max_num_threads: (*int*) maximum number of threads to use in computation (default: 1)
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
            "points_being_sampled": [[0.2], [0.7]],
            "mc_iterations": 10000,
            "max_num_threads": 1,
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
    points_being_sampled = base_schemas.ListOfPointsInDomain(
            missing=[],
            )
    mc_iterations = colander.SchemaNode(
            colander.Int(),
            validator=colander.Range(min=1),
            missing=DEFAULT_EXPECTED_IMPROVEMENT_MC_ITERATIONS,
            )
    max_num_threads = colander.SchemaNode(
            colander.Int(),
            validator=colander.Range(min=1, max=MAX_ALLOWED_NUM_THREADS),
            missing=DEFAULT_MAX_NUM_THREADS,
            )
    gp_historical_info = base_schemas.GpHistoricalInfo()
    domain_info = base_schemas.DomainInfo()
    covariance_info = base_schemas.CovarianceInfo(
            missing=base_schemas.CovarianceInfo().deserialize({}),
            )


class GpEiResponse(base_schemas.StrictMappingSchema):

    """A :class:`moe.views.rest.gp_ei.GpEiView` response colander schema.

    **Output fields**

    :ivar endpoint: (*str*) the endpoint that was called
    :ivar expected_improvement: (*list of float64*) calculated expected improvements (:class:`moe.views.schemas.base_schemas.ListOfExpectedImprovements`)

    **Example Response**

    .. sourcecode:: http

        {
            "endpoint":"gp_ei",
            "expected_improvement":["0.197246898375","0.443163755117","0.155819546878"]
        }

    """

    endpoint = colander.SchemaNode(colander.String())
    expected_improvement = base_schemas.ListOfExpectedImprovements()
