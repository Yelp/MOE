# -*- coding: utf-8 -*-
"""Base request/response schemas for ``gp_next_points_*`` endpoints; e.g., :class:`moe.views.rest.gp_next_points_epi.GpNextPointsEpi`."""
import colander

from moe.optimal_learning.python.constant import DEFAULT_MAX_NUM_THREADS, MAX_ALLOWED_NUM_THREADS, DEFAULT_EXPECTED_IMPROVEMENT_MC_ITERATIONS
from moe.views.schemas import base_schemas


class GpNextPointsRequest(base_schemas.StrictMappingSchema):

    """A request colander schema for the various subclasses of :class:`moe.views.gp_next_points_pretty_view.GpNextPointsPrettyView`; e.g., :class:`moe.views.rest.gp_next_points_epi.GpNextPointsEpi`.

    .. Warning:: Requesting ``num_to_sample`` > ``num_sampled`` can lead to singular GP-variance
      matrices and other conditioning issues (e.g., the resulting points may be tightly
      clustered). We suggest bootstrapping with a few grid points, random search, etc.

    See additional notes in :class:`moe.views.schemas.base_schemas.CovarianceInfo`,
    :class:`moe.views.schemas.base_schemas.BoundedDomainInfo`, :class:`moe.views.schemas.base_schemas.GpHistoricalInfo`.

    **Required fields**

    :ivar gp_historical_info: (:class:`moe.views.schemas.base_schemas.GpHistoricalInfo`) dict of historical data
    :ivar domain_info: (:class:`moe.views.schemas.base_schemas.BoundedDomainInfo`) dict of domain information

    **Optional fields**

    :ivar num_to_sample: (*int*) number of next points to generate (default: 1)
    :ivar mc_iterations: (*int*) number of Monte Carlo (MC) iterations to perform in numerical integration to calculate EI
    :ivar max_num_threads: (*int*) maximum number of threads to use in computation
    :ivar covariance_info: (:class:`moe.views.schemas.base_schemas.CovarianceInfo`) dict of covariance information
    :ivar optimizer_info: (:class:`moe.views.schemas.base_schemas.OptimizerInfo`) dict of optimizer information
    :ivar points_being_sampled: (*list of float64*) points in domain being sampled in concurrent experiments (default: [])

    **General Timing Results**

    Here are some "broad-strokes" timing results for EI optimization.
    These tests are not complete nor comprehensive; they're just a starting point.
    The tests were run on a Ivy Bridge 2.3 GHz quad-core CPU (i7-3615QM). Data was generated
    from a Gaussian Process prior. The optimization parameters were the default
    values (see :mod:`moe.optimal_learning.python.constant`) as of sha
    ``c19257049f16036e5e2823df87fbe0812720e291``.

    Below, ``N = num_sampled``, ``MC = num_mc_iterations``, and ``q = num_to_sample``

    .. Note:: constant liar, kriging, and EPI (with ``num_to_sample = 1``) all fall
      under the ``analytic EI`` name.

    .. Note:: EI optimization times can vary widely as some randomly generated test
      cases are very "easy." Thus we give rough ranges.

    =============  =======================
     Analytic EI
    --------------------------------------
      dim, N           Gradient Descent
    =============  =======================
      3, 20               0.3 - 0.6s
      3, 40               0.5 - 1.4s
      3, 120              0.8 - 2.9s
      6, 20               0.4 - 0.9s
      6, 40              1.25 - 1.8s
      6, 120              2.9 - 5.0s
    =============  =======================

    We expect this to scale as ``~ O(dim)`` and ``~ O(N^3)``. The ``O(N^3)`` only happens
    once per multistart. Per iteration there's an ``O(N^2)`` dependence but as you can
    see, the dependence on ``dim`` is stronger.

    =============  =======================
     MC EI (``N = 20``, ``dim = 3``)
    --------------------------------------
      q, MC           Gradient Descent
    =============  =======================
      2, 10k                50 - 80s
      4, 10k              120 - 180s
      8, 10k              400 - 580s
      2, 40k              230 - 480s
      4, 40k              600 - 700s
    =============  =======================

    We expect this to scale as ``~ O(q^2)`` and ``~ O(MC)``. Scaling with ``dim`` and ``N``
    should be similar to the analytic case.

    **Example Minimal Request**

    .. sourcecode:: http

        Content-Type: text/javascript

        {
            "num_to_sample": 1,
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

    **Example Full Request**

    .. sourcecode:: http

        Content-Type: text/javascript

        {
            "num_to_sample": 1,
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
                "domain_bounds": [
                    {"min": 0.0, "max": 1.0},
                    ],
                },
            "covariance_info": {
                "covariance_type": "square_exponential",
                "hyperparameters": [1.0, 1.0],
                },
            "optimizer_info": {
                "optimizer_type": "gradient_descent_optimizer",
                "num_multistarts": 200,
                "num_random_samples": 4000,
                "optimizer_parameters": {
                    "gamma": 0.5,
                    ...
                    },
                },
        }

    """

    num_to_sample = colander.SchemaNode(
            colander.Int(),
            missing=1,
            validator=colander.Range(min=1),
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
    domain_info = base_schemas.BoundedDomainInfo()
    covariance_info = base_schemas.CovarianceInfo(
            missing=base_schemas.CovarianceInfo().deserialize({}),
            )
    optimizer_info = base_schemas.OptimizerInfo(
            missing=base_schemas.OptimizerInfo().deserialize({}),
            )
    points_being_sampled = base_schemas.ListOfPointsInDomain(
            missing=[],
            )


class GpNextPointsStatus(base_schemas.StrictMappingSchema):

    """A status schema for the various subclasses of :class:`moe.views.gp_next_points_pretty_view.GpNextPointsPrettyView`; e.g., :class:`moe.views.rest.gp_next_points_epi.GpNextPointsEpi`.

    **Output fields**

    :ivar expected_improvement: (*float64 >= 0.0*) EI evaluated at ``points_to_sample`` (:class:`moe.views.schemas.base_schemas.ListOfExpectedImprovements`)
    :ivar optimizer_success: (*dict*) Whether or not the optimizer converged to an optimal set of ``points_to_sample``

    """

    expected_improvement = colander.SchemaNode(
            colander.Float(),
            validator=colander.Range(min=0.0),
            )
    optimizer_success = colander.SchemaNode(
        colander.Mapping(unknown='preserve'),
        default={'found_update': False},
    )


class GpNextPointsResponse(base_schemas.StrictMappingSchema):

    """A response colander schema for the various subclasses of :class:`moe.views.gp_next_points_pretty_view.GpNextPointsPrettyView`; e.g., :class:`moe.views.rest.gp_next_points_epi.GpNextPointsEpi`.

    **Output fields**

    :ivar endpoint: (*str*) the endpoint that was called
    :ivar points_to_sample: (*list of list of float64*) points in the domain to sample next (:class:`moe.views.schemas.base_schemas.ListOfPointsInDomain`)
    :ivar status: (:class:`moe.views.schemas.gp_next_points_pretty_view.GpNextPointsStatus`) dict indicating final EI value and
      optimization status messages (e.g., success)

    **Example Response**

    .. sourcecode:: http

        {
            "endpoint": "gp_ei",
            "points_to_sample": [["0.478332304526"]],
            "status": {
                "expected_improvement": "0.443478498868",
                "optimizer_success": {
                    'gradient_descent_tensor_product_domain_found_update': True,
                    },
                },
        }

    """

    endpoint = colander.SchemaNode(colander.String())
    points_to_sample = base_schemas.ListOfPointsInDomain()
    status = GpNextPointsStatus()
