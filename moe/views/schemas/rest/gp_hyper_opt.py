# -*- coding: utf-8 -*-
"""Request/response schemas for ``gp_hyper_opt`` endpoints."""
import colander

from moe.optimal_learning.python.constant import DEFAULT_MAX_NUM_THREADS, MAX_ALLOWED_NUM_THREADS, LIKELIHOOD_TYPES, LOG_MARGINAL_LIKELIHOOD
from moe.views.schemas import base_schemas


class GpHyperOptRequest(base_schemas.StrictMappingSchema):

    """A :class:`moe.views.rest.gp_hyper_opt.GpHyperOptView` request colander schema.

    .. Note:: Particularly when the amount of historical data is low, the log likelihood
      may grow toward extreme hyperparameter values (i.e., toward 0 or infinity). Select
      reasonable domain bounds. For example, in a driving distance parameter, the scale
      of feet is irrelevant, as is the scale of 1000s of miles.

    .. Note:: MOE's default optimization parameters were tuned for hyperparameter values roughly in [0.01, 100].
      Venturing too far out of this range means the defaults may perform poorly.

    See additional notes in :class:`moe.views.schemas.base_schemas.CovarianceInfo`,
    :class:`moe.views.schemas.base_schemas.GpHistoricalInfo`.

    **Required fields**

    :ivar gp_historical_info: (:class:`moe.views.schemas.base_schemas.GpHistoricalInfo`) object of historical data
    :ivar domain_info: (:class:`moe.views.schemas.base_schemas.DomainInfo`) dict of domain information for the GP
    :ivar hyperparameter_domain_info: (:class:`moe.views.schemas.base_schemas.BoundedDomainInfo`) dict of domain information for the hyperparameter optimization

    **Optional fields**

    :ivar max_num_threads: (*int*) maximum number of threads to use in computation
    :ivar covariance_info: (:class:`moe.views.schemas.base_schemas.CovarianceInfo`) dict of covariance information, used as a starting point for optimization
    :ivar optimizer_info: (:class:`moe.views.schemas.base_schemas.OptimizerInfo`) dict of optimizer information

    **General Timing Results**

    Here are some "broad-strokes" timing results for hyperparameter optimization.
    These tests are not complete nor comprehensive; they're just a starting point.
    The tests were run on a Ivy Bridge 2.3 GHz quad-core CPU (i7-3615QM). Data was generated
    from a Gaussian Process prior. The optimization parameters were the default
    values (see :mod:`moe.optimal_learning.python.constant`) as of sha
    ``c19257049f16036e5e2823df87fbe0812720e291``.

    Below, ``N = num_sampled``.

    ======== ===================== ========================
    Scaling with dim (N = 40)
    -------------------------------------------------------
      dim     Gradient Descent             Newton
    ======== ===================== ========================
      3           85s                      3.6s
      6           80s                      7.2s
      12         108s                     19.5s
    ======== ===================== ========================

    GD scales ``~ O(dim)`` and Newton ``~ O(dim^2)`` although these dim values
    are not large enough to show the asymptotic behavior.

    ======== ===================== ========================
    Scaling with N (dim = 3)
    -------------------------------------------------------
      N       Gradient Descent             Newton
    ======== ===================== ========================
      20        14s                       0.72s
      40        85s                        3.6s
      120       2100s                       60s
    ======== ===================== ========================

    Both methods scale as ``~ O(N^3)`` which is clearly shown here.

    **Example Request**

    .. sourcecode:: http

        Content-Type: text/javascript

        {
            "max_num_threads": 1,
            "gp_historical_info": {
                "points_sampled": [
                        {"value_var": 0.01, "value": 0.1, "point": [0.0]},
                        {"value_var": 0.01, "value": 0.2, "point": [1.0]}
                    ],
                },
            "domain_info": {
                "dim": 1,
                },
            "covariance_info": {
                "covariance_type": "square_exponential",
                "hyperparameters": [1.0, 1.0],
                },
            "hyperparameter_domain_info": {
                "dim": 2,
                "domain_bounds": [
                    {"min": 0.1, "max": 2.0},
                    {"min": 0.1, "max": 2.0},
                    ],
                },
            "optimizer_info": {
                "optimizer_type": "newton_optimizer",
                "num_multistarts": 200,
                "num_random_samples": 4000,
                "optimizer_parameters": {
                    "gamma": 1.2,
                    ...
                    },
                },
            "log_likelihood_info": "log_marginal_likelihood"
        }

    """

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
    hyperparameter_domain_info = base_schemas.BoundedDomainInfo()
    optimizer_info = base_schemas.OptimizerInfo(
            missing=base_schemas.OptimizerInfo().deserialize({}),
            )
    log_likelihood_info = colander.SchemaNode(
            colander.String(),
            validator=colander.OneOf(LIKELIHOOD_TYPES),
            missing=LOG_MARGINAL_LIKELIHOOD,
            )


class GpHyperOptStatus(base_schemas.StrictMappingSchema):

    """A :class:`moe.views.rest.gp_hyper_opt.GpHyperOptView` status schema.

    **Output fields**

    :ivar log_likelihood: (*float64*) The log likelihood at the new hyperparameters
    :ivar grad_log_likelihood: (*list of float64*) The gradient of the log likelihood at the new hyperparameters
    :ivar optimizer_success: (*dict*) Whether or not the optimizer converged to an optimal set of hyperparameters

    """

    log_likelihood = colander.SchemaNode(colander.Float())
    grad_log_likelihood = base_schemas.ListOfFloats()
    optimizer_success = colander.SchemaNode(
        colander.Mapping(unknown='preserve'),
        default={'found_update': False},
    )


class GpHyperOptResponse(base_schemas.StrictMappingSchema):

    """A :class:`moe.views.rest.gp_hyper_opt.GpHyperOptView` response colander schema.

    **Output fields**

    :ivar endpoint: (*str*) the endpoint that was called
    :ivar covariance_info: (:class:`moe.views.schemas.base_schemas.CovarianceInfo`) dict of covariance information
    :ivar status: (:class:`moe.views.schemas.rest.gp_hyper_opt.GpHyperOptStatus`) dict indicating final log likelihood value/gradient and
      optimization status messages (e.g., success)

    **Example Response**

    .. sourcecode:: http

        {
            "endpoint":"gp_hyper_opt",
            "covariance_info": {
                "covariance_type": "square_exponential",
                "hyperparameters": ["0.88", "1.24"],
                },
            "status": {
                "log_likelihood": "-37.3279872",
                "grad_log_likelihood: ["-3.8897e-12", "1.32789789e-11"],
                "optimizer_success": {
                        'newton_found_update': True,
                    },
                },
        }

    """

    endpoint = colander.SchemaNode(colander.String())
    covariance_info = base_schemas.CovarianceInfo()
    status = GpHyperOptStatus()
