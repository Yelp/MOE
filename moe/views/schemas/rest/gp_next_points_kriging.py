# -*- coding: utf-8 -*-
"""Response schemas for ``gp_next_points_kriging`` endpoints."""
import colander

from moe.optimal_learning.python.constant import DEFAULT_KRIGING_NOISE_VARIANCE, DEFAULT_KRIGING_STD_DEVIATION_COEF
from moe.views.schemas.gp_next_points_pretty_view import GpNextPointsRequest


class GpNextPointsKrigingRequest(GpNextPointsRequest):

    """Extends the standard request :class:`moe.views.schemas.gp_next_points_pretty_view.GpNextPointsRequest` with kriging parameters, for use with :class:`moe.views.rest.gp_next_points_kriging.GpNextPointsKriging`.

    See :func:`moe.optimal_learning.python.cpp_wrappers.expected_improvement.kriging_believer_expected_improvement_optimization` for more info.

    .. Warning:: Setting :attr:`kriging_noise_variance` to 0 may cause singular GP covariance
      matrices when paired with large ``num_to_sample`` (for the same reason given in
      :class:`moe.views.schemas.base_schemas.GpHistoricalInfo`).

      Setting large :attr:`kriging_noise_variance` may cause the output ``points_to_sample``
      to cluster--if one heuristic estimate is good and has large noise, MOE will want to
      increase resample that location to increase certainty.

    See additional notes/warnings in :class:`moe.views.schemas.gp_next_points_pretty_view.GpNextPointsRequest`.

    **Required fields**

    :ivar gp_historical_info: (:class:`moe.views.schemas.base_schemas.GpHistoricalInfo`) dict of historical data
    :ivar domain_info: (:class:`moe.views.schemas.base_schemas.BoundedDomainInfo`) dict of domain information

    **Optional fields**

    :ivar num_to_sample: number of next points to generate (default: 1)
    :ivar std_deviation_coef: (*float64*) amount of GP-variance to add to each Kriging estimate, see
      :func:`moe.optimal_learning.python.cpp_wrappers.expected_improvement.kriging_believer_expected_improvement_optimization` (default: 0.0)
    :ivar kriging_noise_variance: (*float64 >= 0.0*) noise variance for each Kriging estimate, see
      :func:`moe.optimal_learning.python.cpp_wrappers.expected_improvement.kriging_believer_expected_improvement_optimization` (default: 0.0)
    :ivar covariance_info: (:class:`moe.views.schemas.base_schemas.CovarianceInfo`) dict of covariance information
    :ivar optimiaztion_info: (:class:`moe.views.schemas.base_schemas.OptimizerInfo`) dict of optimization information

    **General Timing Results**

    See the ``Analytic EI`` table in :class:`moe.views.schemas.gp_next_points_pretty_view.GpNextPointsRequest` for
    rough timing numbers.

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
            missing=DEFAULT_KRIGING_STD_DEVIATION_COEF,
            )
    kriging_noise_variance = colander.SchemaNode(
            colander.Float(),
            missing=DEFAULT_KRIGING_NOISE_VARIANCE,
            validator=colander.Range(min=0.0),
            )
