# -*- coding: utf-8 -*-
"""Response schemas for ``gp_next_points_constant_liar`` endpoints."""
import colander

from moe.optimal_learning.python.constant import CONSTANT_LIAR_METHODS, DEFAULT_CONSTANT_LIAR_METHOD, DEFAULT_CONSTANT_LIAR_LIE_NOISE_VARIANCE
from moe.views.schemas.gp_next_points_pretty_view import GpNextPointsRequest


class GpNextPointsConstantLiarRequest(GpNextPointsRequest):

    """Extends the standard request :class:`moe.views.schemas.gp_next_points_pretty_view.GpNextPointsRequest` with a lie value, for use with :class:`moe.views.rest.gp_next_points_constant_liar.GpNextPointsConstantLiar`.

    See :func:`moe.optimal_learning.python.cpp_wrappers.expected_improvement.constant_liar_expected_improvement_optimization` for more info.

    .. Warning:: Setting :attr:`lie_value` ``< best_so_far`` (``= min(points_sampled_value)``)
      will lead to poor results. The resulting ``points_to_sample`` will be tightly clustered.
      Such results are generally of low value and may cause singular GP-variance matrices too.

    .. Warning:: Setting :attr:`lie_noise_variance` to 0 may cause singular GP covariance
      matrices when paired with large ``num_to_sample`` (for the same reason given in
      :class:`moe.views.schemas.base_schemas.GpHistoricalInfo`).

      Setting large :attr:`lie_noise_variance` may cause the output ``points_to_sample``
      to cluster--if one heuristic estimate is good and has large noise, MOE will want to
      increase resample that location to increase certainty.

    See additional notes/warnings in :class:`moe.views.schemas.gp_next_points_pretty_view.GpNextPointsRequest`.

    **Required fields**

    :ivar gp_historical_info: (:class:`moe.views.schemas.base_schemas.GpHistoricalInfo`) dict of historical data
    :ivar domain_info: (:class:`moe.views.schemas.base_schemas.BoundedDomainInfo`) dict of domain information

    **Optional fields**

    :ivar num_to_sample: (*int*) number of next points to generate (default: 1)
    :ivar lie_method: (*str*) name from `CONSTANT_LIAR_METHODS` representing the liar method to use (default: 'constant_liar_min')
    :ivar lie_value: (*float64*) the 'lie' the Constant Liar heuristic will use (default: None). If `lie_value` is not None the algorithm will use this value instead of one calculated using `lie_method`.
    :ivar lie_noise_variance: (*float64 >= 0.0*) the noise variance of the 'lie' value (default: 0.0)
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
            "lie_value": 0.0,
            "lie_noise_variance": 1e-12,
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

    lie_method = colander.SchemaNode(
            colander.String(),
            missing=DEFAULT_CONSTANT_LIAR_METHOD,
            validator=colander.OneOf(CONSTANT_LIAR_METHODS),
            )
    lie_value = colander.SchemaNode(
            colander.Float(),
            missing=None,
            )
    lie_noise_variance = colander.SchemaNode(
            colander.Float(),
            missing=DEFAULT_CONSTANT_LIAR_LIE_NOISE_VARIANCE,
            validator=colander.Range(min=0.0),
            )
