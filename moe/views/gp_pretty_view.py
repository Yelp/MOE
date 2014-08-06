# -*- coding: utf-8 -*-
"""A class to encapsulate GP 'pretty' views."""
from moe.optimal_learning.python.constant import SQUARE_EXPONENTIAL_COVARIANCE_TYPE
from moe.views.pretty_view import PrettyView


class GpPrettyView(PrettyView):

    """A class to encapsulate GP 'pretty' views.

    See :class:`moe.views.pretty_view.PrettyView` superclass for more details.

    """

    _pretty_default_gp_historical_info = {
            "points_sampled": [
                {"value_var": 0.01, "value": 0.1, "point": [0.0]},
                {"value_var": 0.01, "value": 0.2, "point": [1.0]},
                ],
            }
    _pretty_default_covariance_info = {
            "covariance_type": SQUARE_EXPONENTIAL_COVARIANCE_TYPE,
            "hyperparameters": [1.0, 0.2],
            }
    _pretty_default_domain_info = {
            "dim": 1,
            "domain_type": "tensor_product",
            }
