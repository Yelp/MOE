# -*- coding: utf-8 -*-
"""Classes for MOE optimizable experiments."""
import pprint

from moe.optimal_learning.python.constant import TENSOR_PRODUCT_DOMAIN_TYPE
from moe.optimal_learning.python.data_containers import HistoricalData
from moe.optimal_learning.python.linkers import DOMAIN_TYPES_TO_DOMAIN_LINKS
from moe.views.utils import _build_domain_info
from moe.optimal_learning.python.geometry_utils import ClosedInterval

DEFAULT_DOMAIN = TENSOR_PRODUCT_DOMAIN_TYPE


class Experiment(object):

    """A class for MOE optimizable experiments."""

    def __init__(self, domain_bounds, points_sampled=None, domain_type=DEFAULT_DOMAIN):
        """Construct a MOE optimizable experiment.

        **Required arguments:**

            :param domain_bounds: The bounds for the optimization experiment
            :type domain_bounds: An iterable of iterables describing the [min, max] of the domain for each dimension

        **Optional arguments:**

            :param points_sampled: The historic points sampled and their objective function values
            :type points_sampled: An iterable of iterables describing the [point, value, noise] of each objective function evaluation
            :param domain_type: The type of domain to use
            :type domain_type: A string from ``moe.optimal_learning.python.linkers.DOMAIN_TYPES_TO_DOMAIN_LINKS``

        """
        _domain_bounds = [ClosedInterval(bound[0], bound[1]) for bound in domain_bounds]
        self.domain = DOMAIN_TYPES_TO_DOMAIN_LINKS[domain_type].python_domain_class(_domain_bounds)
        self.historical_data = HistoricalData(
                self.domain.dim,
                sample_points=points_sampled,
                )

    def build_json_payload(self):
        """Construct a json serializeable and MOE REST recognizeable dictionary of the experiment."""
        return {
                'domain_info': _build_domain_info(self.domain),
                'gp_info': self.historical_data.json_payload(),
                }

    def __str__(self):
        """Return a pprint formated version of the experiment dict."""
        return pprint.pformat(self.build_json_payload)
