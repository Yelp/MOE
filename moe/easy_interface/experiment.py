# -*- coding: utf-8 -*-
"""Classes for MOE optimizable experiments."""
from builtins import object
import pprint

from moe.optimal_learning.python.data_containers import HistoricalData
from moe.optimal_learning.python.geometry_utils import ClosedInterval
from moe.optimal_learning.python.python_version.domain import TensorProductDomain


class Experiment(object):

    """A class for MOE optimizable experiments."""

    def __init__(self, domain_bounds, points_sampled=None):
        """Construct a MOE optimizable experiment.

        **Required arguments:**

            :param domain_bounds: The bounds for the optimization experiment
            :type domain_bounds: An iterable of iterables describing the [min, max] of the domain for each dimension

        **Optional arguments:**

            :param points_sampled: The historic points sampled and their objective function values
            :type points_sampled: An iterable of iterables describing the [point, value, noise] of each objective function evaluation

        """
        _domain_bounds = [ClosedInterval(bound[0], bound[1]) for bound in domain_bounds]
        self.domain = TensorProductDomain(_domain_bounds)
        self.historical_data = HistoricalData(
                self.domain.dim,
                sample_points=points_sampled,
                )

    def build_json_payload(self):
        """Construct a json serializeable and MOE REST recognizeable dictionary of the experiment."""
        return {
                'domain_info': self.domain.get_json_serializable_info(),
                'gp_historical_info': self.historical_data.json_payload(),
                }

    def __str__(self):
        """Return a pprint formated version of the experiment dict."""
        return pprint.pformat(self.build_json_payload)
