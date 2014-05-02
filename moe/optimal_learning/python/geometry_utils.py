# -*- coding: utf-8 -*-
"""Geometry utilities. e.g., ClosedInterval, point-plane geometry."""
import collections


# See ClosedInterval (below) for docstring.
_BaseClosedInterval = collections.namedtuple('ClosedInterval', ['min', 'max'])


class ClosedInterval(_BaseClosedInterval):

    r"""Container to represent the mathematical notion of a closed interval, commonly written \ms [a,b]\me.

    The closed interval \ms [a,b]\me is the set of all numbers \ms x \in \mathbb{R}\me such that \ms a \leq x \leq b\me.
    Note that "closed" here indicates the interval *includes* both endpoints.
    An interval with \ms a > b\me is considered empty.

    :ivar min: (*float64*) the "left" bound of the domain, ``a``
    :ivar max: (*float64*) the "right" bound of the domain, ``b``

    """

    __slots__ = ()

    @staticmethod
    def build_closed_intervals_from_list(bounds_list):
        """Construct a list of dim ClosedInterval from an iterable structure of dim iterables with len = 2.

        For example, [[1, 2], [3, 4]] becomes [ClosedInterval(min=1, max=2), ClosedInterval(min=3, max=4)].

        :param bounds_list: bounds to convert
        :type bounds_list: iterable of iterables, where the second dimension has len = 2
        :return: bounds_list converted to list of ClosedInterval
        :rtype: list of ClosedInterval

        """
        return [ClosedInterval(min, max) for (min, max) in bounds_list]

    @property
    def length(self):
        """Compute the length of this ClosedInterval."""
        return self.max - self.min

    def is_inside(self, value):
        """Check if a value is inside this ClosedInterval."""
        return self.min <= value <= self.max

    def is_empty(self):
        """Check whether this ClosedInterval is the emptyset: max < min."""
        return self.max < self.min
