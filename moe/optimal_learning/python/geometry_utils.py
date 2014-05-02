# -*- coding: utf-8 -*-
"""Geometry utilities. e.g., ClosedInterval, point-plane geometry, random point generation."""
import collections

import numpy


def generate_latin_hypercube_points(num_points, domain_bounds):
    """Compute a set of random points inside some domain that lie in a latin hypercube.

    In 2D, a latin hypercube is a latin square--a checkerboard--such that there is exactly one sample in
    each row and each column.  This notion is generalized for higher dimensions where each dimensional
    'slice' has precisely one sample.

    See wikipedia: http://en.wikipedia.org/wiki/Latin_hypercube_sampling
    for more details on the latin hypercube sampling process.

    :param num_points: number of random points to generate
    :type num_points: int > 0
    :param domain_bounds: [min, max] boundaries of the hypercube in each dimension
    :type domain_bounds: list of dim ClosedInterval
    :return: uniformly distributed random points inside the specified hypercube
    :rtype: array of float64 with shape (num_points, dim)

    """
    # TODO(eliu): actually allow users to pass in a random source (GH-56)
    points = numpy.zeros((num_points, len(domain_bounds)), dtype=numpy.float64)
    for i, interval in enumerate(domain_bounds):
        # Cut the range into num_points slices
        subcube_edge_length = interval.length / float(num_points)

        # Create random ordering for slices
        ordering = numpy.arange(num_points)
        numpy.random.shuffle(ordering)

        for j in xrange(num_points):
            point_base = interval.min + subcube_edge_length * ordering[j]
            points[j, i] = point_base + numpy.random.uniform(0.0, subcube_edge_length)

    return points


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
