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
    # TODO(GH-56): Allow users to pass in a random source.
    if num_points == 0:
        return numpy.array([])

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


def generate_grid_points(points_per_dimension, domain_bounds):
    r"""Generate a uniform grid of points on a tensor product region; exponential runtime.

    This can be useful for producing a reasonable set of initial samples when bootstrapping optimal_learning.
    Grid sampling (as opposed to a random sampling, e.g., latin hypercube) is not random. It also guarantees
    sampling of the domain corners.

    .. Note:: This operation is like an outer-product, so 4 points per dimension in 10 dimensions produces
        4^{10} points. This could be built as an iterator instead, but the typical use
        case involves function evaluations at every point, so generating the points is
        not the limiting factor.

    :param points_per_dimension: (n_1, n_2, ... n_{dim}) number of stencil points per spatial dimension.
        If len(points_per_dimension) == 1, then n_i = len(points_per_dimension)
    :type points_per_dimension: tuple or scalar
    :param domain_bounds: the boundaries of a dim-dimensional tensor-product domain
    :type domain_bounds: iterable of dim ClosedInterval
    :return: stencil point coordinates
    :rtype: array of float64 with shape (\Pi_i n_i, dim)

    """
    points_per_dimension = numpy.asarray(points_per_dimension)
    # Empty input OR at least 1 dimension has 0 points
    if points_per_dimension.size == 0 or not points_per_dimension.all():
        return numpy.array([])

    if points_per_dimension.size == 1:
        # resize fills new entries with copies of the original
        points_per_dimension = numpy.resize(points_per_dimension, len(domain_bounds))
    # List of 1D grids w/the specified number of points per dimension
    per_axis_grid = [numpy.linspace(bounds.min, bounds.max, points_per_dimension[i])
                     for i, bounds in enumerate(domain_bounds)]
    # meshgrid produces a list of ndarray that is used to evaluate functions on a grid.
    # The i-th output array has the coordinate of *every* grid point in the i-th dimension.
    mesh_grid = numpy.meshgrid(*per_axis_grid)
    # ravel flattens the input (same as numpy.flatten but it tries to avoid copying)
    # vstack stacks inputs vertically: so for our 1D arrays, the i-th input becomes
    # the i-th row in a matrix. And since each mesh_grid output has *every* coordinate
    # of the grid in that dimension, the *columns* of the stack contain every grid point.
    return numpy.vstack(map(numpy.ravel, mesh_grid)).T


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
