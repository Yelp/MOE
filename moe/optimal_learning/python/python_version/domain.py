# -*- coding: utf-8 -*-
"""Various python implementations of interfaces.domain_interface.DomainInterface (e.g., TensorProduct).

These are currently used to describe domain limits for optimizers (i.e., implementations of interfaces/optimization_interface.py)

Each domain provides functions to describe the set of boundary planes, check whether a point is inside/outside, generate
random points inside, and limit updates (from optimizers) so that a path stays inside the domain.

"""
import copy

import numpy

from moe.optimal_learning.python.geometry_utils import generate_latin_hypercube_points
from moe.optimal_learning.python.interfaces.domain_interface import DomainInterface


class TensorProductDomain(DomainInterface):

    r"""Domain type for a tensor product domain.

    A d-dimensional tensor product domain is ``D = [x_0_{min}, x_0_{max}] X [x_1_{min}, x_1_{max}] X ... X [x_d_{min}, x_d_{max}]``

    """

    def __init__(self, domain_bounds):
        """Construct a TensorProductDomain that can be used with cpp_wrappers.* functions/classes.

        :param domain_bounds: the boundaries of a dim-dimensional tensor-product domain
        :type domain_bounds: iterable of dim ClosedInterval

        """
        self._domain_bounds = copy.deepcopy(domain_bounds)

        for interval in self._domain_bounds:
            if interval.is_empty():
                raise ValueError('Tensor product region is EMPTY.')

    @property
    def dim(self):
        """Return the number of spatial dimensions."""
        return len(self._domain_bounds)

    def check_point_inside(self, point):
        r"""Check if a point is inside the domain/on its boundary or outside.

        :param point: point to check
        :type point: array of float64 with shape (dim)
        :param points_to_sample: points which are being sampled concurrently (i.e., p in q,p-EI)
        :type points_to_sample: array of float64 with shape (num_to_sample, dim)
        :return: true if point is inside the domain
        :rtype: bool

        """
        # Generate a list of bool; i-th entry is True if i-th coordinate is inside the i-th bounds.
        # Then check that all entries are True.
        return all([interval.is_inside(point[i]) for i, interval in enumerate(self._domain_bounds)])

    def generate_random_point_in_domain(self, random_source=None):
        """Generate ``point`` uniformly at random such that ``self.check_point_inside(point)`` is True.

        .. Note:: if you need multiple points, use generate_uniform_random_points_in_domain instead; it
          yields better distributions over many points (via latin hypercube samplling) b/c it guarantees
          that no non-uniform clusters may arise (in subspaces) versus this method which treats all draws
          independently.

        :return: point in domain
        :rtype: array of float64 with shape (dim)

        """
        return numpy.array([numpy.random.uniform(interval.min, interval.max) for interval in self._domain_bounds])

    def generate_uniform_random_points_in_domain(self, num_points, random_source=None):
        r"""Generate ``num_points`` on a latin-hypercube (i.e., like a checkerboard).

        :param num_points: max number of points to generate
        :type num_points: integer >= 0
        :param random_source:
        :type random_source: callable yielding uniform random numbers in [0,1]
        :return: uniform random sampling of points from the domain
        :rtype: array of float64 with shape (num_points, dim)

        """
        # TODO(eliu): actually allow users to pass in a random source (GH-56)
        return generate_latin_hypercube_points(num_points, self._domain_bounds)

    def compute_update_restricted_to_domain(self, max_relative_change, current_point, update_vector):
        r"""Compute a new update so that CheckPointInside(``current_point`` + ``new_update``) is true.

        Changes new_update_vector so that:
          ``point_new = point + new_update_vector``

        has coordinates such that ``CheckPointInside(point_new)`` returns true. We select ``point_new``
        by projecting ``point + update_vector`` to the nearest point on the domain.

        ``new_update_vector`` is a function of ``update_vector``.
        ``new_update_vector`` is just a copy of ``update_vector`` if ``current_point`` is already inside the domain.

        .. NOTE::
            We modify update_vector (instead of returning point_new) so that further update
            limiting/testing may be performed.

        :param max_relative_change: max change allowed per update (as a relative fraction of current distance to boundary)
        :type max_relative_change: float64 in (0, 1]
        :param current_point: starting point
        :type current_point: array of float64 with shape (dim)
        :param update_vector: proposed update
        :type update_vector: array of float64 with shape (dim)
        :return: new update so that the final point remains inside the domain
        :rtype: array of float64 with shape (dim)

        """
        # TODO(eliu): vectorize this (GH-58)
        output_update = numpy.empty(self.dim)
        # Note: since all boundary planes are axis-aligned, projecting becomes very simple.
        for j, step in enumerate(update_vector):
            # Distance to the nearest boundary in the j-th dimension
            distance_to_boundary = numpy.fmin(
                current_point[j] - self._domain_bounds[j].min,
                self._domain_bounds[j].max - current_point[j])

            desired_step = step
            # If we are close to a boundary, optionally (via max_relative_change) limit the step size
            # 0 < max_relative_change <= 1 so at worst we reach the boundary.
            if numpy.fabs(step) > max_relative_change * distance_to_boundary:
                # Move the max allowed distance, in the original direction of travel (obtained via copy-sign)
                desired_step = numpy.copysign(max_relative_change * distance_to_boundary, step)

            output_update[j] = desired_step

        return output_update
