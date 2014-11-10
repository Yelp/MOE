# -*- coding: utf-8 -*-
"""Various python implementations of interfaces.domain_interface.DomainInterface (e.g., TensorProduct).

These are currently used to describe domain limits for optimizers (i.e., implementations of
:mod:`moe.optimal_learning.python.interfaces.optimization_interface`).

Each domain provides functions to:

* Describe the set of boundary planes
* Check whether a point is inside/outside
* Generate random point(s) inside
* Generate points on a fixed grid
* Limit updates (from optimizers) so that a path stays inside the domain

"""
import copy

import numpy

from moe.optimal_learning.python.constant import TENSOR_PRODUCT_DOMAIN_TYPE
from moe.optimal_learning.python.geometry_utils import generate_grid_points, generate_latin_hypercube_points
from moe.optimal_learning.python.interfaces.domain_interface import DomainInterface


class TensorProductDomain(DomainInterface):

    r"""Domain type for a tensor product domain.

    A d-dimensional tensor product domain is ``D = [x_0_{min}, x_0_{max}] X [x_1_{min}, x_1_{max}] X ... X [x_d_{min}, x_d_{max}]``

    """

    domain_type = TENSOR_PRODUCT_DOMAIN_TYPE

    def __init__(self, domain_bounds):
        """Construct a TensorProductDomain with the specified bounds.

        :param domain_bounds: the boundaries of a dim-dimensional tensor-product domain
        :type domain_bounds: iterable of dim :class:`moe.optimal_learning.python.geometry_utils.ClosedInterval`

        """
        self._domain_bounds = copy.deepcopy(domain_bounds)

        for interval in self._domain_bounds:
            if interval.is_empty():
                raise ValueError('Tensor product region is EMPTY.')

    @property
    def dim(self):
        """Return the number of spatial dimensions."""
        return len(self._domain_bounds)

    def get_json_serializable_info(self, minimal=False):
        """Create and return a domain_info dictionary of this domain object.

        :param minimal: True for all domain contents; False for ``domain_type`` and ``dim`` only
        :type minimal: bool
        :return: dict representation of this domain
        :rtype: dict

        """
        response = {
            'domain_type': self.domain_type,
            'dim': self.dim,
        }
        if not minimal:
            response['domain_bounds'] = self._domain_bounds

        return response

    def check_point_inside(self, point):
        r"""Check if a point is inside the domain/on its boundary or outside.

        :param point: point to check
        :type point: array of float64 with shape (dim)
        :return: true if point is inside the domain
        :rtype: bool

        """
        # Generate a list of bool; i-th entry is True if i-th coordinate is inside the i-th bounds.
        # Then check that all entries are True.
        return all([interval.is_inside(point[i]) for i, interval in enumerate(self._domain_bounds)])

    def get_bounding_box(self):
        """Return a list of ClosedIntervals representing a bounding box for this domain."""
        return copy.copy(self._domain_bounds)

    def get_constraint_list(self, start_index=0):
        """Return a list of lambda functions expressing the domain bounds as linear constraints. Used by COBYLA.

        Since COBYLA in scipy only optimizes arrays, we flatten out our points while doing multipoint EI optimization.
        But in order for the constraints to access the correct index, the RepeatedDomain class has to signal which index
        the TensorProductDomain should start from, using the start_index optional parameter.

        That is, RepeatedDomain deals with N d-dimensional points at once. Thus we need N*d constraints (one per
        dimension, once per repeat). Additionally, instead of receiving points with shape (num_repeats, dim), COBYLA
        requires that the points are flattened: (num_repeats*dim, ). Thus this method must know *where* in the
        flattened-list it is writing to and reading from: signaled via ``start_index``.

        :param start_index: the dimension this tensor product domain should start indexing from
        :type start_index: int >= 0
        :return: a list of lambda functions corresponding to constraints
        :rtype: array of lambda functions with shape (dim * 2)

        """
        constraints = []
        for i, interval in enumerate(self._domain_bounds):
            constraints.append((lambda x: x[i + start_index] - interval.min))
            constraints.append((lambda x: interval.max - x[i + start_index]))
        return constraints

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

        See python.geometry_utils.generate_latin_hypercube_points for more details.

        :param num_points: max number of points to generate
        :type num_points: int >= 0
        :param random_source: random source producing uniform random numbers (e.g., numpy.random.uniform) (UNUSED)
        :type random_source: callable yielding uniform random numbers in [0,1]
        :return: uniform random sampling of points from the domain
        :rtype: array of float64 with shape (num_points, dim)

        """
        # TODO(GH-56): Allow users to pass in a random source.
        return generate_latin_hypercube_points(num_points, self._domain_bounds)

    def generate_grid_points_in_domain(self, points_per_dimension, random_source=None):
        """Generate a grid of ``N_0 by N_1 by ... by N_{dim-1}`` points, with each dimension uniformly spaced along the domain boundary.

        See python.geometry_utils.generate_grid_points for more details.

        :param points_per_dimension: (n_1, n_2, ... n_{dim}) number of stencil points per spatial dimension.
            If len(points_per_dimension) == 1, then n_i = len(points_per_dimension)
        :type points_per_dimension: tuple or scalar
        :param random_source: random source producing uniform random numbers (e.g., numpy.random.uniform) (UNUSED)
        :type random_source: callable yielding uniform random numbers in [0,1]
        :return: uniform random sampling of points from the domain

        """
        # TODO(GH-56): Allow users to pass in a random source.
        return generate_grid_points(points_per_dimension, self._domain_bounds)

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
        # TODO(GH-58): Vectorize the loop over j, step.
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
