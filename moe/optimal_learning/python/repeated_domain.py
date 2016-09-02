# -*- coding: utf-8 -*-
"""RepeatedDomain class for handling manipulating sets of points in a (kernel) domain simultaneously."""
from builtins import range
import numpy

from moe.optimal_learning.python.interfaces.domain_interface import DomainInterface


class RepeatedDomain(DomainInterface):

    """A generic domain type for simultaneously manipulating ``num_repeats`` points in a "regular" domain (the kernel).

    .. Note:: Comments in this class are copied from RepeatedDomain in gpp_domain.hpp.

    .. Note:: the kernel domain is *not* copied. Instead, the kernel functions are called
      ``num_repeats`` times in a loop. In some cases, data reordering is also necessary
      to preserve the output properties (e.g., uniform distribution).

    For some use cases (e.g., q,p-EI optimization with q > 1), we need to simultaneously
    manipulate several points within the same domain. To support this use case, we have
    the ``RepeatedDomain``, a light-weight wrapper around any DomainInterface subclass
    that kernalizes that object's functionality.

    In general, kernel domain operations need be performed ``num_repeats`` times, once
    for each point. This class hides the looping logic so that use cases like various
    :class:`moe.optimal_learning.python.interfaces.optimization_interface.OptimizerInterface`
    subclasses do not need to be explicitly aware
    of whether they are optimizing 1 point or 50 points. Instead, the OptimizableInterface
    implementation provides problem_size() and appropriately sized gradient information.
    Coupled with RepeatedDomain, Optimizers can remain oblivious.

    In simpler terms, say we want to solve 5,0-EI in a parameter-space of dimension 3.
    So we would have 5 points moving around in a 3D space. The 3D space, whatever it is,
    is the kernel domain. We "repeat" the kernel 5 times; in practice this mostly amounts to
    simple loops around kernel functions and sometimes data reordering is also needed.

    .. Note:: this operation is more complex than just working in a higher dimensional space.
      3 points in a 2D simplex is not the same as 1 point in a 6D simplex; e.g.,
      ``[(0.5, 0.5), (0.5, 0.5), (0.5, 0.5)]`` is valid in the first scenario but not in the second.

    Where the member domain takes ``kernel_input``, this class's members take an array with
    shape ``(num_repeats, ) + kernel_input.shape``. Similarly ``kernel_output`` becomes an
    array with shape ``(num_repeats, ) + kernel_output.shape``.

    For example, ``check_point_inside()`` calls the kernel domain's ``check_point_inside()``
    function ``num_repeats`` times, returning True only if all ``num_repeats`` input
    points are inside the kernel domain.

    """

    def __init__(self, num_repeats, domain):
        """Construct a RepeatedDomain with the specified input (kernel) domain and number of repeats.

        :param num_repeats: number of times to repeat the input domain
        :type num_repeats: int > 0
        :param domain: the domain to repeat
        :type domain: DomainInterface subclass

        """
        self.num_repeats = num_repeats
        self._domain = domain

    @property
    def dim(self):
        """Return the number of spatial dimensions of the kernel domain."""
        return self._domain.dim

    def check_point_inside(self, points):
        r"""Check if a point is inside the domain/on its boundary or outside.

        :param point: point to check
        :type point: array of float64 with shape (num_repeats, dim)
        :return: true if point is inside the repeated domain
        :rtype: bool

        """
        return all([self._domain.check_point_inside(point) for point in points])

    def get_bounding_box(self):
        """Return a list of ClosedIntervals representing a bounding box for this domain."""
        return self._domain.get_bounding_box()

    def get_constraint_list(self):
        """Return a list of lambda functions expressing the domain bounds as linear constraints. Used by COBYLA.

        Calls ``self._domain.get_constraint_list()`` for each repeat, writing the results sequentially.
        So output[0:2*dim] is from the first repeated domain, output[2*dim:4*dim] is from the second, etc.

        :return: a list of lambda functions corresponding to constraints
        :rtype: array of lambda functions with shape (num_repeats * dim * 2)

        """
        constraints = []
        for i in range(self.num_repeats):
            # Using start_index, start each domain at the correct index when flattening out points in COBYLA.
            constraints.extend(self._domain.get_constraint_list(start_index=self.dim * i))
        return constraints

    def generate_random_point_in_domain(self, random_source=None):
        """Generate ``point`` uniformly at random such that ``self.check_point_inside(point)`` is True.

        .. Note:: if you need multiple points, use generate_uniform_random_points_in_domain instead;
            depending on implementation, it may ield better distributions over many points. For example,
            tensor product type domains use latin hypercube sampling instead of repeated random draws
            which guarantees that no non-uniform clusters may arise (in subspaces) versus this method
            which treats all draws independently.

        :return: point in repeated domain
        :rtype: array of float64 with shape (num_repeats, dim)

        """
        return numpy.array([self._domain.generate_random_point_in_domain(random_source=random_source)
                            for _ in range(self.num_repeats)])

    def generate_uniform_random_points_in_domain(self, num_points, random_source=None):
        r"""Generate AT MOST ``num_points`` uniformly distributed points from the domain.

        Unlike many of this class's other member functions, ``generate_uniform_random_points_in_domain()``
        is not as simple as calling the kernel's member function ``num_repeats`` times. To
        obtain the same distribution, we have to additionally "transpose" (see implementation
        for details).

        .. NOTE::
             The number of points returned may be LESS THAN ``num_points``!

        Implementations may use rejection sampling. In such cases, generating the requested
        number of points may be unreasonably slow, so implementers are allowed to generate
        fewer than ``num_points`` results.

        :param num_points: max number of points to generate
        :type num_points: integer >= 0
        :param random_source:
        :type random_source: callable yielding uniform random numbers in [0,1]
        :return: uniform random sampling of points from the domain; may be fewer than ``num_points``!
        :rtype: array of float64 with shape (num_points_generated, num_repeats, dim)

        """
        output_points = numpy.empty((num_points, self.num_repeats, self.dim))
        # Generate num_repeats sets of points from some sampling (e.g., LHC)
        # Then we "transpose" the output ordering: the i-th point in RepeatedDomain is constructed
        # from the i-th points of LHC_1 ... LHC_{num_repeats}
        num_points_array = numpy.empty(self.num_repeats, dtype=numpy.int64)
        for i in range(self.num_repeats):
            temp = self._domain.generate_uniform_random_points_in_domain(num_points, random_source=random_source)
            # Since generate_uniform_random_points_in_domain() may not always return num_points
            # points, we need to make sure we only use the valid results
            num_points_array[i] = temp.shape[0]
            output_points[:, i, ...] = temp
        # We can only use the smallest num_points that came out of our draws
        return output_points[:numpy.amin(num_points_array), ...]

    def compute_update_restricted_to_domain(self, max_relative_change, current_point, update_vector):
        r"""Compute a new update so that CheckPointInside(``current_point`` + ``return_value``) is true.

        Returns a new update vector in ``return_value`` so that:
          ``point_new = point + return_value``

        has coordinates such that ``CheckPointInside(point_new)`` returns true. We select ``point_new``
        by projecting ``point + update_vector`` to the nearest point on the domain.

        ``return_value`` is a function of ``update_vector``.
        ``return_value`` is just a copy of ``update_vector`` if ``current_point`` is already inside the domain.

        .. NOTE::
            We modify update_vector (instead of returning point_new) so that further update
            limiting/testing may be performed.

        :param max_relative_change: max change allowed per update (as a relative fraction of current distance to boundary)
        :type max_relative_change: float64 in (0, 1]
        :param current_point: starting point
        :type current_point: array of float64 with shape (num_repeats, dim)
        :param update_vector: proposed update
        :type update_vector: array of float64 with shape (num_repeats, dim)
        :return: new update so that the final point remains inside the domain
        :rtype: array of float64 with shape (num_repeats, dim)

        """
        return numpy.array([self._domain.compute_update_restricted_to_domain(
            max_relative_change,
            current_point[i, ...],
            update_vector[i, ...])
                            for i in range(self.num_repeats)])
