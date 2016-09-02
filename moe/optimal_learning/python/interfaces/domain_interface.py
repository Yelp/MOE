# -*- coding: utf-8 -*-
"""Interface for a domain: in/out test, random point generation, and update limiting (for constrained optimization)."""
from builtins import object
from abc import ABCMeta, abstractmethod, abstractproperty
from future.utils import with_metaclass


class DomainInterface(with_metaclass(ABCMeta, object)):

    """Interface for a domain: in/out test, random point generation, and update limiting (for constrained optimization)."""

    @abstractproperty
    def dim(self):
        """Return the number of spatial dimensions."""
        pass

    @abstractmethod
    def check_point_inside(self, point):
        r"""Check if a point is inside the domain/on its boundary or outside.

        :param point: point to check
        :type point: array of float64 with shape (dim)
        :return: true if point is inside the domain
        :rtype: bool

        """
        pass

    @abstractmethod
    def get_bounding_box(self):
        """Return a list of ClosedIntervals representing a bounding box for this domain."""
        pass

    @abstractmethod
    def get_constraint_list(self):
        """Return a list of lambda functions expressing the domain bounds as linear constraints. Used by COBYLA.

        :return: a list of lambda functions corresponding to constraints
        :rtype: array of lambda functions with shape (dim * 2)

        """
        pass

    @abstractmethod
    def generate_random_point_in_domain(self, random_source=None):
        """Generate ``point`` uniformly at random such that ``self.check_point_inside(point)`` is True.

        .. Note:: if you need multiple points, use generate_uniform_random_points_in_domain instead;
            depending on implementation, it may ield better distributions over many points. For example,
            tensor product type domains use latin hypercube sampling instead of repeated random draws
            which guarantees that no non-uniform clusters may arise (in subspaces) versus this method
            which treats all draws independently.

        :return: point in domain
        :rtype: array of float64 with shape (dim)

        """
        pass

    @abstractmethod
    def generate_uniform_random_points_in_domain(self, num_points, random_source):
        r"""Generate AT MOST ``num_points`` uniformly distributed points from the domain.

        .. NOTE::
             The number of points returned may be LESS THAN ``num_points``!

        Implementations may use rejection sampling. In such cases, generating the requested
        number of points may be unreasonably slow, so implementers are allowed to generate
        fewer than ``num_points`` results.

        :param num_points: max number of points to generate
        :type num_points: int >= 0
        :param random_source:
        :type random_source: callable yielding uniform random numbers in [0,1]
        :return: uniform random sampling of points from the domain; may be fewer than ``num_points``!
        :rtype: array of float64 with shape (num_points_generated, dim)

        """
        pass

    @abstractmethod
    def compute_update_restricted_to_domain(self, max_relative_change, current_point, update_vector):
        r"""Compute a new update so that CheckPointInside(``current_point`` + ``new_update``) is true.

        Changes new_update_vector so that:
          ``point_new = point + new_update_vector``

        has coordinates such that ``CheckPointInside(point_new)`` returns true.

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
        pass
