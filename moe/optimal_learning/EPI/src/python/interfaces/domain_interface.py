# -*- coding: utf-8 -*-
"""Interface for a domain: in/out test, random point generation, and update limiting (for constrained optimization)."""
from abc import ABCMeta, abstractmethod, abstractproperty

class DomainInterface(object):

    """Interface for a domain: in/out test, random point generation, and update limiting (for constrained optimization)."""

    __metaclass__ = ABCMeta

    @abstractproperty
    def dim(self):
        """Return the number of spatial dimensions."""
        pass

    @abstractmethod
    def check_point_inside(self, point):
        r"""Check if a point is inside the domain/on its boundary or outside.

        :param point: point to check
        :type point: 1d array[dim] of double
        :param points_to_sample: array of points which are being sampled concurrently (i.e., p in q,p-EI)
        :type points_to_sample: 2d array[num_to_sample][dim] of double
        :return: true if point is inside the domain
        :rtype: bool

        """
        pass

    @abstractmethod
    def generate_uniform_points_in_domain(self, num_points, random_source):
        r"""Generate AT MOST ``num_points`` uniformly distributed points from the domain.

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
        :rtype: 2d array[num_points_generated][dim] of double
        
        """
        pass

    @abstractmethod
    def limit_update(max_relative_change, current_point, update_vector):
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
        :type max_relative_change: double in (0, 1]
        :param current_point: starting point
        :type current_point: 1d array[dim] of double
        :param update_vector: proposed update
        :type update_vector: 1d array[dim] of double
        :return: new update so that the final point remains inside the domain
        :rtype: 1d array[dim] of double

        """
        pass
