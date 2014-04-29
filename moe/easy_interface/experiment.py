# -*- coding: utf-8 -*-
"""Classes for MOE optimizable experiments."""
import pprint
from collections import namedtuple

import numpy

ExperimentPoint = namedtuple('ExperimentPoint', ['point', 'value', 'value_var'])


class Experiment(object):

    """A class for MOE optimizable experiments."""

    def __init__(self, domain):
        """Construct a MOE optimizable experiment.

        Required arguments:
            domain - the domain of the experiment

        """
        self.domain = domain
        self.points_sampled = []
        self.best_point = None

    def __dict__(self):
        """Construct a json serializeable and MOE REST recognizeable dictionary of the experiment."""
        return {
                'domain': self.domain,
                'points_sampled': [point._asdict() for point in self.points_sampled],
                }

    def __str__(self):
        """Return a pprint formated version of the experiment dict."""
        return pprint.pformat(self.__dict__())

    def add_point(self, point_in_domain, value, value_var=0.0):
        """Add a point to the experiment."""
        point = ExperimentPoint(point_in_domain, value, value_var)

        if self.best_point is not None:
            if value < self.best_point.value:
                self.best_point = point
        else:
            self.best_point = point

        self.points_sampled.append(point)

    def generate_uniform_stencil(self, size_of_stencil=3):
        """Generate a uniform stencil to sample.

        This makes the assumption that the domain is a hypercube.
        """
        raw_grid = []
        for dim_domain in self.domain:
            raw_grid.append(
                    numpy.linspace(
                        dim_domain[0],
                        dim_domain[1],
                        num=size_of_stencil
                        )
                    )

        mesh_grid = numpy.meshgrid(*raw_grid)

        points = [[] for _ in xrange(size_of_stencil ** len(self.domain))]
        for dim_grid in mesh_grid:
            flat_grid = dim_grid.flatten()
            for point_idx, point in enumerate(flat_grid):
                points[point_idx].append(point)

        return points
