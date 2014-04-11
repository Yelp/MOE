# -*- coding: utf-8 -*-
from collections import namedtuple

ExperimentPoint = namedtuple('ExperimentPoint', ['point', 'value', 'value_var'])

class Experiment(object):
    """A class for MOE optimizable experiments."""

    def __init__(self, domain):
        """Constructs a MOE optimizable experiment.

        Required arguments:
            domain - the domain of the experiment

        """

        self.domain = domain
        self.points_sampled = []
        self.best_point = None

    def __dict__(self):
        """Constructs a json serializeable and MOE REST recognizeable dictionary of the experiment."""

        return {
                'domain': self.domain,
                'points_sampled': [point._asdict() for point in self.points_sampled],
                }

    def add_point(self, point, value, value_var=0.0):
        """Add a point to the experiment."""
        point = ExperimentPoint(point, value, value_var)

        if self.best_point is not None:
            if value < self.best_point.value:
                self.best_point = point
        else:
            self.best_point = point

        self.points_sampled.append(point)
