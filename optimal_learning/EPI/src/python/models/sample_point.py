# -*- coding: utf-8 -*-
# Copyright (C) 2012 Scott Clark. All rights reserved.

import logging


logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class SamplePoint(object):
    """A point sampled from an unknown function

    :Attributes:
        - point: The point sampled (in the domain of the function)
        - value: The value returned by the function
        - value_std: The std (if any) of the value returned
    """
    def __init__(self, point, value, value_var=0.0):
        """Inits SamplePoint with a point and value (and maybe a value_std=0.0)"""
        self.point = point
        self.value = value
        self.value_var = value_var
        logging.debug("Created point: %s" % (self.__str__()))

    def __str__(self):
        return "SamplePoint: f(%s) = %f +/- %f" % (str(self.point), self.value, self.value_var)

    def euclidean_distance_from_point(self, other_point):
        raise(NotImplementedError)
