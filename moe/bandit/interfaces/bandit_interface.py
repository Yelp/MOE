# -*- coding: utf-8 -*-
r"""Interface for Bandit functions, supports allocate arms and choose arm."""
import numpy

from abc import ABCMeta, abstractmethod


class BanditInterface(object):

    r"""Interface for a bandit algorithm.

    Abstract class to enable bandit functions, supports allocate arms and choose arm.
    Implementers of this interface will never override the method choose_arm.

    Implementers of this ABC are required to manage their own hyperparameters.

    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def allocate_arms(self):
        r"""Compute the allocation to each arm given ``historical_info``, running bandit `subtype`` endpoint with hyperparameters in `hyperparameter_info``.

        Computes the allocation to each arm based on the given subtype, historical info, and hyperparameter info.

        :return: the dictionary of (arm, allocation) key-value pairs
        :rtype: a dictionary of (str, float64) pairs

        """
        pass

    def choose_arm(self):
        r"""First compute the allocation to each arm, then choose the arm based on allocation information.

        Throws an exception when no arm is given in historical info.
        Implementers of this interface will never override this method.

        :return: name of the chosen arm
        :rtype: str

        """
        arms_to_allocations = self.allocate_arms()
        # Generate a numpy array of type (string, float) pairs of arm name and its allocation
        # allocations['arms'] is an array of arm names, allocations['allocation'] is an array of allocations
        allocations = numpy.array([(arm, allocation) for arm, allocation in arms_to_allocations.iteritems()], dtype=([('arm', '|S256'), ('allocation', float)]))
        # The winning arm is chosen based on the distribution of arm allocations.
        winner = numpy.argmax(numpy.random.dirichlet(allocations['allocation']))
        return allocations['arm'][winner]
