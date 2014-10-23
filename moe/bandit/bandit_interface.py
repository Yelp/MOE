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

    @staticmethod
    def choose_arm(arms_to_allocations):
        r"""Choose the arm based on allocation information given in ``arms_to_allocations``.

        Throws an exception when 'arms_to_allocations' is empty.
        Implementers of this interface will never override this method.

        :param arms_to_allocations: the dictionary of (arm, allocation) key-value pairs
        :rtype arms_to_allocations: a dictionary of (str, float64) pairs
        :return: name of the chosen arm
        :rtype: str

        """
        if not arms_to_allocations:
            raise ValueError('arms_to_allocations is empty!')

        allocations = numpy.array(arms_to_allocations.values())
        # The winning arm is chosen based on the distribution of arm allocations.
        winner = numpy.argmax(numpy.random.dirichlet(allocations))
        # While the internal order of a dict is unknowable a priori, the order presented by the various iterators
        # and list-ify methods is always the same as long as the dict is not modified between calls to these methods.
        return arms_to_allocations.keys()[winner]
