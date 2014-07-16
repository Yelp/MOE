# -*- coding: utf-8 -*-
r"""Interface for computation of Bandit."""
import random

from abc import ABCMeta, abstractmethod


class BanditInterface(object):

    r"""Interface for a bandit algorithm.

    Abstract class to enable bandit functions--supports allocate arms and choose arm.

    Implementers of this ABC are required to manage their own hyperparameters.

    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def allocate_arms(self):
        r"""Compute the allocation to each arm given ``historical_info``, running bandit `subtype`` endpoint with hyperparameters in `hyperparameter_info``.

        Computes the allocation to each arm based on the given subtype, historical info, and hyperparameter info.

        :return: the dictionary of (arm, allocation) key-value pairs
        :rtype: a dictionary of (String(), float64) pairs

        """
        pass

    def choose_arm(self):
        r"""First compute the allocation to each arm, then choose the arm based on allocation information.

        Throws an exception when no arm is given in historical info.
        :return: name of the chosen arm
        :rtype: String()

        """
        arms_to_allocations = self.allocate_arms()
        rand = random.random()
        cumulative_probability = 0.0
        arm_chosen = None
        for arm, allocation in arms_to_allocations.iteritems():
            cumulative_probability += allocation
            if rand <= cumulative_probability:
                arm_chosen = arm
                break
        if arm_chosen is None:
            arm_chosen = arm
        return arm_chosen
