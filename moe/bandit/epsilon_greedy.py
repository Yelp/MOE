# -*- coding: utf-8 -*-
"""Classes (Python) to compute the Bandit Epsilon-Greedy arm allocation and choosing the arm to pull next.

See interfaces/bandit_interface.py for further details on bandit.

"""
import numpy

from moe.bandit.constant import DEFAULT_EPSILON, GREEDY
from moe.bandit.epsilon import Epsilon


class EpsilonGreedy(Epsilon):

    r"""Implementation of EpsilonGreedy.

    A class to encapsulate the computation of bandit epsilon greedy.

    See superclass Epsilon for further details.

    """

    def __init__(
            self,
            historical_info,
            epsilon=DEFAULT_EPSILON,
    ):
        """Construct an EpsilonGreedy object. See superclass for details."""
        super(EpsilonGreedy, self).__init__(
            historical_info=historical_info,
            subtype=GREEDY,
            epsilon=epsilon,
            )

    def allocate_arms(self):
        """Compute the allocation to each arm given ``historical_info``, running bandit `subtype`` endpoint with hyperparameters in `hyperparameter_info``.

        Computes the allocation to each arm based on the given subtype, historical info, and hyperparameter info.

        Works with k-armed bandits (k >= 1).

        The Algorithm: http://en.wikipedia.org/wiki/Multi-armed_bandit#Approximate_solutions

        This method will pull the optimal arm (best expected return) with probability 1-epsilon
        with probability epsilon a random arm will be pulled.

        :return: the dictionary of (arm, allocation) key-value pairs
        :rtype: a dictionary of (String(), float64) pairs
        """
        arms_sampled = self._historical_info.arms_sampled
        num_arms = self._historical_info.num_arms
        if not arms_sampled:
            raise
        avg_payoff_arm_name_list = []
        for arm_name, sampled_arm in arms_sampled.iteritems():
            avg_payoff = numpy.float64(sampled_arm.win - sampled_arm.loss) / sampled_arm.total if sampled_arm.total > 0 else 0
            avg_payoff_arm_name_list.append((avg_payoff, arm_name))
        avg_payoff_arm_name_list.sort(reverse=True)
        _, winning_arm = avg_payoff_arm_name_list[0]
        epsilon_allocation = self._epsilon / num_arms
        arms_to_allocations = {}
        for arm_name in arms_sampled.iterkeys():
            arms_to_allocations[arm_name] = epsilon_allocation
        arms_to_allocations[winning_arm] += 1.0 - self._epsilon
        return arms_to_allocations
