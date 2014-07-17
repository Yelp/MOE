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

        In case of a tie, the method will split the probability 1-epsilon among the optimal arms
        and with probability epsilon, a random arm will be pulled.
        For example, if we have three arms, two arms (arm1 and arm2) with an average payoff of 0.5 and a new arm
        (arm3, average payoff is 0). Let the epsilon be 0.12. The allocation will be as follows:
        arm1: 0.48, arm2: 0.48, arm3: 0.04. The calculation is as follows:
        arm1 and arm2 both get the same allocation: (1-epsilon)/2 + epsilon/3 = (1-0.12)/2 + 0.12/3 = 0.44 + 0.04.
        arm3 gets the allocation epsilon/3 = 0.12/3 = 0.04.
        

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
        
        winning_arm_name_list = []
        winning_avg_payoff, _ = avg_payoff_arm_name_list[0]
        for avg_payoff, arm_name in avg_payoff_arm_name_list:
            if avg_payoff < winning_avg_payoff:
                break
            winning_arm_name_list.append(arm_name)

        num_winning_arms = len(winning_arm_name_list)
        epsilon_allocation = self._epsilon / num_arms
        arms_to_allocations = {}

        # With probability epsilon, choose a winning arm at random. Therefore, we split the allocation epsilon among all arms.
        for arm_name in arms_sampled.iterkeys():
            arms_to_allocations[arm_name] = epsilon_allocation

        # With probability 1-epsilon, split allocation among winning arms.
        winning_arm_allocation = (1.0 - self._epsilon) / num_winning_arms
        for winning_arm_name in winning_arm_name_list:
            arms_to_allocations[winning_arm_name] += winning_arm_allocation

        return arms_to_allocations
