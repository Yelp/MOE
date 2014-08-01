# -*- coding: utf-8 -*-
"""Classes (Python) to compute the Bandit Epsilon-Greedy arm allocation and choosing the arm to pull next.

See :class:`moe.bandit.epsilon.Epsilon` for further details on this bandit.

"""
from moe.bandit.constant import DEFAULT_EPSILON, EPSILON_SUBTYPE_GREEDY
from moe.bandit.epsilon import Epsilon


class EpsilonGreedy(Epsilon):

    r"""Implementation of EpsilonGreedy.

    A class to encapsulate the computation of bandit epsilon greedy.

    See superclass :class:`moe.bandit.epsilon.Epsilon` for further details.

    """

    def __init__(
            self,
            historical_info,
            epsilon=DEFAULT_EPSILON,
    ):
        """Construct an EpsilonGreedy object. See superclass :class:`moe.bandit.epsilon.Epsilon` for details."""
        super(EpsilonGreedy, self).__init__(
            historical_info=historical_info,
            subtype=EPSILON_SUBTYPE_GREEDY,
            epsilon=epsilon,
            )

    def allocate_arms(self):
        r"""Compute the allocation to each arm given ``historical_info``, running bandit ``subtype`` endpoint with hyperparameter epsilon.

        Computes the allocation to each arm based on the given subtype, historical info, and hyperparameter epsilon.

        Works with k-armed bandits (k >= 1).

        The Algorithm: http://en.wikipedia.org/wiki/Multi-armed_bandit#Approximate_solutions

        This method will pull the optimal arm (best expected return) with probability :math:`1-\epsilon`,
        with probability :math:`\epsilon` a random arm will be pulled.

        In case of a tie, the method will split the probability :math:`1-\epsilon` among the optimal arms
        and with probability :math:`\epsilon`, a random arm will be pulled.
        For example, if we have three arms, two arms (arm1 and arm2) with an average payoff of 0.5 and a new arm
        (arm3, average payoff is 0). Let the epsilon :math:`\epsilon` be 0.12. The allocation will be as follows:
        arm1: 0.48, arm2: 0.48, arm3: 0.04.

        The calculation is as follows:

        arm1 and arm2 both get the same allocation:

        .. math:: \frac{1-\epsilon}{2} + \frac{\epsilon}{3} = \frac{1-0.12}{2} + \frac{0.12}{3} = 0.44 + 0.04 = 0.48

        arm3 gets the allocation:

        .. math:: \frac{\epsilon}{3} = \frac{0.12}{3} = 0.04

        :return: the dictionary of (arm, allocation) key-value pairs
        :rtype: a dictionary of (String(), float64) pairs
        :raise: ValueError when ``sample_arms`` are empty.

        """
        arms_sampled = self._historical_info.arms_sampled
        num_arms = self._historical_info.num_arms
        if not arms_sampled:
            raise ValueError('sample_arms are empty!')

        winning_arm_names = self.get_winning_arm_names(arms_sampled)

        num_winning_arms = len(winning_arm_names)
        epsilon_allocation = self._epsilon / num_arms
        arms_to_allocations = {}

        # With probability epsilon, choose a winning arm at random. Therefore, we split the allocation epsilon among all arms.
        for arm_name in arms_sampled.iterkeys():
            arms_to_allocations[arm_name] = epsilon_allocation

        # With probability 1-epsilon, split allocation among winning arms.
        winning_arm_allocation = (1.0 - self._epsilon) / num_winning_arms
        for winning_arm_name in winning_arm_names:
            arms_to_allocations[winning_arm_name] += winning_arm_allocation

        return arms_to_allocations
