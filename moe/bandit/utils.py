# -*- coding: utf-8 -*-
"""Utilities for bandit."""


def get_winning_arm_names_from_payoff_arm_name_list(payoff_arm_name_list):
        r"""Compute the set of winning arm names based on the given ``payoff_arm_name_list``..

        Throws an exception when payoff_arm_name_list is empty.

        :param payoff_arm_name_list: a list of (payoff, arm name) tuples
        :type payoff_arm_name_list: list of (float64, str) tuples
        :return: of set of names of the winning arms
        :rtype: frozenset(str)
        :raise: ValueError when ``payoff_arm_name_list`` are empty.

        """
        if not payoff_arm_name_list:
            raise ValueError('payoff_arm_name_list is empty!')

        best_payoff, _ = max(payoff_arm_name_list)

        # Filter out arms that have payoff less than the best payoff
        winning_arm_payoff_name_list = [payoff_arm_name for payoff_arm_name in payoff_arm_name_list if payoff_arm_name[0] == best_payoff]
        # Extract a list of winning arm names from a list of (payoff, arm name) tuples.
        _, winning_arm_name_list = list(map(list, list(zip(*winning_arm_payoff_name_list))))
        winning_arm_names = frozenset(winning_arm_name_list)
        return winning_arm_names


def get_equal_arm_allocations(arms_sampled, winning_arm_names=None):
    r"""Split allocations equally among the given ``winning_arm_names``. If no ``winning_arm_names`` given, split allocations among ``arms_sampled``.

    Throws an exception when arms_sampled is empty.

    :param arms_sampled: a dictionary of arm name to :class:`moe.bandit.data_containers.SampleArm`
    :type arms_sampled: dictionary of (str, SampleArm()) pairs
    :param: winning_arm_names: a set of names of the winning arms
    :type: winning_arm_names: frozenset(str)
    :return: the dictionary of (arm, allocation) key-value pairs
    :rtype: a dictionary of (str, float64) pairs
    :raise: ValueError when ``arms_sampled`` are empty.

    """
    if not arms_sampled:
        raise ValueError('arms_sampled is empty!')

    # If no ``winning_arm_names`` given, split allocations among ``arms_sampled``.
    if winning_arm_names is None:
        winning_arm_names = frozenset([arm_name for arm_name in arms_sampled.keys()])

    num_winning_arms = len(winning_arm_names)
    arms_to_allocations = {}

    winning_arm_allocation = 1.0 / num_winning_arms
    # Split allocation among winning arms, all other arms get allocation of 0.
    for arm_name in arms_sampled.keys():
        arms_to_allocations[arm_name] = winning_arm_allocation if arm_name in winning_arm_names else 0.0

    return arms_to_allocations
