# -*- coding: utf-8 -*-
"""Utilities for bandit."""
import numpy

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
        winning_arm_names = frozenset([arm_name for arm_name in arms_sampled.iterkeys()])

    num_winning_arms = len(winning_arm_names)
    arms_to_allocations = {}

    winning_arm_allocation = 1.0 / num_winning_arms
    # Split allocation among winning arms, all other arms get allocation of 0.
    for arm_name in arms_sampled.iterkeys():
        arms_to_allocations[arm_name] = winning_arm_allocation if arm_name in winning_arm_names else 0.0

    return arms_to_allocations
