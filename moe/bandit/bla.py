# -*- coding: utf-8 -*-
"""Classes (Python) to compute the Bandit BLA arm allocation and choosing the arm to pull next.

See :class:`moe.bandit.interfaces.bandit_interface` for further details on bandit.

"""
import copy
import random

from moe.bandit.interfaces.bandit_interface import BanditInterface
from moe.bandit.utils import get_winning_arm_names_from_payoff_arm_name_list, get_equal_arm_allocations


class BLA(BanditInterface):

    r"""Implementation of the constructor of BLA and method allocate_arms.

    A class to encapsulate the computation of bandit BLA.
    The Algorithm is from the paper: A Generic Solution to Multi-Armed Bernoulli Bandit Problems, Norheim, Bradland, Granmo, OOmmen (2010) ICAART.

    See :class:`moe.bandit.interfaces.bandit_interface` docs for further details.

    """

    def __init__(
            self,
            historical_info,
            subtype=None,
    ):
        """Construct a BLA object.

        :param historical_info: a dictionary of arms sampled
        :type historical_info: dictionary of (str, SampleArm()) pairs (see :class:`moe.bandit.data_containers.SampleArm` for more details)
        :param subtype: subtype of the BLA bandit algorithm (default: None)
        :type subtype: str

        """
        self._historical_info = copy.deepcopy(historical_info)
        self._subtype = subtype

    def get_bla_payoff(self, sampled_arm):
        r"""Compute the expected upper confidence bound payoff using the BLA subtype formula.

        

        :param sampled_arm: a sampled arm
        :type sampled_arm: :class:`moe.bandit.data_containers.SampleArm`
        :return: bla payoff
        :rtype: float64
        :raise: ValueError when ``sampled_arm`` is empty.

        """
        if not sampled_arm:
            raise ValueError('sampled_arm is empty!')
        return random.betavariate(sampled_arm.win + 1, sampled_arm.total - sampled_arm.win + 1)

    def allocate_arms(self):
        r"""Compute the allocation to each arm given ``historical_info``, running bandit ``subtype`` endpoint.

        Computes the allocation to each arm based on the given subtype, and, historical info.

        Works with k-armed bandits (k >= 1).

        The Algorithm is from the paper: A Generic Solution to Multi-Armed Bernoulli Bandit Problems, Norheim, Bradland, Granmo, OOmmen (2010) ICAART.

        If there is at least one unsampled arm, this method will choose to pull the unsampled arm
        (randomly choose an unsampled arm if there are multiple unsampled arms).
        If all arms are pulled at least once, this method will pull the optimal arm
        (best expected upper confidence bound payoff).

        See :func:`moe.bandit.bla.BLA.get_bla_payoff` for details on how to compute the expected upper confidence bound payoff (expected BLA payoff)

        In case of a tie, the method will split the allocation among the optimal arms.
        For example, if we have three arms (arm1, arm2, and arm3) with expected BLA payoff 0.5, 0.5, and 0.1 respectively.
        We split the allocation between the optimal arms arm1 and arm2.

        ``{arm1: 0.5, arm2: 0.5, arm3: 0.0}``

        :return: the dictionary of (arm, allocation) key-value pairs
        :rtype: a dictionary of (str, float64) pairs
        :raise: ValueError when ``sample_arms`` are empty.

        """
        arms_sampled = self._historical_info.arms_sampled
        if not arms_sampled:
            raise ValueError('sample_arms are empty!')

        return get_equal_arm_allocations(arms_sampled, self.get_winning_arm_names(arms_sampled))

    def get_winning_arm_names(self, arms_sampled):
        r"""Compute the set of winning arm names based on the given ``arms_sampled``..

        Throws an exception when arms_sampled is empty.

        :param arms_sampled: a dictionary of arm name to :class:`moe.bandit.data_containers.SampleArm`
        :type arms_sampled: dictionary of (str, SampleArm()) pairs
        :return: set of names of the winning arms
        :rtype: frozenset(str)
        :raise: ValueError when ``arms_sampled`` are empty.

        """
        if not arms_sampled:
            raise ValueError('arms_sampled is empty!')

        bla_payoff_arm_name_list = []
        for arm_name, sampled_arm in arms_sampled.iteritems():
            bla_payoff_arm_name_list.append((self.get_bla_payoff(sampled_arm), arm_name))

        return get_winning_arm_names_from_payoff_arm_name_list(bla_payoff_arm_name_list)
