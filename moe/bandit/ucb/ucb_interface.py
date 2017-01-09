# -*- coding: utf-8 -*-
"""Classes (Python) to compute the Bandit UCB (Upper Confidence Bound) arm allocation and choosing the arm to pull next.

See :mod:`moe.bandit.bandit_interface` for further details on bandit.

"""
import copy

from abc import abstractmethod

from moe.bandit.bandit_interface import BanditInterface
from moe.bandit.utils import get_winning_arm_names_from_payoff_arm_name_list, get_equal_arm_allocations


class UCBInterface(BanditInterface):

    r"""Implementation of the constructor of UCB (Upper Confidence Bound) and method allocate_arms. The method get_ucb_payoff is implemented in subclass.

    A class to encapsulate the computation of bandit UCB.
    The Algorithm: http://moodle.technion.ac.il/pluginfile.php/192340/mod_resource/content/0/UCB.pdf

    To inherit this class, a subclass needs to implement get_ucb_payoff
    (see :func:`moe.bandit.ucb.ucb1.UCB1.get_ucb_payoff` for an example), everything else is already implemented.

    See :mod:`moe.bandit.bandit_interface` docs for further details.

    """

    def __init__(
            self,
            historical_info,
            subtype=None,
    ):
        """Construct a UCB object.

        :param historical_info: a dictionary of arms sampled
        :type historical_info: dictionary of (str, SampleArm()) pairs (see :class:`moe.bandit.data_containers.SampleArm` for more details)
        :param subtype: subtype of the UCB bandit algorithm (default: None)
        :type subtype: str

        """
        self._historical_info = copy.deepcopy(historical_info)
        self._subtype = subtype

    @staticmethod
    def get_unsampled_arm_names(arms_sampled):
        r"""Compute the set of unsampled arm names based on the given ``arms_sampled``..

        Throws an exception when arms_sampled is empty.

        :param arms_sampled: a dictionary of arm name to :class:`moe.bandit.data_containers.SampleArm`
        :type arms_sampled: dictionary of (str, SampleArm()) pairs
        :return: set of names of the unsampled arms
        :rtype: frozenset(str)
        :raise: ValueError when ``arms_sampled`` are empty.

        """
        if not arms_sampled:
            raise ValueError('arms_sampled is empty!')

        unsampled_arm_name_list = [name for name, sampled_arm in arms_sampled.iteritems() if sampled_arm.total == 0]
        return frozenset(unsampled_arm_name_list)

    @abstractmethod
    def get_ucb_payoff(self, sampled_arm, number_sampled):
        r"""Compute the expected upper confidence bound payoff using the UCB subtype formula.

        See definition in subclasses for details.

        :param sampled_arm: a sampled arm
        :type sampled_arm: :class:`moe.bandit.data_containers.SampleArm`
        :param number_sampled: the overall number of pulls so far
        :type number_sampled: int
        :return: ucb payoff
        :rtype: float64
        :raise: ValueError when ``sampled_arm`` is empty.

        """
        pass

    def allocate_arms(self):
        r"""Compute the allocation to each arm given ``historical_info``, running bandit ``subtype`` endpoint.

        Computes the allocation to each arm based on the given subtype, and, historical info.

        Works with k-armed bandits (k >= 1).

        The Algorithm: http://moodle.technion.ac.il/pluginfile.php/192340/mod_resource/content/0/UCB.pdf

        If there is at least one unsampled arm, this method will choose to pull the unsampled arm
        (randomly choose an unsampled arm if there are multiple unsampled arms).
        If all arms are pulled at least once, this method will pull the optimal arm
        (best expected upper confidence bound payoff).

        See :func:`moe.bandit.ucb.ucb_interface.UCBInterface.get_ucb_payoff` for details on how to compute the expected upper confidence bound payoff (expected UCB payoff)

        In case of a tie, the method will split the allocation among the optimal arms.
        For example, if we have three arms (arm1, arm2, and arm3) with expected UCB payoff 0.5, 0.5, and 0.1 respectively.
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

        # If there exists an unsampled arm, return the names of the unsampled arms
        unsampled_arm_names = self.get_unsampled_arm_names(arms_sampled)
        if unsampled_arm_names:
            return unsampled_arm_names

        number_sampled = sum([sampled_arm.total for sampled_arm in arms_sampled.itervalues()])

        ucb_payoff_arm_name_list = [(self.get_ucb_payoff(sampled_arm, number_sampled), arm_name) for arm_name, sampled_arm in arms_sampled.iteritems()]

        return get_winning_arm_names_from_payoff_arm_name_list(ucb_payoff_arm_name_list)
