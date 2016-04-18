# -*- coding: utf-8 -*-
"""Classes (Python) to compute the Bandit Epsilon arm allocation and choosing the arm to pull next.

See :mod:`moe.bandit.bandit_interface` for further details on bandit.

"""
import copy

import numpy

from moe.bandit.constant import DEFAULT_EPSILON
from moe.bandit.bandit_interface import BanditInterface
from moe.bandit.utils import get_winning_arm_names_from_payoff_arm_name_list


class EpsilonInterface(BanditInterface):

    r"""Implementation of the constructor and common methods of Epsilon. Abstract method allocate_arms implemented in subclass.

    A class to encapsulate the computation of bandit epsilon.
    Epsilon is the sole hyperparameter in this class. Subclasses may contain other hyperparameters.

    See :mod:`moe.bandit.bandit_interface` docs for further details.

    """

    def __init__(
            self,
            historical_info,
            subtype=None,
            epsilon=DEFAULT_EPSILON,
    ):
        """Construct an Epsilon object.

        :param historical_info: a dictionary of arms sampled
        :type historical_info: dictionary of (str, SampleArm()) pairs (see :class:`moe.bandit.data_containers.SampleArm` for more details)
        :param subtype: subtype of the epsilon bandit algorithm (default: None)
        :type subtype: str
        :param epsilon: epsilon hyperparameter for the epsilon bandit algorithm (default: :const:`~moe.bandit.constant.DEFAULT_EPSILON`)
        :type epsilon: float64 in range [0.0, 1.0]

        """
        self._historical_info = copy.deepcopy(historical_info)
        self._subtype = subtype
        self._epsilon = epsilon

    @staticmethod
    def get_winning_arm_names(arms_sampled):
        r"""Compute the set of winning arm names based on the given ``arms_sampled``..

        Throws an exception when arms_sampled is empty.
        Implementers of this interface will never override this method.

        :param arms_sampled: a dictionary of arm name to :class:`moe.bandit.data_containers.SampleArm`
        :type arms_sampled: dictionary of (str, SampleArm()) pairs
        :return: of set of names of the winning arms
        :rtype: frozenset(str)
        :raise: ValueError when ``arms_sampled`` are empty.

        """
        if not arms_sampled:
            raise ValueError('arms_sampled is empty!')

        avg_payoff_arm_name_list = []
        for arm_name, sampled_arm in arms_sampled.items():
            avg_payoff = numpy.float64(sampled_arm.win - sampled_arm.loss) / sampled_arm.total if sampled_arm.total > 0 else 0
            avg_payoff_arm_name_list.append((avg_payoff, arm_name))

        return get_winning_arm_names_from_payoff_arm_name_list(avg_payoff_arm_name_list)
