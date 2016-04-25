# -*- coding: utf-8 -*-
"""Classes (Python) to compute the Bandit UCB1 arm allocation and choosing the arm to pull next.

See :class:`moe.bandit.ucb.ucb_interface.UCBInterface` for further details on this bandit.

"""
import math

import numpy

from moe.bandit.constant import UCB_SUBTYPE_1
from moe.bandit.ucb.ucb_interface import UCBInterface


class UCB1(UCBInterface):

    r"""Implementation of UCB1.

    A class to encapsulate the computation of bandit UCB1.
    See :func:`moe.bandit.ucb.ucb_interface.UCBInterface.allocate_arms` for more details on how UCB allocates arms.

    See superclass :class:`moe.bandit.ucb.ucb_interface.UCBInterface` for further details.

    """

    def __init__(
            self,
            historical_info,
    ):
        """Construct an UCB1 object. See superclass :class:`moe.bandit.ucb.ucb_interface.UCBInterface` for details."""
        super(UCB1, self).__init__(
            historical_info=historical_info,
            subtype=UCB_SUBTYPE_1,
            )

    def get_ucb_payoff(self, sampled_arm, number_sampled):
        r"""Compute the expected upper confidence bound payoff using the UCB1 formula.

        The expected upper confidence bound payoff (expected UCB payoff) is computed as follows:

        .. math:: r_j = \mu + \sqrt{\frac{2 \ln n}{n_j}}

        where :math:`\mu` is the average payoff obtained from arm *j* (the given ``sampled_arm``),
        :math:`n_j` is the number of times arm *j* has been pulled (``sampled_arm.total``),
        and *n* is overall the number of pulls so far (``number_sampled``). ``number_sampled`` (number sampled)
        is calculated by summing up total from each arm sampled.

        :param sampled_arm: a sampled arm
        :type sampled_arm: :class:`moe.bandit.data_containers.SampleArm`
        :param number_sampled: the overall number of pulls so far
        :type number_sampled: int
        :return: ucb payoff
        :rtype: float64
        :raise: ValueError when ``sampled_arm`` is empty.

        """
        if not sampled_arm:
            raise ValueError('sampled_arm is empty!')

        # If number_sampled is zero, we haven't sampled any arm at all. All arms are equally likely to be the best arm.
        if number_sampled == 0:
            return 0.0

        avg_payoff = numpy.float64(sampled_arm.win - sampled_arm.loss) / sampled_arm.total if sampled_arm.total > 0 else 0
        return avg_payoff + math.sqrt(2.0 * math.log(number_sampled) / sampled_arm.total)
