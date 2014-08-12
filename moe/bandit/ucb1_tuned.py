# -*- coding: utf-8 -*-
"""Classes (Python) to compute the Bandit UCB1-tuned arm allocation and choosing the arm to pull next.

See :class:`moe.bandit.ucb.UCB` for further details on this bandit.

"""
import math

import numpy

from moe.bandit.constant import UCB_SUBTYPE_1_TUNED
from moe.bandit.ucb import UCB


class UCB1Tuned(UCB):

    r"""Implementation of UCB1-tuned.

    A class to encapsulate the computation of bandit UCB1-tuned.
    See :func:`moe.bandit.ucb.UCB.allocate_arms` for more details on how UCB allocates arms.

    See superclass :class:`moe.bandit.ucb.UCB` for further details.

    """

    def __init__(
            self,
            historical_info,
    ):
        """Construct an UCB1-tuned object. See superclass :class:`moe.bandit.ucb.UCB` for details."""
        super(UCB1Tuned, self).__init__(
            historical_info=historical_info,
            subtype=UCB_SUBTYPE_1_TUNED,
            )

    def get_ucb_payoff(self, sampled_arm, number_sampled):
        r"""Compute the expected upper confidence bound payoff using the UCB1-tuned formula.

        The upper confidence bound for the variance of machine *j* :math:`v_j(n_j)` is computed as follows:

        .. math:: v_j(n_j) = \sigma^2 + \sqrt{\frac{2 \ln n}{n_j}}

        where :math:`\sigma^2` is the sample variance of arm *j* (the given ``sampled_arm``), :math:`n_j` is the number of times arm *j* has been pulled (``sampled_arm.total``),
        and *n* is overall the number of pulls so far (``number_sampled``). ``number_sampled`` (number sampled) is calculated by summing up total from each arm sampled.

        The expected upper confidence bound payoff (expected UCB payoff) is computed as follows:

        .. math:: r_j = \mu + \sqrt{\frac{\ln n}{n_j} \min \{1/4, v_j(n_j)\}}

        where :math:`\mu` is the average payoff obtained from arm *j* and 1/4 is an upper bound on the variance of a Bernoulli random variable.

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

        MAX_BERNOULLI_RANDOM_VARIABLE_VARIANCE = 0.25
        if sampled_arm.variance is None:
            # If variance is None, use the variance of Bernoulli random variable
            p = sampled_arm.win / sampled_arm.total if sampled_arm.total > 0 else 0
            # If total is 0, use the upper bound on the variance of a Bernoulli Random Variable
            variance = p * (1 - p) if sampled_arm.total > 0 else MAX_BERNOULLI_RANDOM_VARIABLE_VARIANCE
        else:
            variance = sampled_arm.variance
        avg_payoff = numpy.float64(sampled_arm.win - sampled_arm.loss) / sampled_arm.total if sampled_arm.total > 0 else 0
        return avg_payoff + math.sqrt(math.log(sampled_arm.total) / number_sampled * min(MAX_BERNOULLI_RANDOM_VARIABLE_VARIANCE, variance + math.sqrt(2.0 * math.log(sampled_arm.total) / number_sampled)))
