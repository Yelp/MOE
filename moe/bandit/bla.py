# -*- coding: utf-8 -*-
"""Classes (Python) to compute the Bandit BLA (Bayesian Learning Automaton) arm allocation and choosing the arm to pull next.

See :class:`moe.bandit.interfaces.bandit_interface` for further details on bandit.

"""
import copy
import random

from moe.bandit.constant import DEFAULT_BLA_SUBTYPE
from moe.bandit.data_containers import BernoulliArm
from moe.bandit.interfaces.bandit_interface import BanditInterface
from moe.bandit.utils import get_winning_arm_names_from_payoff_arm_name_list, get_equal_arm_allocations


class BLA(BanditInterface):

    r"""Implementation of the constructor of BLA (Bayesian Learning Automaton) and method allocate_arms.

    A class to encapsulate the computation of bandit BLA.
    The Algorithm is from the paper: A Generic Solution to Multi-Armed Bernoulli Bandit Problems, Norheim, Bradland, Granmo, OOmmen (2010) ICAART.

    See :class:`moe.bandit.interfaces.bandit_interface` docs for further details.

    """

    def __init__(
            self,
            historical_info,
            subtype=DEFAULT_BLA_SUBTYPE,
            random_seed=None,
    ):
        """Construct a BLA object. BLA only supports Bernoulli trials (payoff 1 for success and 0 for failure).

        :param historical_info: a dictionary of arms sampled
        :type historical_info: dictionary of (str, SampleArm()) pairs (see :class:`moe.bandit.data_containers.SampleArm` for more details)
        :param subtype: subtype of the BLA bandit algorithm (default: :const:`~moe.bandit.constant.DEFAULT_BLA_SUBTYPE`)
        :type subtype: str
        :param random_seed: for testing only (default: None), this flag allows us to provide the seed and get deterministic results for BLA
        :type subtype: float

        :raises ValueError: if the arm is not a valid Bernoulli arm

        """
        self._historical_info = copy.deepcopy(historical_info)
        self._subtype = subtype
        self._random_seed = random_seed
        # Validate that every arm is a Bernoulli arm.
        for arm in self._historical_info.arms_sampled.itervalues():
            if not isinstance(arm, BernoulliArm):
                raise ValueError('All arms have to be Bernoulli arms!')

    def get_bla_payoff(self, sampled_arm):
        r"""Compute the BLA payoff using the BLA subtype formula.

        BLA payoff is computed by sampling from a beta distribution :math`Beta(\alpha, \beta)`
        with :math:`\alpha = number\_wins + 1` and
        :math:`\beta = number\_losses + 1 = number\_total - number\_wins + 1`.

        Note that for an unsampled_arm, :math`Beta(1, 1)` is a uniform distribution.
        Learn more about beta distribution at http://en.wikipedia.org/wiki/Beta_distribution.

        :param sampled_arm: a sampled arm
        :type sampled_arm: :class:`moe.bandit.data_containers.SampleArm`
        :return: bla payoff
        :rtype: float64
        :raise: ValueError when ``sampled_arm`` is empty.

        """
        if not sampled_arm:
            raise ValueError('sampled_arm is empty!')
        # For testing only, set the seed so that the results are deterministic.
        if self._random_seed is not None:
            random.seed(self._random_seed)
        return random.betavariate(sampled_arm.win + 1, sampled_arm.total - sampled_arm.win + 1)

    def allocate_arms(self):
        r"""Compute the allocation to each arm given ``historical_info``, running bandit ``subtype`` endpoint.

        Computes the allocation to each arm based on the given subtype, and, historical info.

        Works with k-armed bandits (k >= 1).

        The Algorithm is from the paper: A Generic Solution to Multi-Armed Bernoulli Bandit Problems, Norheim, Bradland, Granmo, OOmmen (2010) ICAART.
        The original algorithm handles k = 2. We extended the algorithm naturally to handle k >= 1.

        This method will pull the optimal arm (best BLA payoff).

        See :func:`moe.bandit.bla.BLA.get_bla_payoff` for details on how to compute the BLA payoff

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

        bla_payoff_arm_name_list = [(self.get_bla_payoff(sampled_arm), arm_name) for arm_name, sampled_arm in arms_sampled.iteritems()]
        return get_winning_arm_names_from_payoff_arm_name_list(bla_payoff_arm_name_list)
