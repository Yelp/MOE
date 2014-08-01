# -*- coding: utf-8 -*-
"""Classes (Python) to compute the Bandit Epsilon arm allocation and choosing the arm to pull next.

See :class:`moe.bandit.interfaces.bandit_interface` for further details on bandit.

"""

import copy

from moe.bandit.constant import DEFAULT_EPSILON
from moe.bandit.interfaces.bandit_interface import BanditInterface


class Epsilon(BanditInterface):

    r"""Implementation of the constructor of Epsilon. Abstract method allocate_arms implemented in subclass.

    A class to encapsulate the computation of bandit epsilon.
    Epsilon is the sole hyperparameter in this class. Subclasses may contain other hyperparameters.

    See :class:`moe.bandit.interfaces.bandit_interface` docs for further details.

    """

    def __init__(
            self,
            historical_info,
            subtype=None,
            epsilon=DEFAULT_EPSILON,
    ):
        """Construct an Epsilon object.

        :param historical_info: a dictionary of arms sampled
        :type historical_info: dictionary of (String(), SingleArm()) pairs
        :param subtype: subtype of the epsilon bandit algorithm (default: None)
        :type subtype: String()
        :param epsilon: epsilon hyperparameter for the epsilon bandit algorithm (default: :const:`~moe.bandit.constant.DEFAULT_EPSILON`)
        :type epsilon: float64 in range [0.0, 1.0]

        """
        self._historical_info = copy.deepcopy(historical_info)
        self._subtype = subtype
        self._epsilon = epsilon
