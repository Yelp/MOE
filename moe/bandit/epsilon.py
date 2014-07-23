# -*- coding: utf-8 -*-
"""Classes (Python) to compute the Bandit Epsilon arm allocation and choosing the arm to pull next.

See :class:`moe.bandit.interfaces.bandit_interface` for further details on bandit.

"""

from moe.bandit.constant import DEFAULT_EPSILON
from moe.bandit.interfaces.bandit_interface import BanditInterface


class Epsilon(BanditInterface):

    r"""Implementation of Epsilon.

    A class to encapsulate the computation of bandit epsilon.

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
        :param hyperparameter_info: a dictionary of hyperparameter information, (hyperparameter name, hyper parameter value) key-value pairs
        :type hyperparameter_info: dictionary of (String(), *) pairs

        """
        self._historical_info = historical_info
        self._subtype = subtype
        self._epsilon = epsilon
