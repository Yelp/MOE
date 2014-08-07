# -*- coding: utf-8 -*-
"""Classes (Python) to compute the Bandit UCB arm allocation and choosing the arm to pull next.

See :class:`moe.bandit.interfaces.bandit_interface` for further details on bandit.

"""
import copy

from moe.bandit.interfaces.bandit_interface import BanditInterface


class UCB(BanditInterface):

    r"""Implementation of the constructor of UCB. Abstract method allocate_arms implemented in subclass.

    A class to encapsulate the computation of bandit UCB.
    The Algorithm: http://moodle.technion.ac.il/pluginfile.php/192340/mod_resource/content/0/UCB.pdf

    See :class:`moe.bandit.interfaces.bandit_interface` docs for further details.

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
