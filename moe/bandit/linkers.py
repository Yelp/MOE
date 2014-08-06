# -*- coding: utf-8 -*-
"""Links between the implementations of bandit algorithms."""
from collections import namedtuple

from moe.bandit.constant import EPSILON_SUBTYPE_FIRST, EPSILON_SUBTYPE_GREEDY, UCB_SUBTYPE_1
from moe.bandit.epsilon_first import EpsilonFirst
from moe.bandit.epsilon_greedy import EpsilonGreedy
from moe.bandit.ucb1 import UCB1

BanditMethod = namedtuple(
        'BanditMethod',
        [
            'subtype',
            'bandit_class',
            ],
        )


EPSILON_SUBTYPES_TO_BANDIT_METHODS = {
        EPSILON_SUBTYPE_FIRST: BanditMethod(
            subtype=EPSILON_SUBTYPE_FIRST,
            bandit_class=EpsilonFirst,
            ),
        EPSILON_SUBTYPE_GREEDY: BanditMethod(
            subtype=EPSILON_SUBTYPE_GREEDY,
            bandit_class=EpsilonGreedy,
            ),
        }


UCB_SUBTYPES_TO_BANDIT_METHODS = {
        UCB_SUBTYPE_1: BanditMethod(
            subtype=UCB_SUBTYPE_1,
            bandit_class=UCB1,
            ),
        }
