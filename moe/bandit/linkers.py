# -*- coding: utf-8 -*-
"""Links between the implementations of bandit algorithms."""
from collections import namedtuple

from moe.bandit.constant import BANDIT_EPSILON_ENDPOINT, BANDIT_UCB_ENDPOINT, EPSILON_SUBTYPE_FIRST, EPSILON_SUBTYPE_GREEDY, EPSILON_SUBTYPES, UCB_SUBTYPE_1, UCB_SUBTYPES
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


BANDIT_ENDPOINTS_TO_SUBTYPES = {
        BANDIT_EPSILON_ENDPOINT: EPSILON_SUBTYPES,
        BANDIT_UCB_ENDPOINT: UCB_SUBTYPES,
        }
