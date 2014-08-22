# -*- coding: utf-8 -*-
"""Links between the implementations of bandit algorithms."""
from collections import namedtuple

from moe.bandit.bla import BLA
from moe.bandit.constant import BANDIT_BLA_ENDPOINT, BANDIT_EPSILON_ENDPOINT, BANDIT_UCB_ENDPOINT, BLA_SUBTYPE_BLA, BLA_SUBTYPES, EPSILON_SUBTYPE_FIRST, EPSILON_SUBTYPE_GREEDY, EPSILON_SUBTYPES, UCB_SUBTYPE_1, UCB_SUBTYPE_1_TUNED, UCB_SUBTYPES
from moe.bandit.epsilon_first import EpsilonFirst
from moe.bandit.epsilon_greedy import EpsilonGreedy
from moe.bandit.ucb1 import UCB1
from moe.bandit.ucb1_tuned import UCB1Tuned
from moe.views.constant import BANDIT_BLA_ROUTE_NAME, BANDIT_EPSILON_ROUTE_NAME, BANDIT_UCB_ROUTE_NAME

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
        UCB_SUBTYPE_1_TUNED: BanditMethod(
            subtype=UCB_SUBTYPE_1_TUNED,
            bandit_class=UCB1Tuned,
            ),
        }

BLA_SUBTYPES_TO_BANDIT_METHODS = {
        BLA_SUBTYPE_BLA: BanditMethod(
            subtype=BLA_SUBTYPE_BLA,
            bandit_class=BLA,
            ),
        }

BANDIT_ENDPOINTS_TO_SUBTYPES = {
        BANDIT_BLA_ENDPOINT: BLA_SUBTYPES,
        BANDIT_EPSILON_ENDPOINT: EPSILON_SUBTYPES,
        BANDIT_UCB_ENDPOINT: UCB_SUBTYPES,
        }

BANDIT_ROUTE_NAMES_TO_SUBTYPES = {
        BANDIT_BLA_ROUTE_NAME: BLA_SUBTYPES,
        BANDIT_EPSILON_ROUTE_NAME: EPSILON_SUBTYPES,
        BANDIT_UCB_ROUTE_NAME: UCB_SUBTYPES,
        }
