# -*- coding: utf-8 -*-
"""Links between the implementations of bandit algorithms."""
from collections import namedtuple

from moe.bandit.constant import EPSILON_SUBTYPE_FIRST, EPSILON_SUBTYPE_GREEDY
from moe.bandit.epsilon_first import EpsilonFirst
from moe.bandit.epsilon_greedy import EpsilonGreedy

# Epsilon
EpsilonMethod = namedtuple(
        'EpsilonMethod',
        [
            'subtype',
            'bandit_class',
            ],
        )


EPSILON_SUBTYPES_TO_EPSILON_METHODS = {
        EPSILON_SUBTYPE_FIRST: EpsilonMethod(
            subtype=EPSILON_SUBTYPE_FIRST,
            bandit_class=EpsilonFirst,
            ),
        EPSILON_SUBTYPE_GREEDY: EpsilonMethod(
            subtype=EPSILON_SUBTYPE_GREEDY,
            bandit_class=EpsilonGreedy,
            ),
        }
