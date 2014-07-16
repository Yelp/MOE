# -*- coding: utf-8 -*-
"""Links between the implementations of bandit algorithms."""
from collections import namedtuple

from moe.bandit.constant import GREEDY
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
        GREEDY: EpsilonMethod(
            subtype=GREEDY,
            bandit_class=EpsilonGreedy,
            ),
        }
