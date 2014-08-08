# -*- coding: utf-8 -*-
"""Some default configuration parameters for bandit components."""
DEFAULT_BANDIT_HISTORICAL_INFO = {
                           "arms_sampled": {
                                            "arm1": {"win": 20, "loss": 5, "total": 25},
                                            "arm2": {"win": 20, "loss": 10, "total": 30},
                                            "arm3": {"win": 0, "loss": 0, "total": 0},
                                            }
                           }

# Epsilon subtypes
EPSILON_SUBTYPE_FIRST = 'first'
EPSILON_SUBTYPE_GREEDY = 'greedy'
DEFAULT_EPSILON_SUBTYPE = EPSILON_SUBTYPE_GREEDY
EPSILON_SUBTYPES = [
                EPSILON_SUBTYPE_FIRST,
                EPSILON_SUBTYPE_GREEDY,
                ]

# UCB subtypes
UCB_SUBTYPE_1 = 'UCB1'
DEFAULT_UCB_SUBTYPE = UCB_SUBTYPE_1
UCB_SUBTYPES = [
                UCB_SUBTYPE_1,
                ]

# Default Hyperparameters
DEFAULT_EPSILON = 0.05
DEFAULT_TOTAL_SAMPLES = 100
EPSILON_SUBTYPES_TO_DEFAULT_HYPERPARAMETER_INFOS = {
        EPSILON_SUBTYPE_FIRST: {'epsilon': DEFAULT_EPSILON,
                                'total_samples': DEFAULT_TOTAL_SAMPLES},
        EPSILON_SUBTYPE_GREEDY: {'epsilon': DEFAULT_EPSILON},
        }

# Bandit Endpoints
BANDIT_EPSILON_ENDPOINT = 'epsilon_endpoint'
BANDIT_UCB_ENDPOINT = 'ucb_endpoint'
BANDIT_ENDPOINTS = [
                    BANDIT_UCB_ENDPOINT,
                    BANDIT_EPSILON_ENDPOINT,
                    ]
