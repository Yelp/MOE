# -*- coding: utf-8 -*-
"""Some default configuration parameters for bandit components."""
DEFAULT_HISTORICAL_INFO = {
                           "arms_sampled": {
                                            "arm1": {"win": 20, "loss": 5, "total": 25},
                                            "arm2": {"win": 20, "loss": 10, "total": 30},
                                            "arm3": {"win": 0, "loss": 0, "total": 0},
                                            }
                           }

DEFAULT_EPSILON = 0.05

# Epsilon subtypes
GREEDY = 'greedy'
