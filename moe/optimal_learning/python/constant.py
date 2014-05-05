# -*- coding: utf-8 -*-
"""Some default configuration parameters for optimal_learning components."""
from collections import namedtuple

# EI Defaults
ExpectedImprovementParameters = namedtuple(
        'ExpectedImprovementParameters',
        [
            'mc_iterations',
            ],
        )

default_expected_improvement_parameters = ExpectedImprovementParameters(
        mc_iterations=100000,
        )

# GP Defaults
GaussianProcessParameters = namedtuple(
        'GaussianProcessParameters',
        [
            'length_scale',
            'signal_variance',
            ],
        )

default_gaussian_process_parameters = GaussianProcessParameters(
    length_scale=[0.2],
    signal_variance=1.0,
    )

# EI Optimization defaults
EIOptimizationParameters = namedtuple(
    'EIOptimizationParameters',
    [
        'num_multistarts',
        'gd_iterations',
        'max_num_restarts',
        'gamma',
        'pre_mult',
        'mc_iterations',
        'max_relative_change',
        'tolerance',
        ],
    )

CONSTANT_LIAR_MIN = 'constant_liar_min'
CONSTANT_LIAR_MAX = 'constant_liar_max'
CONSTANT_LIAR_MEAN = 'constant_liar_mean'

CONSTANT_LIAR_METHODS = [
        CONSTANT_LIAR_MIN,
        CONSTANT_LIAR_MAX,
        CONSTANT_LIAR_MEAN,
        ]

GRADIENT_DESCENT_OPTIMIZER = 'gradient_descent'

ALL_OPTIMIZERS = [
        GRADIENT_DESCENT_OPTIMIZER,
        ]

default_optimizer_type = GRADIENT_DESCENT_OPTIMIZER
default_num_random_samples = 4000

default_ei_optimization_parameters = EIOptimizationParameters(
    num_multistarts=40,
    gd_iterations=1000,
    max_num_restarts=3,
    gamma=0.9,
    pre_mult=1.0,
    mc_iterations=100000,
    max_relative_change=1.0,
    tolerance=1.0e-7,
    )
