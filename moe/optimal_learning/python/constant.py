# -*- coding: utf-8 -*-
"""Some default configuration parameters for optimal_learning components."""
from collections import namedtuple

import moe.optimal_learning.python.python_version.optimization as python_optimization

# Multithreading constants
DEFAULT_MAX_NUM_THREADS = 1

# Covariance constants
SQUARE_EXPONENTIAL_COVARIANCE_TYPE = 'square_exponential'

# GP Defaults
GaussianProcessParameters = namedtuple(
        'GaussianProcessParameters',
        [
            'length_scale',
            'signal_variance',
            ],
        )

DEFAULT_GAUSSIAN_PROCESS_PARAMETERS = GaussianProcessParameters(
    length_scale=[0.2],
    signal_variance=1.0,
    )

# Domain constants
TENSOR_PRODUCT_DOMAIN_TYPE = 'tensor_product'
SIMPLEX_INTERSECT_TENSOR_PRODUCT_DOMAIN_TYPE = 'simplex_intersect_tensor_product'

NULL_OPTIMIZER = 'null_optimizer'
NEWTON_OPTIMIZER = 'newton_optimizer'
GRADIENT_DESCENT_OPTIMIZER = 'gradient_descent_optimizer'

# EI Defaults
DEFAULT_EXPECTED_IMPROVEMENT_MC_ITERATIONS = 10000
TEST_EXPECTED_IMPROVEMENT_MC_ITERATIONS = 50

DEFAULT_NEWTON_MULTISTARTS = 200
DEFAULT_GRADIENT_DESCENT_MULTISTARTS = 10000
DEFAULT_OPTIMIZATION_MULTISTARTS = 10000
TEST_OPTIMIZATION_MULTISTARTS = 3
PRETTY_OPTIMIZATION_MULTISTARTS = 40

DEFAULT_OPTIMIZATION_NUM_RANDOM_SAMPLES = 4000
TEST_OPTIMIZATION_NUM_RANDOM_SAMPLES = 3

DEFAULT_NEWTON_PARAMETERS = python_optimization.NewtonParameters(
        max_num_steps=100,
        gamma=1.05,
        time_factor=1.0e-2,
        max_relative_change=1.0,
        tolerance=1.0e-9,
        )

TEST_GRADIENT_DESCENT_PARAMETERS = python_optimization.GradientDescentParameters(
        max_num_steps=5,
        max_num_restarts=2,
        num_steps_averaged=1,
        gamma=0.4,
        pre_mult=1.0,
        max_relative_change=1.0,
        tolerance=1.0e-3,
        )

DEMO_GRADIENT_DESCENT_PARAMETERS = python_optimization.GradientDescentParameters(
        max_num_steps=50,
        max_num_restarts=4,
        num_steps_averaged=0,
        gamma=0.4,
        pre_mult=1.4,
        max_relative_change=1.0,
        tolerance=1.0e-6,
        )

DEFAULT_GRADIENT_DESCENT_PARAMETERS = python_optimization.GradientDescentParameters(
        max_num_steps=400,
        max_num_restarts=10,
        num_steps_averaged=10,
        gamma=0.7,
        pre_mult=0.4,
        max_relative_change=1.0,
        tolerance=1.0e-6,
        )

OPTIMIZATION_TYPE_TO_DEFAULT_PARAMETERS = {
        NEWTON_OPTIMIZER: DEFAULT_NEWTON_PARAMETERS,
        GRADIENT_DESCENT_OPTIMIZER: DEFAULT_GRADIENT_DESCENT_PARAMETERS,
        }
