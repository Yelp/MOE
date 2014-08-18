# -*- coding: utf-8 -*-
"""Some default configuration parameters for optimal_learning components."""
from collections import namedtuple

import moe.optimal_learning.python.python_version.optimization as python_optimization
import moe.views.constant as views_constant

# Multithreading constants
#: Default number of threads to use in computation
DEFAULT_MAX_NUM_THREADS = 4
#: Maximum number of threads that a user can specify
#: TODO(GH-301): make this a server configurable value or set appropriate openmp env var
MAX_ALLOWED_NUM_THREADS = 10000

# Covariance type names
SQUARE_EXPONENTIAL_COVARIANCE_TYPE = 'square_exponential'

#: Covariance types supported by :mod:`moe`
COVARIANCE_TYPES = [
        SQUARE_EXPONENTIAL_COVARIANCE_TYPE,
        ]

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

#: Domain types supported by :mod:`moe`
DOMAIN_TYPES = [
        TENSOR_PRODUCT_DOMAIN_TYPE,
        SIMPLEX_INTERSECT_TENSOR_PRODUCT_DOMAIN_TYPE,
        ]

# Optimizer constants
NULL_OPTIMIZER = 'null_optimizer'
NEWTON_OPTIMIZER = 'newton_optimizer'
GRADIENT_DESCENT_OPTIMIZER = 'gradient_descent_optimizer'
L_BFGS_B_OPTIMIZER = 'l_bfgs_b_optimizer'

#: Optimizer types supported by :mod:`moe`
OPTIMIZER_TYPES = [
        NULL_OPTIMIZER,
        NEWTON_OPTIMIZER,
        GRADIENT_DESCENT_OPTIMIZER,
        L_BFGS_B_OPTIMIZER,
        ]

# Likelihood constants
LEAVE_ONE_OUT_LOG_LIKELIHOOD = 'leave_one_out_log_likelihood'
LOG_MARGINAL_LIKELIHOOD = 'log_marginal_likelihood'

#: Log Likelihood types supported by :mod:`moe`
LIKELIHOOD_TYPES = [
        LEAVE_ONE_OUT_LOG_LIKELIHOOD,
        LOG_MARGINAL_LIKELIHOOD,
        ]

# EI Monte-Carlo computation defaults
DEFAULT_EXPECTED_IMPROVEMENT_MC_ITERATIONS = 10000


# Minimal parameters for testing and demos (where speed is more important than accuracy)
TEST_EXPECTED_IMPROVEMENT_MC_ITERATIONS = 50
TEST_OPTIMIZER_MULTISTARTS = 3
TEST_OPTIMIZER_NUM_RANDOM_SAMPLES = 3

TEST_GRADIENT_DESCENT_PARAMETERS = python_optimization.GradientDescentParameters(
        max_num_steps=5,
        max_num_restarts=2,
        num_steps_averaged=1,
        gamma=0.4,
        pre_mult=1.0,
        max_relative_change=1.0,
        tolerance=1.0e-3,
        )

TEST_LBFGSB_PARAMETERS = python_optimization.LBFGSBParameters(
        approx_grad=True,
        max_func_evals=3,
        max_metric_correc=10,
        factr=100000000.0,
        pgtol=1.0e1,
        epsilon=1.0e-1,
        )

DEMO_OPTIMIZER_MULTISTARTS = 50
DEMO_GRADIENT_DESCENT_PARAMETERS = python_optimization.GradientDescentParameters(
        max_num_steps=300,
        max_num_restarts=4,
        num_steps_averaged=0,
        gamma=0.6,
        pre_mult=1.0,
        max_relative_change=1.0,
        tolerance=1.0e-6,
        )

# Default optimization parameters for various combinations of optimizer (Newton, GD) and objective functions (EI, Log Likelihood)
DEFAULT_MULTISTARTS = 300  # Fix once cpp num_multistarts is consistent.

DEFAULT_NEWTON_MULTISTARTS_MODEL_SELECTION = 200
DEFAULT_NEWTON_NUM_RANDOM_SAMPLES_MODEL_SELECTION = 0
DEFAULT_NEWTON_PARAMETERS_MODEL_SELECTION = python_optimization.NewtonParameters(
        max_num_steps=150,
        gamma=1.2,
        time_factor=5.0e-4,
        max_relative_change=1.0,
        tolerance=1.0e-9,
        )

DEFAULT_NULL_NUM_RANDOM_SAMPLES_MODEL_SELECTION = 300000

DEFAULT_GRADIENT_DESCENT_MULTISTARTS_MODEL_SELECTION = 400
DEFAULT_GRADIENT_DESCENT_NUM_RANDOM_SAMPLES_MODEL_SELECTION = 0
DEFAULT_GRADIENT_DESCENT_PARAMETERS_MODEL_SELECTION = python_optimization.GradientDescentParameters(
        max_num_steps=600,
        max_num_restarts=10,
        num_steps_averaged=0,
        gamma=0.9,
        pre_mult=0.25,
        max_relative_change=0.2,
        tolerance=1.0e-5,
        )

DEFAULT_NULL_NUM_RANDOM_SAMPLES_EI_ANALYTIC = 500000

DEFAULT_GRADIENT_DESCENT_MULTISTARTS_EI_ANALYTIC = 600
DEFAULT_GRADIENT_DESCENT_NUM_RANDOM_SAMPLES_EI_ANALYTIC = 50000
DEFAULT_GRADIENT_DESCENT_PARAMETERS_EI_ANALYTIC = python_optimization.GradientDescentParameters(
        max_num_steps=500,
        max_num_restarts=4,
        num_steps_averaged=0,
        gamma=0.6,
        pre_mult=1.0,
        max_relative_change=1.0,
        tolerance=1.0e-7,
        )

DEFAULT_NULL_NUM_RANDOM_SAMPLES_EI_MC = 50000

DEFAULT_GRADIENT_DESCENT_MULTISTARTS_EI_MC = 200
DEFAULT_GRADIENT_DESCENT_NUM_RANDOM_SAMPLES_EI_MC = 4000
DEFAULT_GRADIENT_DESCENT_PARAMETERS_EI_MC = python_optimization.GradientDescentParameters(
        max_num_steps=500,
        max_num_restarts=4,
        num_steps_averaged=100,
        gamma=0.6,
        pre_mult=1.0,
        max_relative_change=1.0,
        tolerance=1.0e-5,
        )

DEFAULT_LBFGSB_MULTISTARTS_QEI = 200
DEFAULT_LBFGSB_NUM_RANDOM_SAMPLES_QEI = 4000
DEFAULT_LBFGSB_PARAMETERS_QEI = python_optimization.LBFGSBParameters(
        approx_grad=True,
        max_func_evals=15000,
        max_metric_correc=10,
        factr=10000000.0,
        pgtol=1.0e-5,
        epsilon=1.0e-8,
        )


# See DefaultOptimizerInfoTuple below for docstring.
_BaseDefaultOptimizerInfoTuple = namedtuple('_BaseDefaultOptimizerInfoTuple', [
    'num_multistarts',
    'num_random_samples',
    'optimizer_parameters',
])


class DefaultOptimizerInfoTuple(_BaseDefaultOptimizerInfoTuple):

    """Container holding default values to use with a :class:`moe.views.schemas.OptimizerInfo`.

    :ivar num_multistarts: (*int > 0*) number of locations from which to start optimization runs
    :ivar num_random_samples: (*int >= 0*) number of random search points to use if multistart optimization fails
    :ivar optimizer_parameters: (*namedtuple*) parameters to use with the core optimizer,
      i.e., one of :class:`moe.optimal_learning.python.python_version.optimization.GradientDescentParameters`,
      :class:`moe.optimal_learning.python.python_version.optimization.NewtonParameters`, etc.

    """

    __slots__ = ()


_EI_ANALYTIC_NULL_OPTIMIZER = DefaultOptimizerInfoTuple(
    1,  # unused but the min value is 1
    DEFAULT_NULL_NUM_RANDOM_SAMPLES_EI_ANALYTIC,
    python_optimization.NullParameters(),
)

_EI_ANALYTIC_DEFAULT_OPTIMIZER = DefaultOptimizerInfoTuple(
    DEFAULT_GRADIENT_DESCENT_MULTISTARTS_EI_ANALYTIC,
    DEFAULT_GRADIENT_DESCENT_NUM_RANDOM_SAMPLES_EI_ANALYTIC,
    DEFAULT_GRADIENT_DESCENT_PARAMETERS_EI_ANALYTIC,
)

_EI_MULTIPOINT_DEFAULT_OPTIMIZER = DefaultOptimizerInfoTuple(
    DEFAULT_LBFGSB_MULTISTARTS_QEI,
    DEFAULT_LBFGSB_NUM_RANDOM_SAMPLES_QEI,
    DEFAULT_LBFGSB_PARAMETERS_QEI,
)

_MODEL_SELECTION_NULL_OPTIMIZER = DefaultOptimizerInfoTuple(
    1,  # unused but the min value is 1
    DEFAULT_NULL_NUM_RANDOM_SAMPLES_MODEL_SELECTION,
    python_optimization.NullParameters(),
)

_MODEL_SELECTION_GRADIENT_DESCENT_OPTIMIZER = DefaultOptimizerInfoTuple(
    DEFAULT_GRADIENT_DESCENT_MULTISTARTS_MODEL_SELECTION,
    DEFAULT_GRADIENT_DESCENT_NUM_RANDOM_SAMPLES_MODEL_SELECTION,
    DEFAULT_GRADIENT_DESCENT_PARAMETERS_MODEL_SELECTION,
)

EI_COMPUTE_TYPE_ANALYTIC = 'ei_analytic'
EI_COMPUTE_TYPE_MONTE_CARLO = 'ei_monte_carlo'
SINGLE_POINT_EI = 'single_point_ei'
MULTI_POINT_EI = 'multi_point_ei'

#: dict mapping from tuples describing endpoints and objective functions to optimizer type strings;
#: i.e., one of :const:`moe.optimal_learning.python.constant.OPTIMIZER_TYPES`.
ENDPOINT_TO_DEFAULT_OPTIMIZER_TYPE = {
    views_constant.GP_NEXT_POINTS_KRIGING_ROUTE_NAME: GRADIENT_DESCENT_OPTIMIZER,
    views_constant.GP_NEXT_POINTS_CONSTANT_LIAR_ROUTE_NAME: GRADIENT_DESCENT_OPTIMIZER,
    (views_constant.GP_NEXT_POINTS_EPI_ROUTE_NAME, MULTI_POINT_EI): L_BFGS_B_OPTIMIZER,
    (views_constant.GP_NEXT_POINTS_EPI_ROUTE_NAME, SINGLE_POINT_EI): GRADIENT_DESCENT_OPTIMIZER,
    (views_constant.GP_HYPER_OPT_ROUTE_NAME, LEAVE_ONE_OUT_LOG_LIKELIHOOD): GRADIENT_DESCENT_OPTIMIZER,
    (views_constant.GP_HYPER_OPT_ROUTE_NAME, LOG_MARGINAL_LIKELIHOOD): NEWTON_OPTIMIZER,
}

#: dict mapping from tuples of optimizer type, endpoint, etc. to default optimizer parameters. The default parameter
#: structs are of type :class:`moe.optimal_learning.python.constant.DefaultOptimizerInfoTuple` and the actual default
#: parameters are defined in :mod:`moe.optimal_learning.python.constant`.
#: Note: (NEWTON_OPTIMIZER, views_constant.GP_HYPER_OPT_ROUTE_NAME, LEAVE_ONE_OUT_LOG_LIKELIHOOD)
#: does not have an entry because this combination is not yet implemented.
#: Newton is also not implemented for any of the GP_NEXT_POINTS_* endpoints.
OPTIMIZER_TYPE_AND_OBJECTIVE_TO_DEFAULT_PARAMETERS = {
    (NULL_OPTIMIZER, views_constant.GP_NEXT_POINTS_KRIGING_ROUTE_NAME): _EI_ANALYTIC_NULL_OPTIMIZER,
    (NULL_OPTIMIZER, views_constant.GP_NEXT_POINTS_CONSTANT_LIAR_ROUTE_NAME): _EI_ANALYTIC_NULL_OPTIMIZER,
    (NULL_OPTIMIZER, views_constant.GP_NEXT_POINTS_EPI_ROUTE_NAME, EI_COMPUTE_TYPE_ANALYTIC): _EI_ANALYTIC_NULL_OPTIMIZER,
    (GRADIENT_DESCENT_OPTIMIZER, views_constant.GP_NEXT_POINTS_KRIGING_ROUTE_NAME): _EI_ANALYTIC_DEFAULT_OPTIMIZER,
    (GRADIENT_DESCENT_OPTIMIZER, views_constant.GP_NEXT_POINTS_CONSTANT_LIAR_ROUTE_NAME): _EI_ANALYTIC_DEFAULT_OPTIMIZER,
    (L_BFGS_B_OPTIMIZER, views_constant.GP_NEXT_POINTS_EPI_ROUTE_NAME, EI_COMPUTE_TYPE_ANALYTIC): _EI_MULTIPOINT_DEFAULT_OPTIMIZER,
    (GRADIENT_DESCENT_OPTIMIZER, views_constant.GP_NEXT_POINTS_EPI_ROUTE_NAME, EI_COMPUTE_TYPE_ANALYTIC): _EI_ANALYTIC_DEFAULT_OPTIMIZER,
    (NULL_OPTIMIZER, views_constant.GP_NEXT_POINTS_EPI_ROUTE_NAME, EI_COMPUTE_TYPE_MONTE_CARLO): DefaultOptimizerInfoTuple(
        1,  # unused but the min value is 1
        DEFAULT_NULL_NUM_RANDOM_SAMPLES_EI_MC,
        python_optimization.NullParameters(),
    ),
    (GRADIENT_DESCENT_OPTIMIZER, views_constant.GP_NEXT_POINTS_EPI_ROUTE_NAME, EI_COMPUTE_TYPE_MONTE_CARLO): DefaultOptimizerInfoTuple(
        DEFAULT_GRADIENT_DESCENT_MULTISTARTS_EI_MC,
        DEFAULT_GRADIENT_DESCENT_NUM_RANDOM_SAMPLES_EI_MC,
        DEFAULT_GRADIENT_DESCENT_PARAMETERS_EI_MC,
    ),
    (NULL_OPTIMIZER, views_constant.GP_HYPER_OPT_ROUTE_NAME, LEAVE_ONE_OUT_LOG_LIKELIHOOD): _MODEL_SELECTION_NULL_OPTIMIZER,
    (NULL_OPTIMIZER, views_constant.GP_HYPER_OPT_ROUTE_NAME, LOG_MARGINAL_LIKELIHOOD): _MODEL_SELECTION_NULL_OPTIMIZER,
    (GRADIENT_DESCENT_OPTIMIZER, views_constant.GP_HYPER_OPT_ROUTE_NAME, LEAVE_ONE_OUT_LOG_LIKELIHOOD): _MODEL_SELECTION_GRADIENT_DESCENT_OPTIMIZER,
    (GRADIENT_DESCENT_OPTIMIZER, views_constant.GP_HYPER_OPT_ROUTE_NAME, LOG_MARGINAL_LIKELIHOOD): _MODEL_SELECTION_GRADIENT_DESCENT_OPTIMIZER,
    (NEWTON_OPTIMIZER, views_constant.GP_HYPER_OPT_ROUTE_NAME, LOG_MARGINAL_LIKELIHOOD): DefaultOptimizerInfoTuple(
        DEFAULT_NEWTON_MULTISTARTS_MODEL_SELECTION,
        DEFAULT_NEWTON_NUM_RANDOM_SAMPLES_MODEL_SELECTION,
        DEFAULT_NEWTON_PARAMETERS_MODEL_SELECTION,
    ),
}

# Constant Liar constants
CONSTANT_LIAR_MIN = 'constant_liar_min'
CONSTANT_LIAR_MAX = 'constant_liar_max'
CONSTANT_LIAR_MEAN = 'constant_liar_mean'

#: Pre-defined constant liar "lie" methods supported by :mod:`moe`
CONSTANT_LIAR_METHODS = [
        CONSTANT_LIAR_MIN,
        CONSTANT_LIAR_MAX,
        CONSTANT_LIAR_MEAN,
        ]

DEFAULT_CONSTANT_LIAR_METHOD = CONSTANT_LIAR_MAX

# TODO(GH-257): Find a better default.
DEFAULT_CONSTANT_LIAR_LIE_NOISE_VARIANCE = 1e-12

# Kriging constants
# TODO(GH-257): Find a better default.
DEFAULT_KRIGING_NOISE_VARIANCE = 1e-8
DEFAULT_KRIGING_STD_DEVIATION_COEF = 0.0
