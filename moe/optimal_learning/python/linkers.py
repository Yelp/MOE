# -*- coding: utf-8 -*-
"""Links between the python and cpp_wrapper implementations of domains, covariances and optimizations."""
from collections import namedtuple

from moe.optimal_learning.python.constant import SQUARE_EXPONENTIAL_COVARIANCE_TYPE, TENSOR_PRODUCT_DOMAIN_TYPE, SIMPLEX_INTERSECT_TENSOR_PRODUCT_DOMAIN_TYPE, NULL_OPTIMIZER, NEWTON_OPTIMIZER, GRADIENT_DESCENT_OPTIMIZER, L_BFGS_B_OPTIMIZER, LOG_MARGINAL_LIKELIHOOD, LEAVE_ONE_OUT_LOG_LIKELIHOOD
import moe.optimal_learning.python.cpp_wrappers.covariance as cpp_covariance
import moe.optimal_learning.python.cpp_wrappers.domain as cpp_domain
from moe.optimal_learning.python.cpp_wrappers.log_likelihood import GaussianProcessLogMarginalLikelihood, GaussianProcessLeaveOneOutLogLikelihood
import moe.optimal_learning.python.cpp_wrappers.optimization as cpp_optimization
import moe.optimal_learning.python.python_version.covariance as python_covariance
import moe.optimal_learning.python.python_version.domain as python_domain
import moe.optimal_learning.python.python_version.optimization as python_optimization


# Covariance
CovarianceLinks = namedtuple(
        'CovarianceLinks',
        [
            'python_covariance_class',
            'cpp_covariance_class',
            ],
        )

COVARIANCE_TYPES_TO_CLASSES = {
        SQUARE_EXPONENTIAL_COVARIANCE_TYPE: CovarianceLinks(
            python_covariance.SquareExponential,
            cpp_covariance.SquareExponential,
            ),
        }

# Domain
DomainLinks = namedtuple(
        'DomainLinks',
        [
            'python_domain_class',
            'cpp_domain_class',
            ],
        )

DOMAIN_TYPES_TO_DOMAIN_LINKS = {
        TENSOR_PRODUCT_DOMAIN_TYPE: DomainLinks(
            python_domain.TensorProductDomain,
            cpp_domain.TensorProductDomain,
            ),
        SIMPLEX_INTERSECT_TENSOR_PRODUCT_DOMAIN_TYPE: DomainLinks(
            None,
            cpp_domain.SimplexIntersectTensorProductDomain,
            ),
        }

# Optimization
OptimizerMethod = namedtuple(
        'OptimizerMethod',
        [
            'optimizer_type',
            'python_parameters_class',
            'cpp_parameters_class',
            'python_optimizer_class',
            'cpp_optimizer_class',
            ],
        )

OPTIMIZER_TYPES_TO_OPTIMIZER_METHODS = {
        NULL_OPTIMIZER: OptimizerMethod(
            optimizer_type=NULL_OPTIMIZER,
            python_parameters_class=python_optimization.NullParameters,
            cpp_parameters_class=cpp_optimization.NullParameters,
            python_optimizer_class=python_optimization.NullOptimizer,
            cpp_optimizer_class=cpp_optimization.NullOptimizer,
            ),
        NEWTON_OPTIMIZER: OptimizerMethod(
            optimizer_type=NEWTON_OPTIMIZER,
            python_parameters_class=python_optimization.NewtonParameters,
            cpp_parameters_class=cpp_optimization.NewtonParameters,
            python_optimizer_class=None,
            cpp_optimizer_class=cpp_optimization.NewtonOptimizer,
            ),
        GRADIENT_DESCENT_OPTIMIZER: OptimizerMethod(
            optimizer_type=GRADIENT_DESCENT_OPTIMIZER,
            python_parameters_class=python_optimization.GradientDescentParameters,
            cpp_parameters_class=cpp_optimization.GradientDescentParameters,
            python_optimizer_class=python_optimization.GradientDescentOptimizer,
            cpp_optimizer_class=cpp_optimization.GradientDescentOptimizer,
            ),
        L_BFGS_B_OPTIMIZER: OptimizerMethod(
            optimizer_type=L_BFGS_B_OPTIMIZER,
            python_parameters_class=python_optimization.LBFGSBParameters,
            cpp_parameters_class=None,
            python_optimizer_class=python_optimization.LBFGSBOptimizer,
            cpp_optimizer_class=None,
            ),
        }

# Log Likelihood
LogLikelihoodMethod = namedtuple(
        'LogLikelihoodMethod',
        [
            'log_likelihood_type',
            'log_likelihood_class',
            ]
        )

LOG_LIKELIHOOD_TYPES_TO_LOG_LIKELIHOOD_METHODS = {
        LOG_MARGINAL_LIKELIHOOD: LogLikelihoodMethod(
            log_likelihood_type=LOG_MARGINAL_LIKELIHOOD,
            log_likelihood_class=GaussianProcessLogMarginalLikelihood,
            ),
        LEAVE_ONE_OUT_LOG_LIKELIHOOD: LogLikelihoodMethod(
            log_likelihood_type=LEAVE_ONE_OUT_LOG_LIKELIHOOD,
            log_likelihood_class=GaussianProcessLeaveOneOutLogLikelihood,
            ),
        }
