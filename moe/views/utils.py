# -*- coding: utf-8 -*-
"""Utilities for MOE views."""
from moe.bandit.data_containers import HistoricalData as BanditHistoricalData
from moe.bandit.data_containers import SampleArm
from moe.optimal_learning.python.constant import L_BFGS_B_OPTIMIZER
from moe.optimal_learning.python.cpp_wrappers.gaussian_process import GaussianProcess
from moe.optimal_learning.python.python_version.gaussian_process import GaussianProcess as pythonGaussianProcess
from moe.optimal_learning.python.data_containers import SamplePoint, HistoricalData
from moe.optimal_learning.python.geometry_utils import ClosedInterval
from moe.optimal_learning.python.linkers import DOMAIN_TYPES_TO_DOMAIN_LINKS, COVARIANCE_TYPES_TO_CLASSES, OPTIMIZER_TYPES_TO_OPTIMIZER_METHODS
from moe.optimal_learning.python.python_version.expected_improvement import MVNDSTParameters


def _make_domain_from_params(params, domain_info_key="domain_info", python_version=False):
    """Create and return a C++ ingestable domain from the request params.

    ``params`` has the following form::

        params = {
            'domain_info': <instance of :class:`moe.views.schemas.base_schemas.BoundedDomainInfo`>
            ...
            }

    """
    domain_info = params.get(domain_info_key)

    domain_bounds_iterable = [ClosedInterval(bound['min'], bound['max']) for bound in domain_info.get('domain_bounds', [])]

    if python_version:
        domain_class = DOMAIN_TYPES_TO_DOMAIN_LINKS[domain_info.get('domain_type')].python_domain_class
    else:
        domain_class = DOMAIN_TYPES_TO_DOMAIN_LINKS[domain_info.get('domain_type')].cpp_domain_class

    return domain_class(domain_bounds_iterable)


def _make_covariance_of_process_from_params(params, covariance_class="cpp"):
    """Create and return a C++ backed covariance_of_process from the request params as a dict.

    ``params`` has the following form::

        params = {
            'covariance_info': <instance of :class:`moe.views.schemas.base_schemas.CovarianceInfo`>,
            ...
            }

    :param params: The request params dict
    :type params: dict

    """
    covariance_info = params.get("covariance_info")
    if covariance_class == "cpp":
        covariance_class = COVARIANCE_TYPES_TO_CLASSES[covariance_info.get('covariance_type')].cpp_covariance_class
    elif covariance_class == "python":
        covariance_class = COVARIANCE_TYPES_TO_CLASSES[covariance_info.get('covariance_type')].python_covariance_class

    hyperparameters = covariance_info.get('hyperparameters')
    if hyperparameters is None:
        domain_info = params.get("domain_info")
        hyperparameters = covariance_class.make_default_hyperparameters(dim=domain_info.get('dim'))

    covariance_of_process = covariance_class(hyperparameters)
    return covariance_of_process


def _make_optimizer_parameters_from_params(params):
    """Figure out which cpp_wrappers.* objects to construct from params, validate and return them.

    :param params: the deserialized REST request, containing optimizer_info
    :type params: a dictionary with a key optimizer_info containing a :class:`moe.views.schemas.OptimizerInfo()` object with optimizer parameters

    """
    optimizer_info = params.get('optimizer_info')
    num_random_samples = optimizer_info.get('num_random_samples')
    validated_optimizer_parameters = optimizer_info.get('optimizer_parameters')

    optimizer_method = OPTIMIZER_TYPES_TO_OPTIMIZER_METHODS[optimizer_info.get('optimizer_type')]

    if optimizer_method.cpp_optimizer_class is not None:
        # TODO(GH-167): Kill this when you reoganize num_multistarts for C++.
        validated_optimizer_parameters['num_multistarts'] = optimizer_info['num_multistarts']
        optimizer_parameters = optimizer_method.cpp_parameters_class(**validated_optimizer_parameters)
        return optimizer_method.cpp_optimizer_class, optimizer_parameters, num_random_samples
    else:
        optimizer_parameters = optimizer_method.python_parameters_class(**validated_optimizer_parameters)
        return optimizer_method.python_optimizer_class, optimizer_parameters, num_random_samples


def _make_mvndst_parameters_from_params(params):
    """Construct mvndst parameters the deserialized REST request.

    :param params: the deserialized REST request, containing mvndst_parameters
    :type params: a dictionary with a key mvndst_parameters containing a :class:`moe.views.schemas.base_schemas.MVNDSTParametersSchema()`

    """
    mvndst_parameters = params.get('mvndst_parameters')
    return MVNDSTParameters(**mvndst_parameters)


def _make_gp_from_params(params):
    """Create and return a C++ backed gaussian_process from the request params as a dict.

    ``params`` has the following form::

        params = {
            'gp_historical_info': <instance of :class:`moe.views.schemas.base_schemas.GpHistoricalInfo`>,
            'domain_info': <instance of :class:`moe.views.schemas.base_schemas.DomainInfo`>,
            'covariance_info': <instance of :class:`moe.views.schemas.base_schemas.CovarianceInfo`>,
            }

    :param params: The request params dict
    :type params: dict

    """
    # Load up the info
    gp_historical_info = params.get("gp_historical_info")
    domain_info = params.get("domain_info")
    points_sampled = gp_historical_info.get('points_sampled')

    sample_point_list = []
    for point in points_sampled:
        sample_point_list.append(
            SamplePoint(
                point['point'],
                point['value'],
                point['value_var'],
            )
        )
    optimizer_info = params.get('optimizer_info', {})
    optimizer_type = optimizer_info.get('optimizer_type', None)

    if optimizer_type == L_BFGS_B_OPTIMIZER:
        covariance_of_process = _make_covariance_of_process_from_params(params, "python")
        gaussian_process = pythonGaussianProcess(
            covariance_of_process,
            HistoricalData(domain_info.get('dim'), sample_point_list),
        )
    else:
        covariance_of_process = _make_covariance_of_process_from_params(params)
        gaussian_process = GaussianProcess(
            covariance_of_process,
            HistoricalData(domain_info.get('dim'), sample_point_list),
        )

    return gaussian_process


def _make_bandit_historical_info_from_params(params, arm_type=SampleArm):
    """Create and return a bandit historical info from the request params as a dict.

    ``params`` has the following form::

        params = {
            'historical_info': <instance of :class:`moe.views.schemas.bandit_pretty_view.BanditHistoricalInfo`>,
            }

    :param params: The request params dict
    :type params: dict

    """
    arms_sampled = {}
    # Load up the info
    for arm_name, sampled_arm in params.get("historical_info").get("arms_sampled").iteritems():
        arms_sampled[arm_name] = arm_type(win=sampled_arm.get("win"), loss=sampled_arm.get("loss", 0), total=sampled_arm.get("total"), variance=sampled_arm.get("variance", None))

    bandit_historical_info = BanditHistoricalData(sample_arms=arms_sampled)

    return bandit_historical_info
