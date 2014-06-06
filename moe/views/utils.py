# -*- coding: utf-8 -*-
"""Utilities for MOE views."""
from numpy.linalg import LinAlgError

from moe.optimal_learning.python.cpp_wrappers.gaussian_process import GaussianProcess
from moe.optimal_learning.python.data_containers import SamplePoint, HistoricalData
from moe.optimal_learning.python.geometry_utils import ClosedInterval
from moe.optimal_learning.python.linkers import DOMAIN_TYPES_TO_DOMAIN_LINKS, COVARIANCE_TYPES_TO_CLASSES, OPTIMIZATION_TYPES_TO_OPTIMIZATION_METHODS
from moe.views.exceptions import SingularMatrixError


def _build_domain_info(domain):
    """Create and return a domain_info dictionary from a :class:`~moe.optimal_learning.python.python_version.domain.Domain` object."""
    return {
            'domain_type': domain.domain_type,
            'dim': domain.dim,
            'domain_bounds': domain._domain_bounds,
            }


def _make_domain_from_params(params, domain_info_key="domain_info", python_version=False):
    """Create and return a C++ ingestable domain from the request params.

    ``params`` has the following form::

        params = {
            'domain_info': <instance of moe.rest.schemas.BoundedDomainInfo>,
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


def _build_covariance_info(covariance):
    """Create and return a covariance_info dictionary from a :class:`~moe.optimal_learning.python.python_version.covaraince.Covaraince` object."""
    return {
            'covariance_type': covariance.covariance_type,
            'hyperparameters': covariance.get_hyperparameters().tolist(),
            }


def _make_covariance_of_process_from_params(params):
    """Create and return a C++ backed covariance_of_process from the request params as a dict.

    ``params`` has the following form::

        params = {
            'covariance_info': <instance of moe.rest.schemas.CovarianceInfo>,
            ...
            }

    :param params: The request params dict
    :type params: dict

    """
    covariance_info = params.get("covariance_info")
    covariance_class = COVARIANCE_TYPES_TO_CLASSES[covariance_info.get('covariance_type')].cpp_covariance_class

    hyperparameters = covariance_info.get('hyperparameters')
    if hyperparameters is None:
        domain_info = params.get("domain_info")
        hyperparameters = covariance_class.make_default_hyperparameters(dim=domain_info.get('dim'))

    covariance_of_process = covariance_class(hyperparameters)
    return covariance_of_process


def _make_optimization_parameters_from_params(params):
    """Figure out which cpp_wrappers.* objects to construct from params, validate and return them.

    :param params: the deserialized REST request, containing ei_optimization_parameters
    :type params: a dictionary with a key ei_optimization_parameters containing a :class:`moe.views.schemas.EiOptimizationParameters()` object with optimization parameters

    """
    optimization_info = params.get('optimization_info')
    num_random_samples = optimization_info.get('num_random_samples')
    validated_optimization_parameters = params.get('optimization_info').get('optimization_parameters')

    optimization_method = OPTIMIZATION_TYPES_TO_OPTIMIZATION_METHODS[optimization_info.get('optimization_type')]

    # TODO(eliu): Kill this when you reoganize num_multistarts for C++
    validated_optimization_parameters['num_multistarts'] = optimization_info['num_multistarts']
    optimization_parameters = optimization_method.cpp_parameters_class(**validated_optimization_parameters)

    return optimization_method.cpp_optimizer_class, optimization_parameters, num_random_samples


def _make_gp_from_params(params):
    """Create and return a C++ backed gaussian_process from the request params as a dict.

    ``params`` has the following form::

        params = {
            'gp_historical_info': <instance of moe.rest.schemas.GpHistoricalInfo>,
            'domain_info': <instance of moe.rest.schemas.DomainInfo>,
            'covariance_info': <instance of moe.rest.schemas.CovarianceInfo>,
            }

    :param params: The request params dict
    :type params: dict

    """
    # Load up the info
    gp_historical_info = params.get("gp_historical_info")
    domain_info = params.get("domain_info")
    points_sampled = gp_historical_info.get('points_sampled')

    covariance_of_process = _make_covariance_of_process_from_params(params)
    gaussian_process = GaussianProcess(
            covariance_of_process,
            HistoricalData(domain_info.get('dim')),
            )

    # Sample from the process
    for point in points_sampled:
        sample_point = SamplePoint(
                point['point'],
                point['value'],
                point['value_var'],
                )
        try:
            gaussian_process.add_sampled_points([sample_point])
        except LinAlgError:
            raise(SingularMatrixError)

    return gaussian_process
