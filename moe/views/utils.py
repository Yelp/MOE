# -*- coding: utf-8 -*-
"""Utilities for MOE views."""
from numpy.linalg import LinAlgError

from moe.optimal_learning.python.linkers import DOMAIN_TYPES_TO_DOMAIN_LINKS, COVARIANCE_TYPES_TO_CLASSES
from moe.optimal_learning.python.data_containers import SamplePoint, HistoricalData
from moe.optimal_learning.python.cpp_wrappers.gaussian_process import GaussianProcess
from moe.optimal_learning.python.geometry_utils import ClosedInterval
from moe.views.exceptions import SingularMatrixError


def _build_domain_info(domain):
    """Create and return a domain_info dictionary from a :class:`~moe.optimal_learning.python.python_version.domain.Domain` object."""
    return {
            'domain_type': domain.domain_type,
            'dim': domain.dim,
            'domain_bounds': domain._domain_bounds,
            }


def _make_domain_from_params(params, python_version=False):
    """Create and return a C++ ingestable domain from the request params.

    ``params`` has the following form::

        params = {
            'domain_info': <instance of moe.rest.schemas.BoundedDomainInfo>,
            ...
            }

    """
    domain_info = params.get("domain_info")

    domain_bounds_iterable = [ClosedInterval(bound['min'], bound['max']) for bound in domain_info.get('domain_bounds', [])]

    if python_version:
        domain_class = DOMAIN_TYPES_TO_DOMAIN_LINKS[domain_info.get('domain_type')].python_domain_class
    else:
        domain_class = DOMAIN_TYPES_TO_DOMAIN_LINKS[domain_info.get('domain_type')].cpp_domain_class

    return domain_class(domain_bounds_iterable)


def _make_gp_from_params(params):
    """Create and return a C++ backed gaussian_process from the request params as a dict.

    ``params`` has the following form::

        params = {
            'gp_info': <instance of moe.rest.schemas.GpInfo>,
            'domain_info': <instance of moe.rest.schemas.DomainInfo>,
            'covariance_info': <instance of moe.rest.schemas.CovarianceInfo>,
            }

    :param params: The request params dict
    :type params: dict

    """
    # Load up the info
    gp_info = params.get("gp_info")
    covariance_info = params.get("covariance_info")
    domain_info = params.get("domain_info")

    points_sampled = gp_info['points_sampled']

    # Build the required objects
    covariance_class = COVARIANCE_TYPES_TO_CLASSES[covariance_info.get('covariance_type')].cpp_covariance_class

    hyperparameters = covariance_info.get('hyperparameters')
    if hyperparameters is None:
        hyperparameters = covariance_class.make_default_hyperparameters(dim=domain_info.get('dim'))

    covariance_of_process = covariance_class(hyperparameters)

    print hyperparameters

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
