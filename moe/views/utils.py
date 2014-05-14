# -*- coding: utf-8 -*-
"""Utilities for MOE views."""
from numpy.linalg import LinAlgError

from moe.optimal_learning.python.data_containers import SamplePoint, HistoricalData
from moe.optimal_learning.python.python_version.covariance import COVARIANCE_TYPES_TO_CLASSES
from moe.optimal_learning.python.cpp_wrappers.gaussian_process import GaussianProcess
from moe.views.exceptions import SingularMatrixError


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
    covariance_class = COVARIANCE_TYPES_TO_CLASSES[covariance_info.get('covariance_type')]
    covariance_of_process = covariance_class(
            covariance_info.get('hyperparameters')
            )

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
