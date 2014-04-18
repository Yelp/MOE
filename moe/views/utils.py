"""Utilities for MOE views."""
from numpy.linalg import LinAlgError

from moe.optimal_learning.EPI.src.python.models.optimal_gaussian_process_linked_cpp import OptimalGaussianProcessLinkedCpp
from moe.optimal_learning.EPI.src.python.models.covariance_of_process import CovarianceOfProcess
from moe.optimal_learning.EPI.src.python.models.sample_point import SamplePoint
from moe.views.exceptions import SingularMatrixError


def _make_default_covariance_of_process(signal_variance=None, length=None):
    """Make a default covariance of process with optional parameters."""
    hyperparameters = [signal_variance]
    hyperparameters.extend(length)

    return CovarianceOfProcess(hyperparameters=hyperparameters)


def _make_gp_from_gp_info(gp_info):
    """Create and return a C++ backed GP from a gp_info dict.

    gp_info has the form of GpInfo in moe/schemas.py

    """
    # Load up the info
    points_sampled = gp_info['points_sampled']
    domain = gp_info['domain']
    signal_variance = gp_info['signal_variance']
    length = gp_info['length_scale']

    # Build the required objects
    covariance_of_process = _make_default_covariance_of_process(
            signal_variance=signal_variance,
            length=length,
            )
    GP = OptimalGaussianProcessLinkedCpp(
            domain=domain,
            covariance_of_process=covariance_of_process,
            )

    # Sample from the process
    for point in points_sampled:
        sample_point = SamplePoint(
                point['point'],
                point['value'],
                point['value_var'],
                )
        try:
            GP.add_sample_point(sample_point, point['value_var'])
        except LinAlgError:
            raise(SingularMatrixError)

    return GP
