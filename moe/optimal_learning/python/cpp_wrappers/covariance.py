# -*- coding: utf-8 -*-
"""Thin covariance-related data containers that can be passed to cpp_wrappers.* functions/classes requiring covariance data.

C++ covariance objects currently do not expose their members to Python. Additionally although C++ has several covariance
functions available, runtime-selection is not yet implemented. The containers here just track the hyperparameters of
covariance functions in a format that can be interpreted in C++ calls.

"""
import numpy

from moe.optimal_learning.python.constant import SQUARE_EXPONENTIAL_COVARIANCE_TYPE
from moe.optimal_learning.python.interfaces.covariance_interface import CovarianceInterface


class SquareExponential(CovarianceInterface):

    r"""Implement the square exponential covariance function.

    .. Note:: comments are copied in :class:`moe.optimal_learning.python.python_version.covariance.SquareExponential`.

    The function:
    ``cov(x_1, x_2) = \alpha * \exp(-1/2 * ((x_1 - x_2)^T * L * (x_1 - x_2)) )``
    where L is the diagonal matrix with i-th diagonal entry ``1/lengths[i]/lengths[i]``

    This covariance object has ``dim+1`` hyperparameters: ``\alpha, lengths_i``

    """

    covariance_type = SQUARE_EXPONENTIAL_COVARIANCE_TYPE

    def __init__(self, hyperparameters):
        r"""Construct a square exponential covariance object that can be used with cpp_wrappers.* functions/classes.

        :param hyperparameters: hyperparameters of the covariance function; index 0 is \alpha (signal variance, \sigma_f^2)
          and index 1..dim are the per-dimension length scales.
        :type hyperparameters: array-like of size dim+1

        """
        self._hyperparameters = numpy.copy(hyperparameters)

    @property
    def num_hyperparameters(self):
        """Return the number of hyperparameters of this covariance function."""
        return self._hyperparameters.size

    @staticmethod
    def make_default_hyperparameters(dim):
        """Return a default set up hyperparameters given the dimension of the space."""
        return numpy.ones(dim + 1)

    def get_hyperparameters(self):
        """Get the hyperparameters (array of float64 with shape (num_hyperparameters)) of this covariance."""
        return numpy.copy(self._hyperparameters)

    def set_hyperparameters(self, hyperparameters):
        """Set hyperparameters to the specified hyperparameters; ordering must match."""
        self._hyperparameters = numpy.copy(hyperparameters)

    hyperparameters = property(get_hyperparameters, set_hyperparameters)

    def get_json_serializable_info(self):
        """Create and return a covariance_info dictionary of this covariance object."""
        return {
                'covariance_type': self.covariance_type,
                'hyperparameters': self.hyperparameters.tolist(),
                }

    def covariance(self, point_one, point_two):
        r"""Compute the covariance function of two points, cov(``point_one``, ``point_two``).

        We do not currently expose a C++ endpoint for this call; see :mod:`moe.optimal_learning.python.interfaces.covariance_interface` for interface specification.

        """
        raise NotImplementedError("C++ wrapper currently does not support computing covariance quantities.")

    def grad_covariance(self, point_one, point_two):
        r"""Compute the gradient of self.covariance(point_one, point_two) with respect to the FIRST argument, point_one.

        We do not currently expose a C++ endpoint for this call; see :mod:`moe.optimal_learning.python.interfaces.covariance_interface` for interface specification.

        """
        raise NotImplementedError("C++ wrapper currently does not support computing covariance quantities.")

    def hyperparameter_grad_covariance(self, point_one, point_two):
        r"""Compute the gradient of self.covariance(point_one, point_two) with respect to its hyperparameters.

        We do not currently expose a C++ endpoint for this call; see :mod:`moe.optimal_learning.python.interfaces.covariance_interface` for interface specification.

        """
        raise NotImplementedError("C++ wrapper currently does not support computing covariance quantities.")

    def hyperparameter_hessian_covariance(self, point_one, point_two):
        r"""Compute the hessian of self.covariance(point_one, point_two) with respect to its hyperparameters.

        We do not currently expose a C++ endpoint for this call; see :mod:`moe.optimal_learning.python.interfaces.covariance_interface` for interface specification.

        """
        raise NotImplementedError("C++ wrapper currently does not support computing covariance quantities.")
