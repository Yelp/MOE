# -*- coding: utf-8 -*-
"""Implementations of covariance functions for use with :mod:`moe.optimal_learning.python.python_version.log_likelihood` and :mod:`moe.optimal_learning.python.python_version.gaussian_process`.

This file contains implementations of CovarianceInterface. Currently, we have
SquareExponential, supporting:

* covariance
* grad_covariance
* hyperparameter_grad_covariance

It also contains a few utilities for computing common mathematical quantities and
initialization. Note that the hessian is not yet implemented (use C++ for that feature).

Gradient (spatial and hyperparameter) functions return all derivatives at once
because there is substantial shared computation. The shared results are by far the
most expensive part of gradient computations; they typically involve exponentiation
and are further at least partially shared with the base covariance computation. In
fact, we could improve performance further by caching [certain] components of the
covariance computation for use with the derivative computations.

"""
import numpy

from moe.optimal_learning.python.constant import SQUARE_EXPONENTIAL_COVARIANCE_TYPE
from moe.optimal_learning.python.interfaces.covariance_interface import CovarianceInterface


class SquareExponential(CovarianceInterface):

    r"""Implement the square exponential covariance function.

    .. Note:: comments are copied from :class:`moe.optimal_learning.python.cpp_wrappers.covariance.SquareExponential`.

    The function:
    ``cov(x_1, x_2) = \alpha * \exp(-1/2 * ((x_1 - x_2)^T * L * (x_1 - x_2)) )``
    where L is the diagonal matrix with i-th diagonal entry ``1/lengths[i]/lengths[i]``

    This covariance object has ``dim+1`` hyperparameters: ``\alpha, lengths_i``

    """

    covariance_type = SQUARE_EXPONENTIAL_COVARIANCE_TYPE

    def __init__(self, hyperparameters):
        r"""Construct a square exponential covariance object with the specified hyperparameters.

        :param hyperparameters: hyperparameters of the covariance function; index 0 is \alpha (signal variance, \sigma_f^2)
          and index 1..dim are the per-dimension length scales.
        :type hyperparameters: array-like of size dim+1

        """
        self.hyperparameters = hyperparameters

    @property
    def num_hyperparameters(self):
        """Return the number of hyperparameters of this covariance function."""
        return self._hyperparameters.size

    def get_hyperparameters(self):
        """Get the hyperparameters (array of float64 with shape (num_hyperparameters)) of this covariance."""
        return numpy.copy(self._hyperparameters)

    def set_hyperparameters(self, hyperparameters):
        """Set hyperparameters to the specified hyperparameters; ordering must match."""
        self._hyperparameters = numpy.copy(hyperparameters)
        self._lengths_sq = numpy.copy(self._hyperparameters[1:])
        self._lengths_sq *= self._lengths_sq

    hyperparameters = property(get_hyperparameters, set_hyperparameters)

    def get_json_serializable_info(self):
        """Create and return a covariance_info dictionary of this covariance object."""
        return {
                'covariance_type': self.covariance_type,
                'hyperparameters': self.hyperparameters.tolist(),
                }

    def covariance(self, point_one, point_two):
        r"""Compute the square exponential covariance function of two points, cov(``point_one``, ``point_two``).

        Square Exponential: ``cov(x_1, x_2) = \alpha * \exp(-1/2 * ((x_1 - x_2)^T * L * (x_1 - x_2)) )``

        .. Note:: comments are copied from the matching method comments of
          :class:`moe.optimal_learning.python.interfaces.covariance_interface.CovarianceInterface`.

        The covariance function is guaranteed to be symmetric by definition: ``covariance(x, y) = covariance(y, x)``.
        This function is also positive definite by definition.

        :param point_one: first input, the point ``x``
        :type point_one: array of float64 with shape (dim)
        :param point_two: second input, the point ``y``
        :type point_two: array of float64 with shape (dim)
        :return: value of covariance between the input points
        :rtype: float64

        """
        temp = point_two - point_one
        temp *= temp
        temp /= self._lengths_sq
        return self._hyperparameters[0] * numpy.exp(-0.5 * temp.sum())

    def grad_covariance(self, point_one, point_two):
        r"""Compute the gradient of self.covariance(point_one, point_two) with respect to the FIRST argument, point_one.

        Gradient of Square Exponential (wrt ``x_1``):
        ``\pderiv{cov(x_1, x_2)}{x_{1,i}} = (x_{2,i} - x_{1,i}) / L_{i}^2 * cov(x_1, x_2)``

        .. Note:: comments are copied from the matching method comments of
          :class:`moe.optimal_learning.python.interfaces.covariance_interface.CovarianceInterface`.

        This distinction is important for maintaining the desired symmetry.  ``Cov(x, y) = Cov(y, x)``.
        Additionally, ``\pderiv{Cov(x, y)}{x} = \pderiv{Cov(y, x)}{x}``.
        However, in general, ``\pderiv{Cov(x, y)}{x} != \pderiv{Cov(y, x)}{y}`` (NOT equal!  These may differ by a negative sign)

        Hence to avoid separate implementations for differentiating against first vs second argument, this function only handles
        differentiation against the first argument.  If you need ``\pderiv{Cov(y, x)}{x}``, just swap points x and y.

        :param point_one: first input, the point ``x``
        :type point_one: array of float64 with shape (dim)
        :param point_two: second input, the point ``y``
        :type point_two: array of float64 with shape (dim)
        :return: grad_cov: i-th entry is ``\pderiv{cov(x_1, x_2)}{x_i}``
        :rtype: array of float64 with shape (dim)

        """
        grad_cov = point_two - point_one
        grad_cov /= self._lengths_sq
        grad_cov *= self.covariance(point_one, point_two)
        return grad_cov

    def hyperparameter_grad_covariance(self, point_one, point_two):
        r"""Compute the gradient of self.covariance(point_one, point_two) with respect to its hyperparameters.

        Gradient of Square Exponential (wrt hyperparameters (``alpha, L``)):
        ``\pderiv{cov(x_1, x_2)}{\theta_0} = cov(x_1, x_2) / \theta_0``
        ``\pderiv{cov(x_1, x_2)}{\theta_0} = [(x_{1,i} - x_{2,i}) / L_i]^2 / L_i * cov(x_1, x_2)``
        Note: ``\theta_0 = \alpha`` and ``\theta_{1:d} = L_{0:d-1}``

        .. Note:: comments are copied from the matching method comments of
          :class:`moe.optimal_learning.python.interfaces.covariance_interface.CovarianceInterface`.

        Unlike GradCovariance(), the order of point_one and point_two is irrelevant here (since we are not differentiating against
        either of them).  Thus the matrix of grad covariances (wrt hyperparameters) is symmetric.

        :param point_one: first input, the point ``x``
        :type point_one: array of float64 with shape (dim)
        :param point_two: second input, the point ``y``
        :type point_two: array of float64 with shape (dim)
        :return: grad_hyperparameter_cov: i-th entry is ``\pderiv{cov(x_1, x_2)}{\theta_i}``
        :rtype: array of float64 with shape (num_hyperparameters)

        """
        cov = self.covariance(point_one, point_two)
        grad_cov = numpy.empty(self.num_hyperparameters)
        grad_cov[0] = cov / self._hyperparameters[0]
        lengths = self._hyperparameters[1:]
        grad_cov_lengths = grad_cov[1:]
        numpy.subtract(point_two, point_one, out=grad_cov_lengths)
        grad_cov_lengths /= lengths
        grad_cov_lengths *= grad_cov_lengths
        grad_cov_lengths /= lengths
        grad_cov_lengths *= cov
        return grad_cov

    def hyperparameter_hessian_covariance(self, point_one, point_two):
        r"""Compute the hessian of self.covariance(point_one, point_two) with respect to its hyperparameters.

        TODO(GH-57): Implement Hessians in Python.

        """
        raise NotImplementedError("Python implementation does not support computing the hessian covariance wrt hyperparameters.")
