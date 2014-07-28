# -*- coding: utf-8 -*-
"""Implementation of GaussianProcessInterface using C++ calls.

This file contains a class to manipulate a Gaussian Process through the C++ implementation (gpp_math.hpp/cpp).

See :mod:`moe.optimal_learning.python.interfaces.gaussian_process_interface` for more details.

"""
import copy

import numpy

import moe.build.GPP as C_GP
import moe.optimal_learning.python.cpp_wrappers.cpp_utils as cpp_utils
from moe.optimal_learning.python.interfaces.gaussian_process_interface import GaussianProcessInterface


class GaussianProcess(GaussianProcessInterface):

    r"""Implementation of a GaussianProcess via C++ wrappers: mean, variance, gradients thereof, and data I/O.

    .. Note:: Comments in this class are copied from
      :mod:`moe.optimal_learning.python.interfaces.gaussian_process_interface.GaussianProcessInterface`

    Object that encapsulates Gaussian Process Priors (GPPs).  A GPP is defined by a set of
    (sample point, function value, noise variance) triples along with a covariance function that relates the points.
    Each point has dimension dim.  These are the training data; for example, each sample point might specify an experimental
    cohort and the corresponding function value is the objective measured for that experiment.  There is one noise variance
    value per function value; this is the measurement error and is treated as N(0, noise_variance) Gaussian noise.

    GPPs estimate a real process \ms f(x) = GP(m(x), k(x,x'))\me (see file docs).  This class deals with building an estimator
    to the actual process using measurements taken from the actual process--the (sample point, function val, noise) triple.
    Then predictions about unknown points can be made by sampling from the GPP--in particular, finding the (predicted)
    mean and variance.  These functions (and their gradients) are provided in ComputeMeanOfPoints, ComputeVarianceOfPoints,
    etc.

    Further mathematical details are given in the implementation comments, but we are essentially computing:

    | ComputeMeanOfPoints    : ``K(Xs, X) * [K(X,X) + \sigma_n^2 I]^{-1} * y = Ks^T * K^{-1} * y``
    | ComputeVarianceOfPoints: ``K(Xs, Xs) - K(Xs,X) * [K(X,X) + \sigma_n^2 I]^{-1} * K(X,Xs) = Kss - Ks^T * K^{-1} * Ks``

    This (estimated) mean and variance characterize the predicted distributions of the actual \ms m(x), k(x,x')\me
    functions that underly our GP.

    The "independent variables" for this object are ``points_to_sample``. These points are both the "p" and the "q" in q,p-EI;
    i.e., they are the parameters of both ongoing experiments and new predictions. Recall that in q,p-EI, the q points are
    called ``points_to_sample`` and the p points are called ``points_being_sampled.`` Here, we need to make predictions about
    both point sets with the GP, so we simply call the union of point sets ``points_to_sample.``

    In GP computations, there is really no distinction between the "q" and "p" points from EI, ``points_to_sample`` and
    ``points_being_sampled``, respectively. However, in EI optimization, we only need gradients of GP quantities wrt
    ``points_to_sample``, so users should call members functions with ``num_derivatives = num_to_sample`` in that context.

    """

    def __init__(self, covariance_function, historical_data):
        """Construct a GaussianProcess object that knows how to call C++ for evaluation of member functions.

        :param covariance_function: covariance object encoding assumptions about the GP's behavior on our data
        :type covariance_function: :class:`moe.optimal_learning.python.interfaces.covariance_interface.CovarianceInterface` subclass
          (e.g., from :mod:`moe.optimal_learning.python.cpp_wrappers.covariance`).
        :param historical_data: object specifying the already-sampled points, the objective value at those points, and the noise variance associated with each observation
        :type historical_data: :class:`moe.optimal_learning.python.data_containers.HistoricalData` object

        """
        self._covariance = copy.deepcopy(covariance_function)

        self._historical_data = copy.deepcopy(historical_data)

        # C++ will maintain its own copy of the contents of hyperparameters and historical_data
        self._gaussian_process = C_GP.GaussianProcess(
            cpp_utils.cppify_hyperparameters(self._covariance.hyperparameters),
            cpp_utils.cppify(historical_data.points_sampled),
            cpp_utils.cppify(historical_data.points_sampled_value),
            cpp_utils.cppify(historical_data.points_sampled_noise_variance),
            self._historical_data.dim,
            self._historical_data.num_sampled,
        )

    @property
    def dim(self):
        """Return the number of spatial dimensions."""
        return self._gaussian_process.dim

    @property
    def num_sampled(self):
        """Return the number of sampled points."""
        return self._gaussian_process.num_sampled

    def get_covariance_copy(self):
        """Return a copy of the covariance object specifying the Gaussian Process.

        :return: covariance object encoding assumptions about the GP's behavior on our data
        :rtype: interfaces.covariance_interface.CovarianceInterface subclass

        """
        return copy.deepcopy(self._covariance)

    @property
    def _points_sampled_value(self):
        """Return the function values measured at each of points_sampled; see :class:`moe.optimal_learning.python.data_containers.HistoricalData`."""
        return self._historical_data.points_sampled_value

    def get_historical_data_copy(self):
        """Return the data (points, function values, noise) specifying the prior of the Gaussian Process.

        :return: object specifying the already-sampled points, the objective value at those points, and the noise variance associated with each observation
        :rtype: data_containers.HistoricalData

        """
        return copy.deepcopy(self._historical_data)

    def compute_mean_of_points(self, points_to_sample):
        r"""Compute the mean of this GP at each of point of ``Xs`` (``points_to_sample``).

        ``points_to_sample`` may not contain duplicate points. Violating this results in singular covariance matrices.

        .. Note:: Comments are copied from
          :mod:`moe.optimal_learning.python.interfaces.gaussian_process_interface.GaussianProcessInterface.compute_mean_of_points`

        :param points_to_sample: num_to_sample points (in dim dimensions) being sampled from the GP
        :type points_to_sample: array of float64 with shape (num_to_sample, dim)
        :return: mean: where mean[i] is the mean at points_to_sample[i]
        :rtype: array of float64 with shape (num_to_sample)

        """
        mu = self._gaussian_process.compute_mean_of_points(
            cpp_utils.cppify(points_to_sample),
            points_to_sample.shape[0],
        )
        return numpy.array(mu)

    def compute_grad_mean_of_points(self, points_to_sample, num_derivatives=-1):
        r"""Compute the gradient of the mean of this GP at each of point of ``Xs`` (``points_to_sample``) wrt ``Xs``.

        ``points_to_sample`` may not contain duplicate points. Violating this results in singular covariance matrices.

        Note that ``grad_mu`` is nominally sized: ``grad_mu[num_to_sample][num_to_sample][dim]``. This is
        the the d-th component of the derivative evaluated at the i-th input wrt the j-th input.
        However, for ``0 <= i,j < num_to_sample``, ``i != j``, ``grad_mu[j][i][d] = 0``.
        (See references or implementation for further details.)
        Thus, ``grad_mu`` is stored in a reduced form which only tracks the nonzero entries.

        .. Note:: Comments are copied from
          :mod:`moe.optimal_learning.python.interfaces.gaussian_process_interface.GaussianProcessInterface.compute_grad_mean_of_points`

        :param points_to_sample: num_to_sample points (in dim dimensions) being sampled from the GP
        :type points_to_sample: array of float64 with shape (num_to_sample, dim)
        :param num_derivatives: return derivatives wrt points_to_sample[0:num_derivatives]; large or negative values are clamped
        :type num_derivatives: int
        :return: grad_mu: gradient of the mean of the GP. ``grad_mu[i][d]`` is actually the gradient
          of ``\mu_i`` wrt ``x_{i,d}``, the d-th dim of the i-th entry of ``points_to_sample``.
        :rtype: array of float64 with shape (num_to_sample, dim)

        """
        num_derivatives = self._clamp_num_derivatives(points_to_sample.shape[0], num_derivatives)
        grad_mu = self._gaussian_process.compute_grad_mean_of_points(
            cpp_utils.cppify(points_to_sample[:num_derivatives, ...]),
            num_derivatives,
        )
        return cpp_utils.uncppify(grad_mu, (num_derivatives, self.dim))

    def compute_variance_of_points(self, points_to_sample):
        r"""Compute the variance (matrix) of this GP at each point of ``Xs`` (``points_to_sample``).

        ``points_to_sample`` may not contain duplicate points. Violating this results in singular covariance matrices.

        The variance matrix is symmetric although we currently return the full representation.

        .. Note:: Comments are copied from
          :mod:`moe.optimal_learning.python.interfaces.gaussian_process_interface.GaussianProcessInterface.compute_variance_of_points`

        :param points_to_sample: num_to_sample points (in dim dimensions) being sampled from the GP
        :type points_to_sample: array of float64 with shape (num_to_sample, dim)
        :return: var_star: variance matrix of this GP
        :rtype: array of float64 with shape (num_to_sample, num_to_sample)

        """
        num_to_sample = points_to_sample.shape[0]
        variance = self._gaussian_process.compute_variance_of_points(
            cpp_utils.cppify(points_to_sample),
            num_to_sample,
        )
        return cpp_utils.uncppify(variance, (num_to_sample, num_to_sample))

    def compute_cholesky_variance_of_points(self, points_to_sample):
        r"""Compute the cholesky factorization of the variance (matrix) of this GP at each point of ``Xs`` (``points_to_sample``).

        ``points_to_sample`` may not contain duplicate points. Violating this results in singular covariance matrices.

        :param points_to_sample: num_to_sample points (in dim dimensions) being sampled from the GP
        :type points_to_sample: array of float64 with shape (num_to_sample, dim)
        :return: cholesky factorization of the variance matrix of this GP, lower triangular
        :rtype: array of float64 with shape (num_to_sample, num_to_sample), only lower triangle filled in

        """
        num_to_sample = points_to_sample.shape[0]
        cholesky_variance = self._gaussian_process.compute_cholesky_variance_of_points(
            cpp_utils.cppify(points_to_sample),
            num_to_sample,
        )
        return cpp_utils.uncppify(cholesky_variance, (num_to_sample, num_to_sample))

    def compute_grad_variance_of_points(self, points_to_sample, num_derivatives=-1):
        r"""Compute the gradient of the variance (matrix) of this GP at each point of ``Xs`` (``points_to_sample``) wrt ``Xs``.

        ``points_to_sample`` may not contain duplicate points. Violating this results in singular covariance matrices.

        This function is similar to compute_grad_cholesky_variance_of_points() (below), except this does not include
        gradient terms from the cholesky factorization. Description will not be duplicated here.

        .. Note:: Comments are copied from
          :mod:`moe.optimal_learning.python.interfaces.gaussian_process_interface.GaussianProcessInterface.compute_grad_variance_of_points`

        :param points_to_sample: num_to_sample points (in dim dimensions) being sampled from the GP
        :type points_to_sample: array of float64 with shape (num_to_sample, dim)
        :param num_derivatives: return derivatives wrt points_to_sample[0:num_derivatives]; large or negative values are clamped
        :type num_derivatives: int
        :return: grad_var: gradient of the variance matrix of this GP
        :rtype: array of float64 with shape (num_derivatives, num_to_sample, num_to_sample, dim)

        """
        num_derivatives = self._clamp_num_derivatives(points_to_sample.shape[0], num_derivatives)
        num_to_sample = points_to_sample.shape[0]

        grad_variance = self._gaussian_process.compute_grad_variance_of_points(
            cpp_utils.cppify(points_to_sample),
            num_to_sample,
            num_derivatives,
        )
        return cpp_utils.uncppify(grad_variance, (num_derivatives, num_to_sample, num_to_sample, self.dim))

    def compute_grad_cholesky_variance_of_points(self, points_to_sample, num_derivatives=-1):
        r"""Compute the gradient of the cholesky factorization of the variance (matrix) of this GP at each point of ``Xs`` (``points_to_sample``) wrt ``Xs``.

        ``points_to_sample`` may not contain duplicate points. Violating this results in singular covariance matrices.

        This function accounts for the effect on the gradient resulting from
        cholesky-factoring the variance matrix.  See Smith 1995 for algorithm details.

        Note that ``grad_chol`` is nominally sized:
        ``grad_chol[num_to_sample][num_to_sample][num_to_sample][dim]``.
        Let this be indexed ``grad_chol[k][j][i][d]``, which is read the derivative of ``var[j][i]``
        with respect to ``x_{k,d}`` (x = ``points_to_sample``)

        .. Note:: Comments are copied from
          :mod:`moe.optimal_learning.python.interfaces.gaussian_process_interface.GaussianProcessInterface.compute_grad_cholesky_variance_of_points`

        :param points_to_sample: num_to_sample points (in dim dimensions) being sampled from the GP
        :type points_to_sample: array of float64 with shape (num_to_sample, dim)
        :param num_derivatives: return derivatives wrt points_to_sample[0:num_derivatives]; large or negative values are clamped
        :type num_derivatives: int
        :return: grad_chol: gradient of the cholesky factorization of the variance matrix of this GP.
          ``grad_chol[k][j][i][d]`` is actually the gradients of ``var_{j,i}`` with
          respect to ``x_{k,d}``, the d-th dimension of the k-th entry of ``points_to_sample``
        :rtype: array of float64 with shape (num_derivatives, num_to_sample, num_to_sample, dim)

        """
        num_derivatives = self._clamp_num_derivatives(points_to_sample.shape[0], num_derivatives)
        num_to_sample = points_to_sample.shape[0]

        grad_chol_decomp = self._gaussian_process.compute_grad_cholesky_variance_of_points(
            cpp_utils.cppify(points_to_sample),
            num_to_sample,
            num_derivatives,
        )
        return cpp_utils.uncppify(grad_chol_decomp, (num_derivatives, num_to_sample, num_to_sample, self.dim))

    def add_sampled_points(self, sampled_points):
        r"""Add sampled point(s) (point, value, noise) to the GP's prior data.

        Also forces recomputation of all derived quantities for GP to remain consistent.

        :param sampled_points: :class:`moe.optimal_learning.python.SamplePoint` objects to load
          into the GP (containing point, function value, and noise variance)
        :type sampled_points: list of :class:`~moe.optimal_learning.python.SamplePoint` objects (or SamplePoint-like iterables)

        """
        # TODO(GH-159): When C++ can pass back numpy arrays, we can stop keeping a duplicate in self._historical_data.
        num_sampled_prev = self.num_sampled
        num_to_add = len(sampled_points)
        self._historical_data.append_sample_points(sampled_points)

        # new_historical_data = HistoricalData(self.dim, sampled_points)
        self._gaussian_process.add_sampled_points(
            cpp_utils.cppify(self._historical_data.points_sampled[num_sampled_prev:, ...]),
            cpp_utils.cppify(self._historical_data.points_sampled_value[num_sampled_prev:]),
            cpp_utils.cppify(self._historical_data.points_sampled_noise_variance[num_sampled_prev:]),
            num_to_add,
        )

    def sample_point_from_gp(self, point_to_sample, noise_variance=0.0):
        r"""Sample a function value from a Gaussian Process prior, provided a point at which to sample.

        Uses the formula ``function_value = gpp_mean + sqrt(gpp_variance) * w1 + sqrt(noise_variance) * w2``, where ``w1, w2``
        are draws from N(0,1).

        Normal RNG source is held within the C++ GaussianProcess object.

        .. NOTE::
             Set noise_variance to 0 if you want "accurate" draws from the GP.
             BUT if the drawn (point, value) pair is meant to be added back into the GP (e.g., for testing), then this point
             MUST be drawn with noise_variance equal to the noise associated with "point" as a member of "points_sampled"

        .. Note:: Comments are copied from
          :mod:`moe.optimal_learning.python.interfaces.gaussian_process_interface.GaussianProcessInterface.sample_point_from_gp`

        :param point_to_sample: point (in dim dimensions) at which to sample from this GP
        :type points_to_sample: array of float64 with shape (dim)
        :param noise_variance: amount of noise to associate with the sample
        :type noise_variance: float64 >= 0.0
        :return: sample_value: function value drawn from this GP
        :rtype: float64

        """
        return self._gaussian_process.sample_point_from_gp(
            cpp_utils.cppify(point_to_sample),
            noise_variance,
        )
