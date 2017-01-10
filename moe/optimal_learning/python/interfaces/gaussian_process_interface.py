# -*- coding: utf-8 -*-
"""Interface for a GaussianProcess: mean, variance, gradients thereof, and data I/O.

This file contains two classes,
:class:`moe.optimal_learning.python.interfaces.gaussian_process_interface.GaussianProcessDataInterface` and
:class:`moe.optimal_learning.python.interfaces.gaussian_process_interface.GaussianProcessInterface`.
They specifies the interface that a GaussianProcess
implementation must satisfy in order to be used in computation/optimization of ExpectedImprovement, etc.
Python currently does not natively support interfaces, so we are commandeering ABCs for that purpose.

See package docs in :mod:`moe.optimal_learning.python.interfaces` for an introduction to Gaussian Processes.

"""
from builtins import object
from abc import ABCMeta, abstractmethod, abstractproperty
from future.utils import with_metaclass


class GaussianProcessDataInterface(with_metaclass(ABCMeta, object)):

    """Core data interface for constructing or manipulating a Gaussian Process Prior (GPP).

    This interface exists as a convenience to safely access the fundamental components of a GPP.

    Assumes the underlying GP has mean zero.

    Includes functions to return *copies* of the covariance function (see CovarianceInterface)
    and observed, historical data (coordinates, function values, noise variance; see
    :class:`moe.optimal_learning.python.data_containers.HistoricalData`) of a GP object or
    an object supporting computations on GPs.

    With the zero mean assumption, a "Gaussian Process" is fully determined by its covariance
    function (Rasmussen & Williams, Chp 2.2). Then the "Prior" is fully determined by our
    past observations. Together, the covariance and historical data produce the heart
    of a Gaussian Process Prior.

    """

    @abstractmethod
    def get_covariance_copy(self):
        """Return a copy of the covariance object specifying the Gaussian Process.

        :return: covariance object encoding assumptions about the GP's behavior on our data
        :rtype: :class:`moe.optimal_learning.python.interfaces.covariance_interface.CovarianceInterface` subclass

        """
        pass

    @abstractmethod
    def get_historical_data_copy(self):
        """Return the data (points, function values, noise) specifying the prior of the Gaussian Process.

        :return: object specifying the already-sampled points, the objective value at those points, and the noise variance associated with each observation
        :rtype: :class:`moe.optimal_learning.python.data_containers.HistoricalData`

        """
        pass

    def get_core_data_copy(self):
        """Tuple of covariance, historical_data copies for convenience; see specific getters."""
        return self.get_covariance_copy(), self.get_historical_data_copy()


class GaussianProcessInterface(with_metaclass(ABCMeta, GaussianProcessDataInterface)):

    r"""Interface for a GaussianProcess: mean, variance, gradients thereof, and data I/O.

    .. Note:: comments in this class are copied from GaussianProcess in gpp_math.hpp and duplicated in cpp_wrappers.gaussian_process
      and duplicated in
      :class:`moe.optimal_learning.python.cpp_wrappers.gaussian_process.GaussianProcess` and
      :class:`moe.optimal_learning.python.python_version.gaussian_process.GaussianProcess`

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

    @staticmethod
    def _clamp_num_derivatives(num_points, num_derivatives):
        """Clamp num_derivatives so that the result is 0 <= result <= num_points; negative num_derivatives yields num_points.

        :param num_points: number of total points
        :type num_points: int > 0
        :param num_derivatives: number of points to differentiate against
        :type num_derivatives: int

        """
        if num_derivatives < 0 or num_derivatives > num_points:
            return num_points
        else:
            return num_derivatives

    @abstractproperty
    def dim(self):
        """Return the number of spatial dimensions."""
        pass

    @abstractproperty
    def num_sampled(self):
        """Return the number of sampled points."""
        pass

    @abstractmethod
    def compute_mean_of_points(self, points_to_sample):
        r"""Compute the mean of this GP at each of point of ``Xs`` (``points_to_sample``).

        ``points_to_sample`` may not contain duplicate points. Violating this results in singular covariance matrices.

        .. Note:: Comments are copied from GaussianProcess in gpp_math.hpp and duplicated in
          :meth:`moe.optimal_learning.python.cpp_wrappers.gaussian_process.GaussianProcess.compute_mean_of_points`.

        :param points_to_sample: num_to_sample points (in dim dimensions) being sampled from the GP
        :type points_to_sample: array of float64 with shape (num_to_sample, dim)
        :return: mean: where mean[i] is the mean at points_to_sample[i]
        :rtype: array of float64 with shape (num_to_sample)

        """
        pass

    @abstractmethod
    def compute_grad_mean_of_points(self, points_to_sample, num_derivatives):
        r"""Compute the gradient of the mean of this GP at each of point of ``Xs`` (``points_to_sample``) wrt ``Xs``.

        ``points_to_sample`` may not contain duplicate points. Violating this results in singular covariance matrices.

        Note that ``grad_mu`` is nominally sized: ``grad_mu[num_to_sample][num_to_sample][dim]``. This is
        the the d-th component of the derivative evaluated at the i-th input wrt the j-th input.
        However, for ``0 <= i,j < num_to_sample``, ``i != j``, ``grad_mu[j][i][d] = 0``.
        (See references or implementation for further details.)
        Thus, ``grad_mu`` is stored in a reduced form which only tracks the nonzero entries.

        .. Note:: Comments are copied from GaussianProcess in gpp_math.hpp and duplicated in
          :meth:`moe.optimal_learning.python.cpp_wrappers.gaussian_process.GaussianProcess.compute_grad_mean_of_points`.

        :param points_to_sample: num_to_sample points (in dim dimensions) being sampled from the GP
        :type points_to_sample: array of float64 with shape (num_to_sample, dim)
        :param num_derivatives: return derivatives wrt points_to_sample[0:num_derivatives]; large or negative values are clamped
        :type num_derivatives: int
        :return: grad_mu: gradient of the mean of the GP. ``grad_mu[i][d]`` is actually the gradient
          of ``\mu_i`` wrt ``x_{i,d}``, the d-th dim of the i-th entry of ``points_to_sample``.
        :rtype: array of float64 with shape (num_derivatives, dim)

        """
        pass

    @abstractmethod
    def compute_variance_of_points(self, points_to_sample):
        r"""Compute the variance (matrix) of this GP at each point of ``Xs`` (``points_to_sample``).

        ``points_to_sample`` may not contain duplicate points. Violating this results in singular covariance matrices.

        The variance matrix is symmetric although we currently return the full representation.

        .. Note:: Comments are copied from GaussianProcess in gpp_math.hpp and duplicated in
          :meth:`moe.optimal_learning.python.cpp_wrappers.gaussian_process.GaussianProcess.compute_variance_of_points`.

        :param points_to_sample: num_to_sample points (in dim dimensions) being sampled from the GP
        :type points_to_sample: array of float64 with shape (num_to_sample, dim)
        :return: var_star: variance matrix of this GP
        :rtype: array of float64 with shape (num_to_sample, num_to_sample)

        """
        pass

    @abstractmethod
    def compute_cholesky_variance_of_points(self, points_to_sample):
        r"""Compute the cholesky factorization of the variance (matrix) of this GP at each point of ``Xs`` (``points_to_sample``).

        ``points_to_sample`` may not contain duplicate points. Violating this results in singular covariance matrices.

        :param points_to_sample: num_to_sample points (in dim dimensions) being sampled from the GP
        :type points_to_sample: array of float64 with shape (num_to_sample, dim)
        :return: cholesky factorization of the variance matrix of this GP, lower triangular
        :rtype: array of float64 with shape (num_to_sample, num_to_sample), lower triangle filled in

        """
        pass

    @abstractmethod
    def compute_grad_variance_of_points(self, points_to_sample, num_derivatives):
        r"""Compute the gradient of the variance (matrix) of this GP at each point of ``Xs`` (``points_to_sample``) wrt ``Xs``.

        ``points_to_sample`` may not contain duplicate points. Violating this results in singular covariance matrices.

        This function is similar to compute_grad_cholesky_variance_of_points() (below), except this does not include
        gradient terms from the cholesky factorization. Description will not be duplicated here.

        .. Note:: Comments are copied from GaussianProcess in gpp_math.hpp and duplicated in
          :meth:`moe.optimal_learning.python.cpp_wrappers.gaussian_process.GaussianProcess.compute_grad_variance_of_points`.

        :param points_to_sample: num_to_sample points (in dim dimensions) being sampled from the GP
        :type points_to_sample: array of float64 with shape (num_to_sample, dim)
        :param num_derivatives: return derivatives wrt points_to_sample[0:num_derivatives]; large or negative values are clamped
        :type num_derivatives: int
        :return: grad_var: gradient of the variance matrix of this GP
        :rtype: array of float64 with shape (num_derivatives, num_to_sample, num_to_sample, dim)

        """
        pass

    @abstractmethod
    def compute_grad_cholesky_variance_of_points(self, points_to_sample, num_derivatives):
        r"""Compute the gradient of the cholesky factorization of the variance (matrix) of this GP at each point of ``Xs`` (``points_to_sample``) wrt ``Xs``.

        ``points_to_sample`` may not contain duplicate points. Violating this results in singular covariance matrices.

        This function accounts for the effect on the gradient resulting from
        cholesky-factoring the variance matrix.  See Smith 1995 for algorithm details.

        Note that ``grad_chol`` is nominally sized:
        ``grad_chol[num_to_sample][num_to_sample][num_to_sample][dim]``.
        Let this be indexed ``grad_chol[k][j][i][d]``, which is read the derivative of ``var[j][i]``
        with respect to ``x_{k,d}`` (x = ``points_to_sample``)

        .. Note:: Comments are copied from GaussianProcess in gpp_math.hpp and duplicated in
          :meth:`moe.optimal_learning.python.cpp_wrappers.gaussian_process.GaussianProcess.compute_grad_cholesky_variance_of_points`.

        :param points_to_sample: num_to_sample points (in dim dimensions) being sampled from the GP
        :type points_to_sample: array of float64 with shape (num_to_sample, dim)
        :param var_of_grad: index of ``points_to_sample`` to be differentiated against
        :type var_of_grad: integer in {0, .. ``num_to_sample``-1}
        :return: grad_chol: gradient of the cholesky factorization of the variance matrix of this GP.
          ``grad_chol[k][j][i][d]`` is actually the gradients of ``var_{j,i}`` with
          respect to ``x_{k,d}``, the d-th dimension of the k-th entry of ``points_to_sample``
        :rtype: array of float64 with shape (num_derivatives, num_to_sample, num_to_sample, dim)

        """
        pass

    @abstractmethod
    def add_sampled_points(self, sampled_points):
        r"""Add a sampled points (point, value, noise) to the GP's prior data.

        Also forces recomputation of all derived quantities for GP to remain consistent.

        :param sampled_points: SamplePoint objects to load into the GP (containing point, function value, and noise variance)
        :type sampled_points: single :class:`moe.optimal_learning.python.SamplePoint` or list of SamplePoint objects

        """
        pass

    @abstractmethod
    def sample_point_from_gp(self, point_to_sample, noise_variance=0.0):
        r"""Sample a function value from a Gaussian Process prior, provided a point at which to sample.

        Uses the formula ``function_value = gpp_mean + sqrt(gpp_variance) * w1 + sqrt(noise_variance) * w2``, where ``w1, w2``
        are draws from N(0,1).

        Implementers are responsible for providing a N(0,1) source.

        .. NOTE::
             Set noise_variance to 0 if you want "accurate" draws from the GP.
             BUT if the drawn (point, value) pair is meant to be added back into the GP (e.g., for testing), then this point
             MUST be drawn with noise_variance equal to the noise associated with "point" as a member of "points_sampled"

        .. Note:: Comments are copied from GaussianProcess in gpp_math.hpp and duplicated in
          :meth:`moe.optimal_learning.python.cpp_wrappers.gaussian_process.GaussianProcess.sample_point_from_gp`.

        :param point_to_sample: point (in dim dimensions) at which to sample from this GP
        :type points_to_sample: array of float64 with shape (dim)
        :param noise_variance: amount of noise to associate with the sample
        :type noise_variance: float64 >= 0.0
        :return: sample_value: function value drawn from this GP
        :rtype: float64

        """
        pass
