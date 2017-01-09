# -*- coding: utf-8 -*-
"""Utilities for computing covariance matrices and related structures."""
import numpy


def build_covariance_matrix(covariance, points_sampled, noise_variance=None):
    r"""Compute the covariance matrix, ``K``, of a list of points, ``X_i``.

    .. NOTE:: These comments are copied from BuildCovarianceMatrix() in gpp_math.cpp.

    Matrix is computed as:
    ``A_{i,j} = covariance(X_i, X_j) + \delta_{i,j}*noise_i``.
    where ``\delta_{i,j}`` is the Kronecker ``delta``, equal to 1 if ``i == j`` and 0 else.
    Result is SPD assuming covariance operator is SPD and points are unique.

    Generally, this is called from other functions with "points_sampled" as the input and not any
    arbitrary list of points; hence the very specific input name.

    Point list cannot contain duplicates.  Doing so (or providing nearly duplicate points) can lead to
    semi-definite matrices or very poor numerical conditioning.

    :param covariance: the covariance function encoding assumptions about the GP's behavior on our data
    :type covariance: interfaces.covariance_interface.CovarianceInterface subclass
    :param points_sampled: points, ``X_i``
    :type points_sampled: array of float64 with shape (points_sampled.shape[0], dim)
    :param noise_variance: i-th entry is amt of noise variance to add to i-th diagonal entry; i.e., noise measuring i-th point
    :type noise_variance: array of float64 with shape (points_sampled.shape[0])
    :return: covariance matrix
    :rtype: array of float64 with shape(points_sampled.shape[0], points_sampled.shape[0]), order='F'

    .. Note:: Fortran ordering is important here; scipy.linalg factor/solve methods
        (e.g., cholesky, solve_triangular) implicitly require order='F' to enable
        overwriting. This output is commonly overwritten.

    """
    cov_mat = numpy.zeros((points_sampled.shape[0], points_sampled.shape[0]), order='F')
    # Only form the lower triangle; matrix is symmetric.
    for j, point_two in enumerate(points_sampled):
        for i, point_one in enumerate(points_sampled[j:, ...], start=j):
            cov_mat[i, j] = covariance.covariance(point_one, point_two)

    # Copy into the (strict) upper triangle.
    # TODO(GH-62): We could avoid this step entirely and only fill the lower triangle.
    cov_mat += numpy.tril(cov_mat, k=-1).T

    if noise_variance is not None:
        cov_mat += numpy.diag(noise_variance)

    return cov_mat


def build_mix_covariance_matrix(covariance, points_sampled, points_to_sample):
    """Compute the "mix" covariance matrix, ``Ks``, of ``Xs`` and ``X`` (``points_to_sample`` and ``points_sampled``, respectively).

    .. NOTE:: These comments are copied from BuildMixCovarianceMatrix() in gpp_math.cpp.

    Matrix is computed as:
    ``A_{i,j} = covariance(X_i, Xs_j).``
    Result is not guaranteed to be SPD and need not even be square.

    Generally, this is called from other functions with "points_sampled" and "points_to_sample" as the
    input lists and not any arbitrary list of points; hence the very specific input name.  But this
    is not a requirement.

    Point lists cannot contain duplicates with each other or within themselves.

    :param covariance: the covariance function encoding assumptions about the GP's behavior on our data
    :type covariance: interfaces.covariance_interface.CovarianceInterface subclass
    :param points_sampled: points, ``X_i``
    :type points_sampled: array of float64 with shape (points_sampled.shape[0], dim)
    :param points_to_sample: points, ``Xs_i``
    :type points_to_sample: array of float64 with shape (points_to_sample.shape[0], dim)
    :return: "mix" covariance matrix
    :rtype: array of float64 with shape (points_sampled.shape[0], points_to_sample.shape[0]), order='F'

    .. Note:: Fortran ordering is important here; scipy.linalg factor/solve methods
        (e.g., cholesky, solve_triangular) implicitly require order='F' to enable
        overwriting. This output is commonly overwritten.

    """
    cov_mat = numpy.empty((points_sampled.shape[0], points_to_sample.shape[0]), order='F')
    for j, point_two in enumerate(points_to_sample):
        for i, point_one in enumerate(points_sampled):
            cov_mat[i, j] = covariance.covariance(point_one, point_two)

    return cov_mat


def build_hyperparameter_grad_covariance_matrix(covariance, points_sampled):
    r"""Build ``A_{jik} = \pderiv{K_{ij}}{\theta_k}``.

    .. NOTE:: These comments are copied from BuildHyperparameterGradCovarianceMatrix() in gpp_model_selection.cpp.

    Build ``A_{jik} = \pderiv{K_{ij}}{\theta_k}``
    Hence the outer loop structure is identical to BuildCovarianceMatrix().

    Note the structure of the resulting tensor is ``num_hyperparameters`` blocks of size
    ``num_sampled X num_sampled``.  Consumers of this want ``dK/d\theta_k`` located sequentially.
    However, for a given pair of points (x, y), it is more efficient to compute all
    hyperparameter derivatives at once.  Thus the innermost loop writes to all
    ``num_hyperparameters`` blocks at once.

    Consumers of this result generally require complete storage (i.e., will not take advantage
    of its symmetry), so instead of ignoring the upper triangles, we copy them from the
    (already-computed) lower triangles to avoid redundant work.

    Since CovarianceInterface.HyperparameterGradCovariance() returns a vector of size ``|\theta_k|``,
    the inner loop writes all relevant entries of ``A_{jik}`` simultaneously to prevent recomputation.

    :param covariance: the covariance function encoding assumptions about the GP's behavior on our data
    :type covariance: interfaces.covariance_interface.CovarianceInterface subclass
    :param points_sampled: points, ``X_i``
    :type points_sampled: array of float64 with shape (points_sampled.shape[0], dim)
    :return: gradient of covariance matrix wrt hyperparameters
    :rtype: array of float64 with shape (points_sampled.shape[0], points_sampled.shape[0], num_hyperparameters), order='F'

    .. Note:: Fortran ordering is important here; scipy.linalg factor/solve methods
        (e.g., cholesky, solve_triangular) implicitly require order='F' to enable
        overwriting. This output is commonly overwritten.

    """
    cov_mat = numpy.empty((points_sampled.shape[0], points_sampled.shape[0], covariance.num_hyperparameters), order='F')
    for i, point_one in enumerate(points_sampled):
        for j, point_two in enumerate(points_sampled):
            cov_mat[j, i, ...] = covariance.hyperparameter_grad_covariance(point_one, point_two)

    return cov_mat
