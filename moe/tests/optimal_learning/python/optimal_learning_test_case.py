# -*- coding: utf-8 -*-
"""Base test case class for optimal_learning tests; includes some additional asserts for numerical tests.

TODO(GH-175): Generalize ping testing code used in some derivative tests (e.g., covariance, log likelihood pinging)
to be more DRY (model after C++ test cases). We can set up one ping tester and just pass it objective functions.

"""
import numpy


class OptimalLearningTestCase(object):

    """Base test case for the optimal_learning library.

    This includes extra asserts for checking relative differences of floating point scalars/vectors and
    a routine to check that points are distinct.

    """

    @staticmethod
    def assert_scalar_within_absolute(value, truth, tol):
        """Check whether a scalar ``value`` is equal to ``truth``: ``|value - truth| <= tol``.

        :param value: scalar to check
        :type value: float64
        :param truth: exact/desired result
        :type value: float64
        :param tol: max permissible absolute difference
        :type tol: float64
        :raise: AssertionError value, truth are not equal to within tolerance

        """
        __tracebackhide__ = True
        diff = numpy.fabs(value - truth)
        assert diff <= tol, 'value = {0:.18E}, truth = {1:.18E}, diff = {2:.18E}, tol = {3:.18E}'.format(value, truth, diff, tol)

    @staticmethod
    def assert_scalar_within_relative(value, truth, tol):
        """Check whether a scalar ``value`` is relatively equal to ``truth``: ``|value - truth|/|truth| <= tol``.

        :param value: scalar to check
        :type value: float64
        :param truth: exact/desired result
        :type value: float64
        :param tol: max permissible relative difference
        :type tol: float64
        :raise: AssertionError value, truth are not relatively equal

        """
        __tracebackhide__ = True
        denom = numpy.fabs(truth)
        if denom < numpy.finfo(numpy.float64).tiny:
            denom = 1.0  # do not divide by 0
        diff = numpy.fabs((value - truth) / denom)
        assert diff <= tol, 'value = {0:.18E}, truth = {1:.18E}, diff = {2:.18E}, tol = {3:.18E}'.format(value, truth, diff, tol)

    @staticmethod
    def assert_vector_within_relative(value, truth, tol):
        """Check whether a vector is element-wise relatively equal to ``truth``: ``|value[i] - truth[i]|/|truth[i]| <= tol``.

        :param value: vector to check
        :type value: array of float64 with arbitrary shape
        :param truth: exact/desired vector result
        :type value: array of float64 with shape matching ``value``
        :param tol: max permissible relative difference
        :type tol: float64
        :raise: AssertionError value[i], truth[i] are not relatively equal for every i

        """
        __tracebackhide__ = True
        assert value.shape == truth.shape, 'value.shape = {0} != truth.shape = {1}'.format(value.shape, truth.shape)
        for index in numpy.ndindex(value.shape):
            OptimalLearningTestCase.assert_scalar_within_relative(value[index], truth[index], tol)

    @staticmethod
    def assert_points_distinct(point_list, tol):
        """Check whether the distance between every pair of points is larger than tolerance.

        :param point_list: points to check
        :type point_list: array of float64 with shape (num_points, dim)
        :param tol: the minimum allowed (absolute) distance between points
        :type tol: float64
        :raise: AssertionError when every point is not more than tolerance distance apart

        """
        __tracebackhide__ = True
        for i in xrange(point_list.shape[0]):
            for j in xrange(i + 1, point_list.shape[0]):
                temp = point_list[i, ...] - point_list[j, ...]
                dist = numpy.linalg.norm(temp)
                OptimalLearningTestCase.assert_scalar_within_relative(dist, 0.0, tol)
