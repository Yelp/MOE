# -*- coding: utf-8 -*-
"""Base test case class for optimal_learning tests; includes some additional asserts for numerical tests.

TODO(eliu): (GH-175) Generalize ping testing code used in some derivative tests (e.g., covariance, log likelihood pinging)
to be more DRY (model after C++ test cases). We can set up one ping tester and just pass it objective functions.

"""
import numpy
import testify as T


class OptimalLearningTestCase(T.TestCase):

    """Base test case for the optimal_learning library.

    This includes extra asserts for checking relative differences of floating point scalars/vectors and
    a routine to check that points are distinct.

    """

    def assert_scalar_within_relative(self, value, truth, tol):
        """Check whether a scalar ``value`` is relatively equal to ``truth``: ``|value - truth|/|truth| <= tol``.

        :param value: scalar to check
        :type value: float64
        :param truth: exact/desired result
        :type value: float64
        :param tol: max permissible absolute difference
        :type tol: float64
        :raise: AssertionError value, truth are not relatively equal

        """
        denom = numpy.fabs(truth)
        if denom < numpy.finfo('float64').tiny:
            denom = 1.0  # do not divide by 0
        diff = numpy.fabs((value - truth) / denom)
        T.assert_lte(
            diff,
            tol,
            message='value = {0:.18E}, truth = {1:.18E}, diff = {2:.18E}, tol = {3:.18E}'.format(value, truth, diff, tol),
        )

    def assert_vector_within_relative(self, value, truth, tol):
        """Check whether a vector is element-wise relatively equal to ``truth``: ``|value[i] - truth[i]|/|truth[i]| <= tol``.

        :param value: scalar to check
        :type value: float64
        :param truth: exact/desired result
        :type value: float64
        :param tol: max permissible relative difference
        :type tol: float64
        :raise: AssertionError value[i], truth[i] are not relatively equal for every i

        """
        T.assert_equal(
            value.shape,
            truth.shape,
            message='value.shape = {0} != truth.shape = {1}'.format(value.shape, truth.shape),
        )
        for index in numpy.ndindex(value.shape):
            self.assert_scalar_within_relative(value[index], truth[index], tol)

    def assert_points_distinct(self, point_list, tol):
        """Check whether the distance between every pair of points is larger than tolerance.

        :param point_list: points to check
        :type point_list: array of float64 with shape (num_points, dim)
        :param tol: the minimum allowed (absolute) distance between points
        :type tol: float64
        :raise: AssertionError when every point is not more than tolerance distance apart

        """
        for i in xrange(point_list.shape[0]):
            for j in xrange(i + 1, point_list.shape[0]):
                temp = point_list[i, ...] - point_list[j, ...]
                dist = numpy.linalg.norm(temp)
                self.assert_scalar_within_relative(dist, 0.0, tol)
