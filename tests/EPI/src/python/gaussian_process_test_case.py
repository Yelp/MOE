# -*- coding: utf-8 -*-

import testify as T
import numpy

from optimal_learning.EPI.src.python.data_containers import SamplePoint
from optimal_learning.EPI.src.python.models.covariance_of_process import CovarianceOfProcess
from optimal_learning.EPI.src.python.models.optimal_gaussian_process_linked_cpp import OptimalGaussianProcessLinkedCpp
from optimal_learning.EPI.src.python.models.optimal_gaussian_process import OptimalGaussianProcess as OptimalGaussianProcessPython
import optimal_learning.EPI.src.python.lib.math

class GaussianProcessTestCase(T.TestCase):

    default_domain = [[-1.0, 1.0], [-1.0, 1.0]]
    default_covariance_signal_variance = 1.0
    default_covariance_length = [0.5]
    default_gaussian_process_class = OptimalGaussianProcessLinkedCpp
    default_sample_variance = 0.01
    tol = 1e-12 # TODO eliu look into this ticket #43006

    def assert_relatively_equal(self, value_one, value_two, tol=None):
        """Assert that two values are relatively equal,
        |value_one - value_two|/|value_one| <= eps
        """
        if tol is None:
            tol = self.tol
        denom = abs(value_one)
        if (denom == 0.0):
            denom = 1.0;
        T.assert_lte(abs(value_one - value_two)/denom,
                 tol,
                 )

    def assert_lists_relatively_equal(self, list_one, list_two, tol=None):
        """Assert two lists are relatively equal."""
        T.assert_length(list_one, len(list_two))
        for i, list_one_item in enumerate(list_one):
            list_two_item = list_two[i]
            self.assert_relatively_equal(list_one_item, list_two_item, tol)

    def assert_matrix_relatively_equal(self, matrix_one, matrix_two, tol=None):
        """Assert two matrices are relatively equal."""
        for row_idx, row_matrix_one in enumerate(matrix_one):
            row_matrix_two = matrix_two[row_idx]
            self.assert_lists_relatively_equal(row_matrix_one, row_matrix_two, tol)

    def _make_default_covariance_of_process(self, signal_variance=None, length=None):
        """Make a default covariance of process with optional parameters
        """
        if signal_variance is None:
            signal_variance = self.default_covariance_signal_variance
        if length is None:
            length = self.default_covariance_length
        hyperparameters = [signal_variance]
        hyperparameters.extend(length)

        return CovarianceOfProcess(hyperparameters=hyperparameters)

    def _make_default_gaussian_process(self, gaussian_process_class=None, domain=None, default_sample_variance=None, signal_variance=None, length=None, covariance_of_process=None, max_number_of_threads=1):
        """Make a default gaussian process with optional parameters
        using a default covariance of process
        """
        if domain is None:
            domain = self.default_domain
        if gaussian_process_class is None:
            gaussian_process_class = self.default_gaussian_process_class
        if default_sample_variance is None:
            default_sample_variance = self.default_sample_variance

        # build the covariance_of_process object
        if covariance_of_process is None:
            covariance_of_process = self._make_default_covariance_of_process(
                    signal_variance=signal_variance,
                    length=length,
                    )

        return gaussian_process_class(
            domain=domain,
            covariance_of_process=covariance_of_process,
            default_sample_variance=default_sample_variance,
            max_number_of_threads=max_number_of_threads,
            )

    def _sample_points_from_gaussian_process(self, gaussian_process, points_to_sample, random_normal_values=None, extra_gaussian_process=None, default_sample_variance=None):
        """Samples the points in points_to_sample by drawing them from the gaussian_process
        and them adding them to gaussian_process, and also to extra_gaussian_process if it is provided
        will use the values provided in random_normal_values for drawing the points if given need 2 per point
        see gaussian_process.sample_from_process for details
        """
        if default_sample_variance is None:
                default_sample_variance = self.default_sample_variance

        for point_on, point_to_sample in enumerate(points_to_sample):
            # wrap the point in a numpy array
            point_point = numpy.array(point_to_sample)

            # grab the normal values we need (or not)
            if random_normal_values:
                random_normal = random_normal_values[point_on * 2]
                sample_variance_normal = random_normal_values[point_on * 2 + 1]
            else:
                random_normal = None
                sample_variance_normal = default_sample_variance

            # draw a value from gaussian_process at point_point
            point_val = gaussian_process.sample_from_process(
                    point_point,
                    random_normal=random_normal,
                    )

            # wrap the point in the SamplePoint class
            sample_point = SamplePoint(point_point, point_val, default_sample_variance)

            # add the point to the GPs
            gaussian_process.add_sample_point(sample_point)
            if extra_gaussian_process:
                extra_gaussian_process.add_sample_point(sample_point)

    def _make_random_processes_from_latin_hypercube(self, domain, num_points_in_sample, default_sample_variance=None, covariance_of_process=None, default_signal_variance=None):
        if default_sample_variance is None:
            default_sample_variance = self.default_sample_variance

        cpp_GP = self._make_default_gaussian_process(
                gaussian_process_class=OptimalGaussianProcessLinkedCpp,
                domain=domain,
                covariance_of_process=covariance_of_process,
                default_sample_variance=default_sample_variance,
                )
        python_GP = self._make_default_gaussian_process(
                gaussian_process_class=OptimalGaussianProcessPython,
                domain=domain,
                covariance_of_process=covariance_of_process,
                default_sample_variance=default_sample_variance,
                )

        # A num_points_in_sample random latin hypercube points
        stencil_points_to_sample = optimal_learning.EPI.src.python.lib.math.get_latin_hypercube_points(num_points_in_sample, domain)

        # Sample stencil
        self._sample_points_from_gaussian_process(
                gaussian_process=cpp_GP,
                points_to_sample=stencil_points_to_sample,
                extra_gaussian_process=python_GP,
                default_sample_variance=default_sample_variance,
                )

        return cpp_GP, python_GP
