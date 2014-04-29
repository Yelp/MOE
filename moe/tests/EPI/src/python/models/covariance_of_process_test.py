# -*- coding: utf-8 -*-

import testify as T
import numpy

from moe.tests.EPI.src.python.OLD_gaussian_process_test_case import OLDGaussianProcessTestCase
from moe.optimal_learning.EPI.src.python.models.covariance_of_process import CovarianceOfProcess
from moe.optimal_learning.EPI.src.python.models.gaussian_process import GaussianProcess
from moe.optimal_learning.EPI.src.python.models.optimal_gaussian_process_linked_cpp import OptimalGaussianProcessLinkedCpp
from moe.optimal_learning.EPI.src.python.lib.math import get_latin_hypercube_points

MACHINE_PRECISION = 1e-8

class HyperparameterUpdateTest(OLDGaussianProcessTestCase):
    """Tests optimal_learning.EPI.src.python.models.covariance_of_process.update_hyper_parameters
    """

    def test_hyperparameters_cov_grad_pings(self):
        h = 2e-3
        domain = [[-5.0, 5.0], [-5.0, 5.0]]
        real_length = numpy.array([0.5, 0.9])

        GP = self._make_default_gaussian_process(
                gaussian_process_class=GaussianProcess,
                domain=domain,
                length=real_length,
                default_sample_variance=0.01
                )
        GP.cop.GP = GP

        # sample some points
        points_to_sample = get_latin_hypercube_points(20, domain)
        self._sample_points_from_gaussian_process(
                GP,
                points_to_sample
                )

        hyperparameters = numpy.array(GP.cop.hyperparameters)

        points_to_test_one = get_latin_hypercube_points(10, domain)

        for test_point_on, point_one in enumerate(points_to_test_one):
            point_two = point_one + 0.01 # we want the points to be close so there is a >> 0 grad
            for i, hyperparameter in enumerate(hyperparameters):
                # the analytic method
                calc_grad = GP.cop.hyperparameter_grad_table[i](
                        point_one,
                        point_two,
                        kwargs=GP.cop.hyperparameter_grad_table_kwargs[i]
                        )

                # finite difference
                # save old
                hyperparameters_old = numpy.array(GP.cop.hyperparameters)

                # hyper + h
                hyperparameters_plus_step = numpy.array(GP.cop.hyperparameters)
                hyperparameters_plus_step[i] += h
                GP.cop.hyperparameters = hyperparameters_plus_step
                fd_plus_step = GP.cop.cov(point_one, point_two)
                GP.cop.hyperparameters = hyperparameters_old

                # hyper - h
                hyperparameters_minus_step = numpy.array(GP.cop.hyperparameters)
                hyperparameters_minus_step[i] -= h
                GP.cop.hyperparameters = hyperparameters_minus_step
                fd_minus_step = GP.cop.cov(point_one, point_two)
                GP.cop.hyperparameters = hyperparameters_old

                # calculate finite diff
                fd_grad = (fd_plus_step - fd_minus_step) / (2.0 * h)

                self.assert_relatively_equal(calc_grad, fd_grad, 5e-5)

    def test_hyperparameters_marginal_grad_pings(self):
        h = 2.0e-3
        domain = [[-1.0, 1.0], [-1.0, 1.0]]
        real_length = numpy.array([0.5, 0.9])

        GP = self._make_default_gaussian_process(
                gaussian_process_class=GaussianProcess,
                domain=domain,
                length=real_length,
                default_sample_variance=0.01
                )
        GP.cop.GP = GP

        # sample some points
        points_to_sample = get_latin_hypercube_points(20, domain)
        self._sample_points_from_gaussian_process(
                GP,
                points_to_sample
                )

        hyperparameters = numpy.array(GP.cop.hyperparameters)

        fd_grad = numpy.zeros(len(GP.cop.hyperparameters))
        for i, _ in enumerate(fd_grad):
            hyperparameters_plus_step = numpy.array(GP.cop.hyperparameters)
            hyperparameters_plus_step[i] += h
            hyperparameters_minus_step = numpy.array(GP.cop.hyperparameters)
            hyperparameters_minus_step[i] -= h

            fg_part_forward = -GP.cop._neg_log_marginal_likelihood_given_hyperparameters(
                    hyperparameters_plus_step
                    )
            fg_part_backward = -GP.cop._neg_log_marginal_likelihood_given_hyperparameters(
                    hyperparameters_minus_step
                    )
            fd_grad[i] = (fg_part_forward - fg_part_backward) / (2.0 * h)

        calc_grad = GP.cop._gradient_of_marginal_wrt_hyperparameters(hyperparameters)

        for i, fd_grad_part in enumerate(fd_grad):
            calc_grad_part = calc_grad[i]
            self.assert_relatively_equal(fd_grad_part, calc_grad_part, 6e-4)

    def test_marginal_stays_below_zero(self):
        real_length = numpy.array([2.0])
        domain = [[0.0, 10.0], [0.0, 10.0]]

        # make a python based gaussian process
        GP = self._make_default_gaussian_process(
                gaussian_process_class=GaussianProcess,
                domain=domain,
                length=real_length,
                default_sample_variance=0.01
                )

        # grab 50 random points to sample
        points_to_sample = get_latin_hypercube_points(50, domain)

        for point in points_to_sample:
            # draw the point from the GP and add it back
            self._sample_points_from_gaussian_process(
                    GP,
                    [point]
                    )
            # assert the marginal stays below zero
            T.assert_lte(
                    GP.get_log_marginal_likelihood(),
                    0.0
                    )

    @T.suite('disabled', reason='Broken, #53237')
    def _test_hyperparameter_update_does_no_worse_than_initial_points(self):
        """ Broken test!
        This test calls GP.cop.update_hyperparameters() to attempt hyperparameter optimization.
        The current optimization options all call out to scipy's suite, and none of them can handle
        constrained optimization problems.

        As a result, this test always fails, either attempting
        to query with negative hyperparameters or [nearly] 0 hyperparameters. The former
        can result in log(negative_number) which breaks things; the latter results in a severe
        loss of precision, causing scipy to return without accomplishing anything.
        """
        domain = [[-5.0, 5.0], [-5.0, 5.0]]
        real_length = numpy.array([0.5, 0.9])

        GP = self._make_default_gaussian_process(
                gaussian_process_class=GaussianProcess,
                domain=domain,
                length=real_length,
                default_sample_variance=0.01
                )

        # sample some points
        points_to_sample = get_latin_hypercube_points(20, domain)
        self._sample_points_from_gaussian_process(
                GP,
                points_to_sample
                )

        # save the current likelihood
        GP.cop.hyperparameters = [1.0, 0.55, 0.85]
        old_marginal_likelihood = GP.get_log_marginal_likelihood()

        # sample some points
        points_to_sample = get_latin_hypercube_points(20, domain)
        self._sample_points_from_gaussian_process(
                GP,
                points_to_sample,
                )

        # update the hyperparameters and find the new likelihood
        GP.cop.update_hyperparameters(GP)
        new_marginal_likelihood = GP.get_log_marginal_likelihood()

        # assert we did no worse by updating the hyperparameters
        T.assert_lt(
                old_marginal_likelihood,
                new_marginal_likelihood,
                )

    def test_python_and_cpp_return_same_marginal(self):
        domain = [[-5.0, 5.0], [-5.0, 5.0]]
        real_length = numpy.array([0.5, 0.8])

        python_GP = self._make_default_gaussian_process(
                gaussian_process_class=GaussianProcess,
                domain=domain,
                length=real_length,
                default_sample_variance=0.01,
                )
        cpp_GP = self._make_default_gaussian_process(
                gaussian_process_class=OptimalGaussianProcessLinkedCpp,
                domain=domain,
                length=real_length,
                default_sample_variance=0.01,
                )

        # grab 50 random points to sample
        points_to_sample = get_latin_hypercube_points(50, domain)

        for point in points_to_sample:
            # draw the point from the GP and add it back
            self._sample_points_from_gaussian_process(
                    python_GP,
                    [point],
                    extra_gaussian_process=cpp_GP,
                    )
            # assert the marginals are relatively equal
            self.assert_relatively_equal(
                python_GP.get_log_marginal_likelihood(),
                cpp_GP.get_log_marginal_likelihood(),
                1.0e-14,
                )

            # assert gradients (wrt hyperparams) of marginals are relatively equal
            python_GP.cop.GP = python_GP
            python_grad_marginal = python_GP.cop._gradient_of_marginal_wrt_hyperparameters(python_GP.cop.hyperparameters)
            cpp_grad_marginal = cpp_GP.get_hyperparam_grad_log_marginal_likelihood()
            for i in range(len(cpp_grad_marginal)):
                self.assert_relatively_equal(python_grad_marginal[i], cpp_grad_marginal[i], 5.0e-13)


class CovarianceOfProcessTest(OLDGaussianProcessTestCase):

    one_dim_test_sets = [
                [1.0, [0.1]], # default
                [2.0, [0.1]],
                [1.0, [1.0]],
                [0.1, [10.0]],
                [1.0, [1.0]],
                [0.1, [10.0]],
                ]

    three_dim_test_sets = [
                [1.0, [0.1, 0.1, 0.1]],
                [1.0, [0.1, 0.2, 0.1]],
                [1.0, [0.1, 0.2, 0.3]],
                [2.0, [0.1, 0.1, 0.1]],
                [2.0, [0.1, 0.2, 0.1]],
                [2.0, [0.1, 0.2, 0.3]],
                [0.1, [10.0, 1.0, 0.1]],
                [1.0, [10.0, 1.0, 0.1]],
                [10.0, [10.0, 1.0, 0.1]],
                [0.1, [10.0, 1.0, 0.1]],
                [1.0, [10.0, 1.0, 0.1]],
                [10.0, [10.0, 1.0, 0.1]],
                ]

    def test_default_cov_one_dim(self):
        """Test the default covariance function against correct values for different sets of hyperparameters
        """

        for signal_variance, length in self.one_dim_test_sets:
            cop = self._make_default_covariance_of_process(
                    signal_variance=signal_variance,
                    length=length
                    )

            T.assert_equal(cop.is_default_covariance, True)

            # One length away
            T.assert_lte(
                    numpy.abs(cop.cov(numpy.array([0]), numpy.array(length)) - signal_variance * numpy.exp(-0.5)),
                    MACHINE_PRECISION
                    )
            # Sym
            T.assert_lte(
                    numpy.abs(cop.cov(numpy.array(length), numpy.array([0])) - signal_variance * numpy.exp(-0.5)),
                    MACHINE_PRECISION
                    )

            # One length * sqrt 2 away
            T.assert_lte(
                    numpy.abs(cop.cov(numpy.array([0]), numpy.array([length[0] * numpy.sqrt(2)])) - signal_variance * numpy.exp(-1.0)),
                    MACHINE_PRECISION
                    )

    def test_default_cov_two_dim_with_one_length(self):
        """Test the default covariance function against correct values for different sets of hyperparameters
        """

        for signal_variance, length in self.one_dim_test_sets:
            cop = self._make_default_covariance_of_process(
                    signal_variance=signal_variance,
                    length=length
                    )

            T.assert_equal(cop.is_default_covariance, True)

            # One length away
            T.assert_lte(
                    numpy.abs(cop.cov(numpy.array([0, 0]), numpy.array([0, length[0]])) - signal_variance * numpy.exp(-0.5)),
                    MACHINE_PRECISION
                    )
            T.assert_lte(
                    numpy.abs(cop.cov(numpy.array([0, 0]), numpy.array([length[0], 0])) - signal_variance * numpy.exp(-0.5)),
                    MACHINE_PRECISION
                    )
            T.assert_lte(
                    numpy.abs(cop.cov(numpy.array([0, 0]), numpy.array([numpy.sqrt(2)/2.0*length[0], numpy.sqrt(2)/2.0*length[0]])) - signal_variance * numpy.exp(-0.5)),
                    MACHINE_PRECISION
                    )

            # One length * sqrt 2 away
            T.assert_lte(
                    numpy.abs(cop.cov(numpy.array([0, 0]), numpy.array([0, length[0] * numpy.sqrt(2)])) - signal_variance * numpy.exp(-1.0)),
                    MACHINE_PRECISION
                    )
            # Sym
            T.assert_lte(
                    numpy.abs(cop.cov(numpy.array([0, length[0] * numpy.sqrt(2)]), numpy.array([0, 0])) - signal_variance * numpy.exp(-1.0)),
                    MACHINE_PRECISION
                    )

    def test_default_cov_three_dim_with_full_length(self):
        """Test the default covariance function against correct values for different sets of hyperparameters
        """
        for signal_variance, length in self.three_dim_test_sets:

            cop = self._make_default_covariance_of_process(
                    signal_variance=signal_variance,
                    length=length
                    )

            T.assert_equal(cop.is_default_covariance, True)

            # One length away
            T.assert_lte(
                    numpy.abs(cop.cov(numpy.array([0, 0, 0]), numpy.array([0, 0, length[2]])) - signal_variance * numpy.exp(-0.5)),
                    MACHINE_PRECISION
                    )
            T.assert_lte(
                    numpy.abs(cop.cov(numpy.array([0, 0, 0]), numpy.array([0, length[1], 0])) - signal_variance * numpy.exp(-0.5)),
                    MACHINE_PRECISION
                    )
            T.assert_lte(
                    numpy.abs(cop.cov(numpy.array([0, 0, 0]), numpy.array([length[0], 0, 0])) - signal_variance * numpy.exp(-0.5)),
                    MACHINE_PRECISION
                    )
            T.assert_lte(
                    numpy.abs(cop.cov(numpy.array([0, 0, 0]), numpy.array([numpy.sqrt(3)/3.0*length[0], numpy.sqrt(3)/3.0*length[1], numpy.sqrt(3)/3.0*length[2]])) - signal_variance * numpy.exp(-0.5)),
                    MACHINE_PRECISION
                    )

            # Sym
            T.assert_lte(
                    numpy.abs(cop.cov(numpy.array([numpy.sqrt(3)/3.0*length[0], numpy.sqrt(3)/3.0*length[1], numpy.sqrt(3)/3.0*length[2]]), numpy.array([0, 0, 0])) - signal_variance * numpy.exp(-0.5)),
                    MACHINE_PRECISION
                    )

    def test_default_grad_cov_three_dim_with_full_length(self):
        """Test the default covariance function against correct values for different sets of hyperparameters
        """

        for signal_variance, length in self.three_dim_test_sets:
            cop = self._make_default_covariance_of_process(
                    signal_variance=signal_variance,
                    length=length
                    )

            T.assert_equal(cop.is_default_covariance, True)

            # Same point
            grad_cov = cop.grad_cov(numpy.array([0, 0, 0]), numpy.array([0, 0, 0]))
            T.assert_equal(grad_cov[0], 0.0)
            T.assert_equal(grad_cov[1], 0.0)
            T.assert_equal(grad_cov[2], 0.0)

            # One length away
            grad_cov = cop.grad_cov(numpy.array([0, 0, 0]), numpy.array([0, 0, length[2]]))
            T.assert_equal(grad_cov[0], 0.0)
            T.assert_equal(grad_cov[1], 0.0)
            T.assert_gt(grad_cov[2], 0.0)

            # Sym is opposite
            grad_cov = cop.grad_cov(numpy.array([0, 0, length[2]]), numpy.array([0, 0, 0]))
            T.assert_equal(grad_cov[0], 0.0)
            T.assert_equal(grad_cov[1], 0.0)
            T.assert_lt(grad_cov[2], 0.0)

    def test_setting_new_cov_function(self):
        """Test setting a new cov function
        """

        def always_one(point_one, point_two):
            return 1.0

        def always_zero(point_one, point_two):
            return numpy.zeros(len(point_one))

        cop = CovarianceOfProcess(cov_func=always_one, grad_cov_func=always_zero)

        T.assert_equal(cop.is_default_covariance, False)

        # Test cov
        T.assert_equal(
                cop.cov(numpy.array([0]), numpy.array([0])),
                1.0
                )
        T.assert_equal(
                cop.cov(numpy.array([1]), numpy.array([3])),
                1.0
                )
        T.assert_equal(
                cop.cov(numpy.array([0, 0, 0]), numpy.array([0, 0, 0])),
                1.0
                )
        T.assert_equal(
                cop.cov(numpy.array([1, 2, 3]), numpy.array([3, 2, 1])),
                1.0
                )

        # Test grad_cov
        T.assert_equal(
                cop.grad_cov(numpy.array([0]), numpy.array([0]))[0],
                0.0
                )
        T.assert_equal(
                cop.grad_cov(numpy.array([1]), numpy.array([3]))[0],
                0.0
                )
        T.assert_equal(
                cop.grad_cov(numpy.array([0, 0, 0]), numpy.array([0, 0, 0]))[0],
                0.0
                )
        T.assert_equal(
                cop.grad_cov(numpy.array([1, 2, 3]), numpy.array([3, 2, 1]))[1],
                0.0
                )


if __name__ == "__main__":
    T.run()
