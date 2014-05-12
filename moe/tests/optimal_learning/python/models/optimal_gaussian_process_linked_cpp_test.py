# -*- coding: utf-8 -*-

import testify as T
import numpy
import random

from moe.tests.optimal_learning.python.OLD_gaussian_process_test_case import OLDGaussianProcessTestCase
import moe.build.GPP as C_GP
import moe.optimal_learning.python.lib.math
from moe.optimal_learning.python.data_containers import SamplePoint


class GaussianProcessNumericalAnalysisTestCase(OLDGaussianProcessTestCase):
    tol = 1e-12 # TODO eliu look into this ticket #43006
    rel_tol = numpy.finfo(numpy.float64).tiny # 1e-308

    def _assert_relative_diff_lte_tol(self, val_one, val_two):
        if val_one == val_two:
            return True
        elif val_one < self.rel_tol and val_two < self.rel_tol:
            return True

        self.assert_relatively_equal(
                val_one,
                val_two,
                tol=self.tol
                )

class Get1DAnalyticEIAndGradTest(GaussianProcessNumericalAnalysisTestCase):
    """Tests the python vs cpp implementations of .get_mean_and_var_of_points
    """
    tol = 1e-10 # TODO eliu look into this ticket #43006

    def test_random_process_returns_same_1D_EI_and_grad(self):
        """Test that that cpp and python return similar results for
        EI and grad_EI in the analytic setting for random points
        drawn from a random prior
        """
        test_cases = [ # [num_points_to_test, length_scale, [domain]]
                [4, 0.1, [[-5.0, 15.0]]],
                [13, 0.1, [[-10.0, 10.0], [-10.0, 10.0], [-10.0, 10.0]]],
                [11, 0.5, [[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]]],
                [10, 0.01, [[50.0, 55.0], [-80.1, -76.4], [-15005.5, -15000.0]]],
                ]

        for [num_points_in_sample, length_scale, domain] in test_cases:
            covariance_of_process = self._make_default_covariance_of_process(length=[length_scale])

            # make the gaussian processes
            cpp_GP, python_GP = self._make_random_processes_from_latin_hypercube(
                    domain,
                    num_points_in_sample,
                    covariance_of_process=covariance_of_process,
                    default_sample_variance=0.0
                    )

            # select a stencil of num_points_to_test random latin hypercube points
            for _ in range(10):
                points_to_sample = moe.optimal_learning.python.lib.math.get_latin_hypercube_points(num_points_in_sample, domain)

                EI_c = cpp_GP.get_1D_analytic_expected_improvement(points_to_sample[0])
                EI_p = python_GP.get_1D_analytic_expected_improvement(points_to_sample[0])
                self._assert_relative_diff_lte_tol(EI_p, EI_c)

                grad_EI_c = cpp_GP.get_1D_analytic_grad_EI(points_to_sample[0])
                grad_EI_p = python_GP.get_1D_analytic_grad_EI(points_to_sample[0])
                for i in range(len(domain)):
                    self._assert_relative_diff_lte_tol(grad_EI_p[i], grad_EI_c[i])

class GetCholeskyDecompAndGradTest(GaussianProcessNumericalAnalysisTestCase):
    """Tests the python vs cpp implementations of .get_mean_and_var_of_points
    """
    tol = 1e-10 # TODO eliu look into this ticket #43006
    rel_tol = 1e-16

    def _assert_var_and_grad_var_equal(self, var_one, grad_var_one, var_two, grad_var_two):
        for point_on, var_one_row in enumerate(var_one):
            var_two_row = var_two[point_on][:]
            for i in range(len(var_one_row)):
                self._assert_relative_diff_lte_tol(var_one_row[i], var_two_row[i])
                for d in range(len(grad_var_one[point_on][i])):
                    self._assert_relative_diff_lte_tol(grad_var_one[point_on][i][d], grad_var_two[point_on][i][d])

    def test_random_process_returns_same_chol_var_and_grad_chol_var(self):
        """Test that the python and cpp return similar values for
        the cholesky decomp of the variance and the associated gradient
        for random points drawn from a random process
        """
        test_cases = [ # [num_points_to_test, length_scale, [domain]]
                [4, 4, 0.1, [[-5.0, 15.0]]],
                [10, 13, 0.1, [[-10.0, 10.0], [-10.0, 10.0], [-10.0, 10.0]]],
                [9, 11, 0.5, [[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]]],
                [5, 10, 0.01, [[50.0, 55.0], [-80.1, -76.4], [-15005.5, -15000.0]]],
                ]

        for [num_points_to_test, num_points_in_sample, length_scale, domain] in test_cases:
            covariance_of_process = self._make_default_covariance_of_process(length=[length_scale])
            cpp_GP, python_GP = self._make_random_processes_from_latin_hypercube(
                    domain,
                    num_points_in_sample,
                    covariance_of_process=covariance_of_process,
                    default_sample_variance=0.0
                    )

            # select 4 points to perturb from our stencil as points to sample
            points_to_sample = []
            points_sampled = [p.point for p in cpp_GP.points_sampled]
            for i, point_on in enumerate(random.sample(range(num_points_in_sample), num_points_to_test)):
                points_to_sample.append(
                        points_sampled[point_on] + numpy.random.uniform(
                            -1.0 * length_scale, # min val
                            length_scale, # max val
                            size=len(domain)
                            )
                        )
            points_to_sample = numpy.array(points_to_sample)

            # get chol_var and grad_chol_var via cpp and python for each potential point_to_sample
            for i in range(len(points_to_sample)):
                chol_var_c, grad_chol_var_c = cpp_GP.cholesky_decomp_and_grad(points_to_sample, var_of_grad=i)
                chol_var_p, grad_chol_var_p = python_GP.cholesky_decomp_and_grad(points_to_sample, var_of_grad=i)
                self._assert_var_and_grad_var_equal(chol_var_p, grad_chol_var_p, chol_var_c, grad_chol_var_c)

class GetMeanAndVarOfPointsTest(GaussianProcessNumericalAnalysisTestCase):
    """Tests the python vs cpp implementations of .get_mean_and_var_of_points
    """
    tol = 1e-11 # TODO eliu look into this ticket #43006

    def _assert_mean_and_var_equal(self, mean_one, var_one, mean_two, var_two):
        for point_on, var_one_row in enumerate(var_one):
            var_two_row = var_two[point_on][:]
            self._assert_relative_diff_lte_tol(mean_one[point_on], mean_two[point_on])
            for i in range(len(var_one_row)):
                self._assert_relative_diff_lte_tol(var_one_row[i], var_two_row[i])

    def test_random_process_returns_same_mu_and_var(self):
        """Test that the python and cpp return similar values for
        the mean and variance of a process
        for random points drawn from a random process
        """
        test_cases = [ # [num_points_to_test, length_scale, [domain]]
                [10, 0.1, [[-5.0, 15.0]]],
                [10, 0.1, [[-10.0, 10.0], [-10.0, 10.0], [-10.0, 10.0]]],
                [20, 0.5, [[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]]],
                [5, 0.01, [[50.0, 51.0], [-80.1, -76.4], [-15000.5, -15000.0]]],
                ]

        for [num_points_in_sample, length_scale, domain] in test_cases:
            covariance_of_process = self._make_default_covariance_of_process(length=[length_scale])
            cpp_GP, python_GP = self._make_random_processes_from_latin_hypercube(
                    domain,
                    num_points_in_sample,
                    covariance_of_process=covariance_of_process,
                    default_sample_variance=0.0
                    )

            points_to_sample = []
            points_sampled = [p.point for p in cpp_GP.points_sampled]

            # Sample stencil
            for i in range(num_points_in_sample):
                points_to_sample.append(points_sampled[i] + numpy.random.uniform(
                        -1.0 * length_scale, # min val
                        length_scale, # max val
                        size=len(domain)
                        ))

            # get mean and variance via cpp and python
            mu_c, var_c = cpp_GP.get_mean_and_var_of_points(points_to_sample)
            mu_p, var_p = python_GP.get_mean_and_var_of_points(points_to_sample)

            self._assert_mean_and_var_equal(mu_p, var_p, mu_c, var_c)

class GetGradMuTest(GaussianProcessNumericalAnalysisTestCase):

    def test_random_process_returns_same_grad_mu(self):
        """Test that the python and cpp return similar values for
        the gradient of the mean
        for random points drawn from a random process
        """
        test_cases = [ # [num_points_to_test, length_scale, [domain]]
                [10, 0.1, [[-5.0, 15.0]]],
                [10, 0.1, [[-10.0, 10.0], [-10.0, 10.0], [-10.0, 10.0]]],
                [20, 0.5, [[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]]],
                [5, 0.01, [[50.0, 51.0], [-80.1, -76.4], [-15000.5, -15000.0]]],
                ]

        for [num_points_in_sample, length_scale, domain] in test_cases:
            covariance_of_process = self._make_default_covariance_of_process(length=[length_scale])
            cpp_GP, python_GP = self._make_random_processes_from_latin_hypercube(
                    domain,
                    num_points_in_sample,
                    covariance_of_process=covariance_of_process,
                    default_sample_variance=0.0
                    )

            points_to_sample = []
            points_sampled = [p.point for p in cpp_GP.points_sampled]

            # Sample stencil
            for i in range(num_points_in_sample):
                points_to_sample.append(points_sampled[i] + numpy.random.uniform(
                        -1.0 * length_scale, # min val
                        length_scale, # max val
                        size=len(domain)
                        ))

            # get grad_mu via cpp and python
            grad_mu_c = cpp_GP.get_grad_mu(points_to_sample)
            grad_mu_p = python_GP.get_grad_mu(points_to_sample)

            for point_on, point_grad_mu_p in enumerate(grad_mu_p):
                for i in range(len(domain)):
                    py_ans = point_grad_mu_p[i]
                    cpp_ans = grad_mu_c[point_on][i]
                    self._assert_relative_diff_lte_tol(py_ans, cpp_ans)

class GetMultistartBestTest(OLDGaussianProcessTestCase):

    def test_one_dimensional_get_multistart_best_from_prior_within_domain(self):
        """Test that get_multistart_best returns a point within the domain
        for various 1D functions drawn from the prior over different domains
        """
        domains_to_test = [[-10.0, 10.0], [-1.0, 1.0], [50.0, 51.0]]

        for domain_min, domain_max in domains_to_test:
            domain = [[domain_min, domain_max]]
            GP = self._make_default_gaussian_process(
                    domain=domain,
                    default_sample_variance=0.0
                    )

            # The edges and the middle
            # shift stencil off of domain boundaries to prevent singular matrices
            stencil_points_to_sample = [
                    numpy.array([domain_min + 1.e-4]),
                    numpy.array([(domain_max + domain_min)/2]),
                    numpy.array([domain_max - 1.0e-4]),
                    ]

            # Sample stencil
            for point in stencil_points_to_sample:
                point_val = GP.sample_from_process(point)
                sample_point = SamplePoint(point, point_val)
                GP.add_sample_point(sample_point)

            next_step = GP.get_multistart_best()[0] # only one point is returned

            # The next step must be within the domain
            T.assert_gte(next_step, domain_min)
            T.assert_lte(next_step, domain_max)

    def test_two_dimensional_get_multistart_best_from_prior_within_domain(self):
        """Test that get_multistart_best returns a point within the domain
        for various 2D functions drawn from the prior over different domains
        """
        numpy.random.seed(314)
        domains_to_test = [[[-10.0, 10.0], [-10.0, 10.0]], [[-1.0, 1.0], [-1.0, 1.0]], [[50.0, 51.0], [-80.1, -76.4]]]

        for domain in domains_to_test:
            GP = self._make_default_gaussian_process(
                    domain=domain,
                    default_sample_variance=0.0
                    )

            # The edges and the middle
            x1_min = domain[0][0]
            x1_max = domain[0][1]
            x1_mid = (x1_min + x1_max)/2.0
            x2_min = domain[1][0]
            x2_max = domain[1][1]
            x2_mid = (x2_min + x2_max)/2.0
            stencil_points_to_sample = [
                    numpy.array([x1_min, x2_min]), # lower left
                    numpy.array([x1_min, x2_mid]), # mid left
                    numpy.array([x1_min, x2_max]), # top left
                    numpy.array([x1_mid, x2_min]), # mid middle
                    numpy.array([x1_mid, x2_mid]), # center
                    numpy.array([x1_mid, x2_max]), # mid top
                    numpy.array([x1_max, x2_min]), # lower right
                    numpy.array([x1_max, x2_mid]), # mid right
                    numpy.array([x1_max, x2_max]), # top right
                    ]

            # Sample stencil
            for point in stencil_points_to_sample:
                point_val = GP.sample_from_process(point)
                sample_point = SamplePoint(point, point_val)
                GP.add_sample_point(sample_point)

            next_step = GP.get_multistart_best() # only one point is returned

            # The next step must be within the domain
            T.assert_gte(next_step[0], x1_min)
            T.assert_lte(next_step[0], x1_max)
            T.assert_gte(next_step[1], x2_min)
            T.assert_lte(next_step[1], x2_max)

    def _three_dimensional_get_multistart_best_from_prior_within_domain(self, max_number_of_threads=1):
        """Test that get_multistart_best returns a point within the domain
        for various 2D functions drawn from the prior over different domains

        Can run with more than 1 thread
        """
        numpy.random.seed(314)
        domains_to_test = [
                [[-10.0, 10.0], [-10.0, 10.0], [-10.0, 10.0]],
                [[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]],
                [[50.0, 51.0], [-80.1, -76.4], [-15000.5, -15000.0]],
                ]

        for domain in domains_to_test:
            GP = self._make_default_gaussian_process(
                    domain=domain,
                    default_sample_variance=0.0,
                    max_number_of_threads=max_number_of_threads,
                    )

            # select a stencil of num_points_to_test random latin hypercube points
            stencil_points_to_sample = moe.optimal_learning.python.lib.math.get_latin_hypercube_points(10, domain)

            # Sample stencil
            for point in stencil_points_to_sample:
                point_val = GP.sample_from_process(point)
                sample_point = SamplePoint(point, point_val)
                GP.add_sample_point(sample_point)

            next_step = GP.get_multistart_best() # only one point is returned

            # The next step must be within the domain
            for i, sub_domain in enumerate(domain):
                T.assert_gte(next_step[i], sub_domain[0])
                T.assert_lte(next_step[i], sub_domain[1])

    def test_three_dimensional_get_multistart_best_from_prior_within_domain(self):
        self._three_dimensional_get_multistart_best_from_prior_within_domain(max_number_of_threads=1)

    def test_three_dimensional_get_multistart_best_from_prior_within_domain_multithreaded(self):
        self._three_dimensional_get_multistart_best_from_prior_within_domain(max_number_of_threads=3)


if __name__ == "__main__":
    T.run()
