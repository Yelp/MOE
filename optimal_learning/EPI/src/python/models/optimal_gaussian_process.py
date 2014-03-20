# -*- coding: utf-8 -*-
# Copyright (C) 2012 Scott Clark. All rights reserved.

import numpy
import logging
import scipy.stats # for stats dists

import optimal_learning.EPI.src.python.lib.mvncdf # multivariate normal cdf
import optimal_learning.EPI.src.python.models.covariance_of_process
import optimal_learning.EPI.src.python.lib.math
from optimal_learning.EPI.src.python.models.gaussian_process import GaussianProcess


logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class OptimalGaussianProcess(GaussianProcess):
    """docstring!
    """

    def __init__(self, max_number_of_threads=1, *args, **kwargs):
        """max_number_of_threads passed through to subclasses (not used in this class)
        """
        super(OptimalGaussianProcess, self).__init__(*args, **kwargs)

    def get_1D_analytic_grad_EI(self, point_to_sample):
        """Return the analytic derivative of the EI for a single sample point

        see GiLeCa08

        :Returns: double
        """
        mu_star, var_star = self.get_mean_and_var_of_points([point_to_sample])

        cholesky_decomp, grad_cholesky_decomp = self.cholesky_decomp_and_grad([point_to_sample])

        grad_mu = self.get_grad_mu([point_to_sample])

        sigma = numpy.sqrt(abs(var_star))

        mu_diff = (self.best_so_far - mu_star)

        C = (self.best_so_far - mu_star)/sigma
        pdf_C = scipy.stats.norm.pdf((self.best_so_far - mu_star)/sigma)
        cdf_C = scipy.stats.norm.cdf((self.best_so_far - mu_star)/sigma)

        d_C = (-sigma*grad_mu - grad_cholesky_decomp*(self.best_so_far - mu_star))/(sigma**2)

        d_A = -grad_mu*cdf_C + mu_diff*pdf_C*d_C

        d_B = grad_cholesky_decomp*pdf_C + sigma*(-C)*pdf_C*d_C

        return d_A + d_B

    def get_1D_analytic_expected_improvement(self, point_to_sample):
        """Calculate the expected improvement for a single point exactly using normal pdf/cdf

        :Returns: double
        """
        mu_star, var_star = self.get_mean_and_var_of_points([point_to_sample])

        if self.domain and optimal_learning.EPI.src.python.lib.math.not_in_domain(point_to_sample, self.domain):
            return 0.0

        return numpy.max([0.0,(self.best_so_far - mu_star)*scipy.stats.norm.cdf((self.best_so_far - mu_star)/numpy.sqrt(abs(var_star))) + numpy.sqrt(abs(var_star))*scipy.stats.norm.pdf((self.best_so_far - mu_star)/numpy.sqrt(abs(var_star)))])

    def get_2D_analytic_expected_improvement(self, point_one, point_two):
        """Calculate the expected improvement for two points exactly

        following GiLeCa08, with some errors corrected

        :Returns: double
        """

        if self.domain:
            if optimal_learning.EPI.src.python.lib.math.not_in_domain(point_one, self.domain) or optimal_learning.EPI.src.python.lib.math.not_in_domain(point_two, self.domain):
                return 0.0

        eps = 0.00001

        mu_star, var_star = self.get_mean_and_var_of_points([point_one, point_two])

        mu_one = mu_star[0]
        mu_two = mu_star[1]
        sigma_one = numpy.sqrt(abs(var_star[0][0]))
        sigma_two = numpy.sqrt(abs(var_star[1][1]))
        sigma_cross = var_star[1][0]

        if sigma_one < eps and sigma_two < eps:
            return numpy.max([0.0, self.best_so_far - numpy.min([mu_one, mu_two])])
        elif sigma_one < eps:
            return numpy.min([self.get_1D_analytic_expected_improvement(point_two), self.best_so_far - mu_one])
        elif sigma_two < eps:
            return numpy.min([self.get_1D_analytic_expected_improvement(point_one), self.best_so_far - mu_two])

        def B_func(m1, m2, s1, s2, c12, best):
            return (m1 - best)*D_func(m1, m2, s1, s2, c12, best) + s1*E_func(m1, m2, s1, s2, c12, best)

        def E_func(m1, m2, s1, s2, c12, best):
            p12 = c12/(s1*s2)
            beta1 = (m1 - m2)/(s2*numpy.sqrt(abs(1.0 - p12*p12)))
            alpha1 = (s1 - p12*s2)/(s2*numpy.sqrt(abs(1.0 - p12*p12)))
            gamma1 = (best - m1)/s1

            # from phase 3 (different from the proposition!)
            return alpha1*scipy.stats.norm.pdf(numpy.abs(beta1)/numpy.sqrt(abs(1.0 + alpha1*alpha1)))/numpy.sqrt(abs(1 + alpha1*alpha1))*scipy.stats.norm.cdf(numpy.sqrt(abs(1.0 + alpha1*alpha1))*(gamma1 + (alpha1*beta1)/(1.0 + alpha1*alpha1))) - scipy.stats.norm.pdf(gamma1)*scipy.stats.norm.cdf(alpha1*gamma1 + beta1)

        def D_func(m1, m2, s1, s2, c12, best):
            Gamma = numpy.array([[s1*s1, c12 - s1*s1],[c12 - s1*s1, s2*s2 + s1*s1 - 2.0*c12]])

            return optimal_learning.EPI.src.python.lib.mvncdf.mvnormcdf(numpy.array([-numpy.inf,-numpy.inf]), numpy.array([best - m1, m1 - m2]), numpy.array([0.0,0.0]), Gamma)

        return self.get_1D_analytic_expected_improvement(point_one) + self.get_1D_analytic_expected_improvement(point_two) + B_func(mu_one, mu_two, sigma_one, sigma_two, sigma_cross, self.best_so_far) + B_func(mu_two, mu_one, sigma_two, sigma_one, sigma_cross, self.best_so_far)

    def _get_expected_improvement_via_MC(self, points_to_sample, iterations=1000):
        # for three or more points find the EI via Monte Carlo
        mu_star, var_star = self.get_mean_and_var_of_points(points_to_sample)
        cholesky_decomp, grad_cholesky_decomp = self.cholesky_decomp_and_grad(points_to_sample)

        normals = numpy.random.normal(size = (iterations,len(points_to_sample)))

        aggregate = 0.0
        for it in range(iterations):
            improvement_this_step = self.best_so_far - numpy.min(mu_star + numpy.dot(cholesky_decomp, normals[it][:].T))
            if improvement_this_step > 0:
                aggregate += improvement_this_step
        return aggregate/float(iterations)

    def get_expected_improvement(self, points_to_sample, iterations=1000):
        """Calculate the expected improvement at *points_to_sample* over a certain number of MC *iterations* in a *domain*

        :Returns: double
        """

        # if we resrict the function to a domain there can be no improvement outside of it
        if self.domain:
            for point in points_to_sample:
                if optimal_learning.EPI.src.python.lib.math.not_in_domain(point, self.domain):
                    return 0.0

        # if we are sampling one or two points find the analytic result
        if len(points_to_sample) == 1:
            return numpy.max([0.0, self.get_1D_analytic_expected_improvement(points_to_sample[0])])
        elif len(points_to_sample) == 2:
            return numpy.max([0.0, self.get_2D_analytic_expected_improvement(points_to_sample[0], points_to_sample[1])])
        else:
            return self._get_expected_improvement_via_MC(self, points_to_sample, iterations=1000)

    def get_expected_grad_EI(self, current_point, points_being_sampled, iterations=100):
        """get the expected EI grad at [current_point, points_being_sampled] wrt current_point"""

        # TODO if outside the domain, pull it towards the center

        union_of_points = [current_point]
        union_of_points.extend(points_being_sampled)

        mu, var_star = self.get_mean_and_var_of_points(union_of_points)
        cholesky_decomp, grad_chol_decomp = self.cholesky_decomp_and_grad(union_of_points)
        grad_mu = self.get_grad_mu(union_of_points)

        normals = numpy.random.normal(size=(iterations, len(union_of_points)))

        aggregate_dx = 0.0

        for it in range(iterations):
            from_var = mu + numpy.dot(cholesky_decomp, normals[it][:].T)
            improvement = self.best_so_far - from_var
            if len(improvement) > 1:
                for i in range(len(improvement)):
                    if improvement[i] >= numpy.max(improvement) and improvement[i] > 0.0:
                        if i == 0:
                            aggregate_dx -= grad_mu[0]
                        aggregate_dx -= optimal_learning.EPI.src.python.lib.math.matrix_vector_multiply(grad_chol_decomp, normals[it][:].T)[i]
            else:
                if improvement > 0.0:
                    aggregate_dx += -grad_mu[0] - grad_chol_decomp[0][0]*normals[it][0]

        return aggregate_dx/float(iterations)

    def get_next_step(self, starting_point, points_being_sampled, domain=None, gamma=0.8, iterations=1000, max_N=400):
        """get the next step using polyak-ruppert"""
        x_hat = starting_point

        x_path = [starting_point]

        n = 1
        while n < max_N:
            alpha_n = 0.01*n**-gamma

            gx_n = self.get_expected_grad_EI(x_path[-1], points_being_sampled, iterations=10)

            x_path.append(x_path[-1] + alpha_n*gx_n)

            if n%(max_N/20) == 0:
                x_hat = numpy.mean(x_path[1:], axis=0)

            n += 1

        union_of_points = [x_hat]
        union_of_points.extend(points_being_sampled)
        EI = self.get_expected_improvement(union_of_points)

        return x_path, x_hat, EI

    def get_next_step_multistart(self, starting_points, points_being_sampled, gamma=0.9, iterations=1000, max_N=100):
        """get next step using multistart polyak-ruppert"""
        polyak_ruppert_paths = []
        best_improvement = -numpy.inf
        best_point = None
        for starting_point in starting_points:
            x_path, x_hat, EI = self.get_next_step(starting_point, points_being_sampled, gamma=gamma, iterations=iterations, max_N=max_N)
            if not self.domain or not optimal_learning.EPI.src.python.lib.math.not_in_domain(x_hat, self.domain):
                polyak_ruppert_paths.append([x_path, x_hat, EI])
                if polyak_ruppert_paths[-1][2] > best_improvement:
                    best_improvement = polyak_ruppert_paths[-1][2]
                    best_point = polyak_ruppert_paths[-1][1]

        return best_point, best_improvement, polyak_ruppert_paths

    def get_multistart_best(self, random_restarts=5, points_being_sampled=[]):
        best_improvement = -numpy.inf
        best_next_step = optimal_learning.EPI.src.python.lib.math.make_rand_point(self.domain)
        for _ in range(random_restarts):
            path, next_step, improvement = self.get_next_step(optimal_learning.EPI.src.python.lib.math.make_rand_point(self.domain), points_being_sampled)
            #path, next_step, improvement = self.get_next_step(numpy.array([-2.141592, 11.274999]), [numpy.array([1.0,0.0])], [0,1])
            if improvement > best_improvement and not optimal_learning.EPI.src.python.lib.math.not_in_domain(next_step, self.domain):
                best_improvement = improvement
                best_next_step = next_step
        return best_next_step
