# -*- coding: utf-8 -*-

import numpy # for sci comp

import moe.optimal_learning.EPI.src.python.models.sample_point
import moe.optimal_learning.EPI.src.python.models.covariance_of_process
import moe.optimal_learning.EPI.src.python.lib.math

import logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class GaussianProcess(object):
    """A gaussian process of a sample function

    :Attributes:
        - cop: The *covariance_of_process* a CovarianceOfProcess object
               default: *cop=CovarianceOfProcess()*
        - domain: the domain to optimize over,
            if no domain is passed in it will search over all real numbers
        - initial_best_so_far: initial value of best_so_far: numpy.min(values_of_samples) if samples exist, o/w max double
        - default_sample_variance: noise variance to use if no noise is explicitly specified
        - max_number_of_threads: max number of threads to use; only applicable to LinkedCpp subclass

    :Internal state variables:
        - domain: the domain of the function
        - cholesky_L: see paper
        - cholesky_alpha: see paper
        - points_sampled: The SamplePoint(s) sampled thus far
        - values_of_samples: The values of points sampled thus far in a list
        - best_so_far: numpy.min(values_of_samples) or *initial_best_so_far* if no points sampled
    """
    def __init__(self, domain=None, covariance_of_process=None, initial_best_so_far=numpy.finfo('d').max, default_sample_variance=0.0, max_number_of_threads=1):
        """Inits a GaussianProcess"""
        # the domain of the function [[x1_min, x2_max], ..., [xn_min, xn_max]]
        self.domain = domain

        # information about the points sampled and their values and noises
        self.points_sampled = []
        self.values_of_samples = []
        self.sample_variance_of_samples = []
        self.default_sample_variance = default_sample_variance
        self.best_so_far = initial_best_so_far # lowest value found thus far

        if not covariance_of_process:
            self.cop = moe.optimal_learning.EPI.src.python.models.covariance_of_process.CovarianceOfProcess()
        else:
            #TODO type check
            self.cop = covariance_of_process

        # internal state of several important matricies
        self.cholesky_L = None
        self.cholesky_alpha = None
        self.covariance_matrix = self.build_covariance_matrix()
        self.K_inv = None

    def __str__(self):
        data = 'Points Sampled:\n'
        for sample in self.points_sampled:
            data += "%s\n" % sample.__str__()
        data += "cholesky_L:\n"
        data += str(self.cholesky_L)
        data += "\ncholesky_alpha:\n"
        data += str(self.cholesky_alpha)
        return data

    def update_cop(self):
        self.cop.update_hyper_parameters(self)
        self.update_state()

    def update_state(self):
        """Updates the state variables"""
        self.covariance_matrix = self.build_covariance_matrix()
        self.update_chol_parts()
        self.K_inv = numpy.linalg.inv(self.covariance_matrix)

    def add_sample_point(self, sample_point, sample_variance=None):
        """Add a SamplePoint *sample_point* to the GPP"""
        if not sample_variance:
            sample_variance = self.default_sample_variance

        self.points_sampled.append(sample_point)
        self.sample_variance_of_samples.append(sample_variance)
        self.values_of_samples.append(sample_point.value)

        # update best_so_far
        if sample_point.value < self.best_so_far:
            self.best_so_far = sample_point.value

        self.update_state()

    ###################################
    ### MEAN and VARIANCE UTILITIES ###
    ###################################

    def build_covariance_matrix(self, grad_function=None, kwargs={}):
        """build covariance matrix for inputs, see paper

        :Returns: 2-D numpy array
        """
        if grad_function:
            cov_func = grad_function
        else:
            cov_func = self.cop.cov
        K = numpy.zeros((len(self.points_sampled), len(self.points_sampled)))
        for i, sample_one in enumerate(self.points_sampled):
            for j, sample_two in enumerate(self.points_sampled):
                K[i][j] = cov_func(sample_one.point, sample_two.point, kwargs=kwargs)
        if not grad_function:
            K = K + numpy.diag(self.sample_variance_of_samples)
        return K

    def build_mix_covariance_matrix(self, points_to_sample):
        """Covariance matrix of points to sample vs points sampled already, see paper

        :Returns: 2-D numpy array
        """
        K_star = numpy.zeros((len(points_to_sample),len(self.points_sampled)))
        for i, point_to_sample in enumerate(points_to_sample):
            for j, sampled in enumerate(self.points_sampled):
                K_star[i][j] = self.cop.cov(point_to_sample, sampled.point)
        return K_star

    def build_grad_sample_covariance_matrix(self, points_to_sample):
        """Grad covariance of points to sample, see paper

        :Returns: 3-D list
        """
        grad_K_star_star = []
        for i, point_one in enumerate(points_to_sample):
            grad_K_star_star.append([])
            for j, point_two in enumerate(points_to_sample):
                grad_K_star_star[i].append(self.cop.grad_cov(point_one, point_two))
        return grad_K_star_star

    def build_sample_covariance_matrix(self, points_to_sample):
        """Covariance of points to sample, see paper

        :Returns: 2-D numpy array
        """
        K_star_star = numpy.zeros((len(points_to_sample), len(points_to_sample)))
        for i, point_one in enumerate(points_to_sample):
            for j, point_two in enumerate(points_to_sample):
                K_star_star[i][j] = self.cop.cov(point_one, point_two)
        return K_star_star

    def update_chol_parts(self, include_sample_variance=True):
        """Get the cholesky parts of the GP

        See page 19 of *Gaussian Processes for Machine Learning* by Rasmussen and Williams.
        """
        # build covariance matrix for inputs
        K = self.covariance_matrix

        # build chol decomp
        if include_sample_variance:
            sample_variance_matrix = numpy.diag(
                    [sample_var for sample_var in self.sample_variance_of_samples]
                    )
            L = moe.optimal_learning.EPI.src.python.lib.math.cholesky_decomp(K + sample_variance_matrix)
        else:
            L = moe.optimal_learning.EPI.src.python.lib.math.cholesky_decomp(K)

        # solve for (cholesky) alpha, K^-1 * y^T, see RW pg19
        y = numpy.zeros(len(self.points_sampled))
        for i, sample in enumerate(self.points_sampled):
            y[i] = sample.value
        alpha = numpy.linalg.solve(L.T,numpy.linalg.solve(L,y))

        self.cholesky_L = L
        self.cholesky_alpha = alpha

    def sample_from_process(self, point_to_sample, random_normal=None, sample_variance=None, sample_variance_normal=None):
        """Return a sample value from the process at a given point

        Returns the mean at that point plus the variance multipled by a random normal variable (or one provided)
        """
        if sample_variance is None:
            sample_variance = self.default_sample_variance

        # get the mean and variance at the point to sample
        mu, var = self.get_mean_and_var_of_points([point_to_sample])

        # draw the normals if needed
        if not random_normal:
            random_normal = numpy.random.normal()
        if not sample_variance_normal:
            sample_variance_normal = numpy.random.normal()

        return mu[0] + numpy.sqrt(var[0][0])*random_normal + numpy.sqrt(sample_variance)*sample_variance_normal

    def _neg_log_marginal_likelihood(self):
        # Eq 5.8 pg 113 of RW
        if not self.values_of_samples:
            raise(ValueError, "There is no marginal if no values have been sampled")
        K = self.build_covariance_matrix()
        y = numpy.array(self.values_of_samples)
        n = len(y)
        K_inv_y = numpy.linalg.solve(K, y)
        log_det_K = 0
        L = moe.optimal_learning.EPI.src.python.lib.math.cholesky_decomp(K)
        for i, mat_row in enumerate(L):
            log_det_K += 2.0 * numpy.log( mat_row[i] )

        return 0.5 * numpy.dot(y.T, K_inv_y) + \
                0.5 * log_det_K + \
                0.5 * n*numpy.log(2*numpy.pi)

    def get_log_marginal_likelihood(self):
        return -self._neg_log_marginal_likelihood()

    def get_mean_and_var_of_points(self, points_to_sample):
        """Given a set of points to sample compute their mean and variance from the current state of the GPP

        follows algorithm 2.1 (pg 19) of *Gaussian Processes for Machine Learning* by Rasmussen and Williams

        :Returns: 1-D numpy array, 2-D numpy array
        """
        # variance for points to sample
        K_star_star = self.build_sample_covariance_matrix(points_to_sample)

        # check if any points have been sampled yet and return with prior if not
        if not self.points_sampled:
            mu_star = numpy.zeros(len(points_to_sample))
            return mu_star, K_star_star

        # build covariance matrix for test data
        K_star = self.build_mix_covariance_matrix(points_to_sample)

        # calculate means
        mu_star = numpy.dot(K_star,self.cholesky_alpha)

        # calculate variance matrix
        v = numpy.linalg.solve(self.cholesky_L, K_star.T)
        var_star = K_star_star - numpy.dot(v.T,v)

        return mu_star, var_star

    def get_grad_mu(self, points_to_sample):
        """Returns the gradient of the mean of the GPP at *points_to_sample*

        :Returns: 1-D numpy array
        """
        # build grad_K_star
        grad_K_star = moe.optimal_learning.EPI.src.python.lib.math.make_empty_2D_list(len(points_to_sample),len(self.points_sampled))
        for i, to_sample in enumerate(points_to_sample):
            for j, sampled in enumerate(self.points_sampled):
                grad_K_star[i][j] = numpy.array(self.cop.grad_cov(to_sample, sampled.point))

        grad_mu = moe.optimal_learning.EPI.src.python.lib.math.matrix_vector_multiply(
                numpy.array(grad_K_star),
                numpy.linalg.solve(self.covariance_matrix, numpy.array(self.values_of_samples).T)
                )

        return grad_mu

    def build_grad_K_star(self, points_to_sample):
        # build grad_K_star
        grad_K_star = moe.optimal_learning.EPI.src.python.lib.math.make_empty_2D_list(len(points_to_sample),len(self.points_sampled))
        for l, to_sample in enumerate(points_to_sample):
            for m, sampled in enumerate(self.points_sampled):
                grad_K_star[l][m] = numpy.array(self.cop.grad_cov(to_sample, sampled.point))
        return numpy.array(grad_K_star)

    def get_grad_cov_component(self, points_to_sample, i, j, var):
        """Get a specific component (i,j) of the grad cholesky decomposition of the variance
        w.r.t the variable (in points_to_sample) var

        This is expensive to do for all combos O(2*L^2*N^2) and should be offloaded to C (length = size points_to_sample, N = size_of_sampled)

        see S.P.Smith; *Differentiation of the Cholesky Algorithm*; 1995

        :Returns: 2-D numpy array, n-D numpy array
        """

        print "WHAT AM I DOING???"
        raise(NotImplementedError)
        point_one = points_to_sample[i]
        point_two = points_to_sample[j]

        if var == i:
            comp = self.cop.grad_cov(points_to_sample[i], points_to_sample[j])
        elif var == j:
            comp = self.cop.grad_cov(points_to_sample[j], points_to_sample[i])
        else:
            return 0.0

        grad_K_star = numpy.array(self.build_grad_K_star(points_to_sample))[:,:,0]

        K_star = numpy.array(self.build_mix_covariance_matrix(points_to_sample))

        o_comp = numpy.dot(grad_K_star, numpy.dot(self.K_inv, K_star.T))[i][j]

        comp2 = 0

        # TODO This can be sped up considerably...
        if var == i and var == j:
            for idx_one, sampled_one in enumerate(self.points_sampled):
                for idx_two, sampled_two in enumerate(self.points_sampled):
                    comp -= 2*self.K_inv[idx_two][idx_one]*self.cop.cov(point_one, sampled_one.point)*self.cop.grad_cov(point_one, sampled_two.point)
            return comp
        elif var == i:
            for idx_one, sampled_one in enumerate(self.points_sampled):
                for idx_two, sampled_two in enumerate(self.points_sampled):
                    comp -= self.K_inv[idx_two][idx_one]*self.cop.cov(point_two, sampled_one.point)*self.cop.grad_cov(point_one, sampled_two.point)
                    comp2 -= self.K_inv[idx_two][idx_one]*self.cop.cov(point_two, sampled_one.point)*self.cop.grad_cov(point_one, sampled_two.point)
            logging.debug(comp2[0], -o_comp)

        elif var == j:
            for idx_one, sampled_one in enumerate(self.points_sampled):
                for idx_two, sampled_two in enumerate(self.points_sampled):
                    comp -= self.K_inv[idx_two][idx_one]*self.cop.cov(point_one, sampled_two.point)*self.cop.grad_cov(point_two, sampled_one.point)



        return comp

    def cholesky_decomp_and_grad(self, points_to_sample, var_of_grad=0, eps=0.000001):
        """Get the cholesky decomposition of the variance and its gradient

        see S.P.Smith; *Differentiation of the Cholesky Algorithm*; 1995

        :Returns: 2-D numpy array, n-D numpy array
        """

        mu_star, var_star = self.get_mean_and_var_of_points(points_to_sample)

        grad_cholesky_decomp = moe.optimal_learning.EPI.src.python.lib.math.make_empty_2D_list(len(points_to_sample), len(points_to_sample))

        cholesky_decomp = var_star.copy() # Just to start!

        # Step 1 of Appendix 2
        for i, point_one in enumerate(points_to_sample):
            for j, point_two in enumerate(points_to_sample):
                if var_of_grad == i and var_of_grad == j:
                    grad_cholesky_decomp[i][j] = self.cop.grad_cov(point_one, point_two)
                    for idx_one, sampled_one in enumerate(self.points_sampled):
                        for idx_two, sampled_two in enumerate(self.points_sampled):
                            grad_cholesky_decomp[i][j] -= 2*self.K_inv[idx_two][idx_one]*self.cop.cov(point_one, sampled_one.point)*self.cop.grad_cov(point_one, sampled_two.point)
                elif var_of_grad == i:
                    grad_cholesky_decomp[i][j] = self.cop.grad_cov(point_one, point_two)
                    for idx_one, sampled_one in enumerate(self.points_sampled):
                        for idx_two, sampled_two in enumerate(self.points_sampled):
                            grad_cholesky_decomp[i][j] -= self.K_inv[idx_two][idx_one]*self.cop.cov(point_two, sampled_one.point)*self.cop.grad_cov(point_one, sampled_two.point)
                elif var_of_grad == j:
                    grad_cholesky_decomp[i][j] = self.cop.grad_cov(point_two, point_one)
                    for idx_one, sampled_one in enumerate(self.points_sampled):
                        for idx_two, sampled_two in enumerate(self.points_sampled):
                            grad_cholesky_decomp[i][j] -= self.K_inv[idx_two][idx_one]*self.cop.cov(point_one, sampled_two.point)*self.cop.grad_cov(point_two, sampled_one.point)
                else:
                    grad_cholesky_decomp[i][j] = numpy.zeros(len(points_to_sample[0]))

        # TODO make more pythonic
        # zero out the upper half of the matrix
        for i in range(len(points_to_sample)):
            for j in range(len(points_to_sample)):
                if i < j:
                    cholesky_decomp[i][j] = 0.0
                    grad_cholesky_decomp[i][j] = numpy.zeros(len(points_to_sample[0]))

        # Step 2 of Appendix 2
        for k in range(len(points_to_sample)):
            if cholesky_decomp[k][k] > eps:
                cholesky_decomp[k][k] = numpy.sqrt(abs(cholesky_decomp[k][k]))
                grad_cholesky_decomp[k][k] = 0.5*grad_cholesky_decomp[k][k]/cholesky_decomp[k][k]
                for j in range(k+1, len(points_to_sample)):
                    cholesky_decomp[j][k] = cholesky_decomp[j][k]/cholesky_decomp[k][k]
                    grad_cholesky_decomp[j][k] = (grad_cholesky_decomp[j][k] - cholesky_decomp[j][k]*grad_cholesky_decomp[k][k])/cholesky_decomp[k][k]
                for j in range(k+1, len(points_to_sample)):
                    for i in range(j, len(points_to_sample)):
                        cholesky_decomp[i][j] = cholesky_decomp[i][j] - cholesky_decomp[i][k]*cholesky_decomp[j][k]
                        grad_cholesky_decomp[i][j] = grad_cholesky_decomp[i][j] - grad_cholesky_decomp[i][k]*cholesky_decomp[j][k] - cholesky_decomp[i][k]*grad_cholesky_decomp[j][k]

        return cholesky_decomp, grad_cholesky_decomp
