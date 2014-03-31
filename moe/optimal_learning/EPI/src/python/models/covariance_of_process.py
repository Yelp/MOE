# -*- coding: utf-8 -*-

import numpy
import logging

import moe.optimal_learning.EPI.src.python.lib.math
import moe.optimal_learning.EPI.src.python.models.sample_point
from scipy.optimize.optimize import fminbound
from scipy.optimize.optimize import fmin_bfgs


logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Constants
BFGS_LOO_CV_UPDATER = 'bfgs_leave_one_out_cross_validation_updater'
FMIN_MARGINAL_UPDATER = 'fmin_marginal_likelihood_updater'
BFGS_MARGINAL_UPDATER = 'bfgs_marginal_likelihood_updater'

class CovarianceOfProcess(object):
    """The covariance properties of a process

    :kwargs:
        - cov_func: The covariance function, a function of two input points
                    default: :math:`cov(x_1, x_2) = a*exp(-(1/2l)*|x_1 - x_2|^p)`
        - grad_cov_func: The gradient of the covariance function, a function of two input points.
                         The gradient is taken with respect to the first point.
        - hyperparameter_grad_table: a mapping from hyperparameters to gradients of covariance wrt them
        - length, alpha: Parameters of the default square exponential covariance function.
                        Not used if explicit *cov* and *grad_cov* functions are given.

    :state functions:
        - self.cov(x1, x2): The covariance function of the process
        - self.grad_cov(x1, x2): The gradient of the covariance function of the process

    :References:
        - www.gaussianprocess.org/gpml/chapters/RW.pdf (most notably chapter 5)
    """

    def __init__(self, cov_func=None, grad_cov_func=None, hyperparameters=[], hyperparameter_grad_table=[], hyperparameter_grad_table_kwargs=[]):
        """Inits the CovarianceOfProcess with either an square exponential cov function
        with given (or default parameters) OR with a user defined cov and grad_cov function
        """

        self.hyperparameters = hyperparameters
        self.is_default_covariance = False

        def default_cov(point_one, point_two, kwargs=None):
            """The square exponential covariance function

            alpha * exp( -1/2 * r L r^T )
            r = point_one - point_two
            L = diagonal matrix with length scales
            """

            def __str__(self):
                return "The square exponential covariance function eq 5.1 pg 106 RW GP book"

            alpha = self.hyperparameters[0]
            length = self.hyperparameters[1:]

            covariance = alpha * numpy.exp( -0.5 *
                    moe.optimal_learning.EPI.src.python.lib.math.vector_diag_vector_product(
                        numpy.array(point_one),
                        1.0 / (numpy.array(length) * numpy.array(length)),
                        numpy.array(point_two)
                        )
                    )

            return covariance

        def default_cov_grad_length_one_length_scale(point_one, point_two, length):
            """Return the gradient of the default covariance function wrt a single length scale
            """
            return moe.optimal_learning.EPI.src.python.lib.math.vector_diag_vector_product(
                    numpy.array(point_one),
                    1.0 / (numpy.array(length) * numpy.array(length) * numpy.array(length)),
                    numpy.array(point_two)
                    ) * default_cov(point_one, point_two)

        def default_cov_grad_length(point_one, point_two, kwargs={}):
            """Return the gradient of the default covariance function wrt length

            If we want to pick out a specific length component to diff against
            all of the other components will go to zero
            otherwise, we want to use all components
            """

            length = self.hyperparameters[1:]
            if len(length) == 1:
                return default_cov_grad_length_one_length_scale(point_one, point_two, length)

            # pull out the length component we are taking the grad wrt
            length_index_vector = numpy.zeros(len(point_one))
            length_index_vector[kwargs['length_index']] = 1.0
            diag_vec = [1.0/length[kwargs['length_index']]**3]

            return moe.optimal_learning.EPI.src.python.lib.math.vector_diag_vector_product(
                    length_index_vector * numpy.array(point_one),
                    diag_vec,
                    length_index_vector * numpy.array(point_two)
                    ) * default_cov(point_one, point_two)

        def default_cov_grad_alpha(point_one, point_two, kwargs={}):
            """Return the gradient of the default covariance function wrt alpha
            """
            alpha = self.hyperparameters[0]
            return default_cov(point_one, point_two) / alpha

        def grad_default_cov(point_one, point_two):
            """The gradient (wrt r = (point_one - point_two)) of the square exponential covariance function, see paper"""

            def __str__(self):
                return "The gradient (wrt r = (point_one - point_two)) of the square exponential covariance function"

            length = self.hyperparameters[1:]
            return numpy.array(
                        1.0 / (numpy.array(length) * numpy.array(length)) *
                        numpy.array(numpy.array(point_two) - numpy.array(point_one)).T *
                        default_cov(point_one, point_two)
                    )

        # A cov_func and grad_cov_func were not defined, use default (exp cov, see paper)
        if not cov_func and not grad_cov_func:
            self.cov = default_cov
            self.grad_cov = grad_default_cov
            # set the alpha hyperparameters
            self.hyperparameter_grad_table = [
                    default_cov_grad_alpha
                    ]
            self.hyperparameter_grad_table_kwargs = [{}]
            # set the length hyperparameters
            length = self.hyperparameters[1:]
            if len(length) == 1:
                self.hyperparameter_grad_table.append(default_cov_grad_length)
                self.hyperparameter_grad_table_kwargs.append({})
            else:
                for i, _ in enumerate(self.hyperparameters[1:]):
                    self.hyperparameter_grad_table.append(default_cov_grad_length)
                    self.hyperparameter_grad_table_kwargs.append({'length_index': i})

            self.is_default_covariance = True

        # The user defines both a cov and grad_cov function
        elif not not cov_func and not not grad_cov_func:
            self.cov = cov_func
            self.grad_cov = grad_cov_func
            self.hyperparameter_grad_table = hyperparameter_grad_table
            self.hyperparameter_grad_table_kwargs = hyperparameter_grad_table_kwargs

        else:
            raise(ValueError, "Need both cov_func and grad_cov_func or neither (not xor)")

    #######################################
    ### Hyperparameter Update Functions ###
    #######################################

    def _neg_log_marginal_likelihood_given_hyperparameters(self, hyperparameters):
        """log likelihood given both components
        """
        # Store current values
        old_hyperparameters = self.hyperparameters[:]
        self.hyperparameters = hyperparameters
        # Compute needed components
        likelihood = -self.GP.get_log_marginal_likelihood()
        # Put them back
        self.hyperparameters = old_hyperparameters
        return likelihood

    def _gradient_of_marginal_wrt_hyperparameters(self, hyperparameters):
        # Store current values
        old_hyperparameters = self.hyperparameters[:]
        self.hyperparameters = hyperparameters
        # Compute needed components
        the_gradient = numpy.zeros(len(self.hyperparameters))
        for i, grad_function in enumerate(self.hyperparameter_grad_table):
            the_gradient[i] = self._marginal_gradient_of_variable(
                    grad_function,
                    kwargs=self.hyperparameter_grad_table_kwargs[i]
                    )
        # Put them back
        self.hyperparameters = old_hyperparameters
        return the_gradient

    def _marginal_gradient_of_variable(self, hyperparameter_grad_func, kwargs={}):
        """follows pg114 of RW"""
        K = self.GP.build_covariance_matrix()
        y = numpy.array(self.GP.values_of_samples)
        grad_cov_hyper_K = self.GP.build_covariance_matrix(grad_function=hyperparameter_grad_func, kwargs=kwargs)

        K_inv_y = numpy.linalg.solve(K, y.T)
        grad_K_K_inv_y = numpy.dot(grad_cov_hyper_K, K_inv_y)
        first_term = 0.5 * numpy.dot(K_inv_y.T, grad_K_K_inv_y)

        K_inv_grad_K = numpy.linalg.solve(K, grad_cov_hyper_K)
        second_term = -0.5 * numpy.trace(K_inv_grad_K)

        grad_hyper_K = first_term + second_term
        return grad_hyper_K

    def _update_hyperparameters_via_marginal_fmin(self, lower_ratio=0.5, upper_ratio=2.0):
        """Update via scipy.optimize.optimize.fminbound, derivative free
        This expects len(self.hyperparameters) = 1

        References:
            * http://www.scipy.org/doc/api_docs/SciPy.optimize.optimize.html#fminbound
        """
        if len(self.hyperparameters) > 1:
            raise(ValueError, "Whoa there, why would fmin_bound allow more than 1 dim? What is this, matlab?")
        start = 1
        end = 0
        n_its = 0
        while numpy.abs(start - end) > 1e-4 and n_its < 10:
            start = self._neg_log_marginal_likelihood_given_hyperparameters(self.hyperparameters)
            lower_bound = numpy.array(self.hyperparameters) * lower_ratio
            upper_bound = numpy.array(self.hyperparameters) * upper_ratio
            # The parameters can change up to a half order of magnitude up or down
            # http://www.scipy.org/doc/api_docs/SciPy.optimize.optimize.html#fminbound
            fmin_out = fminbound(
                    self._neg_log_marginal_likelihood_given_hyperparameters,
                    lower_bound,
                    upper_bound,
                    full_output=1,
                    disp=3,
                    )
            n_its += 1
            self.hyperparameters = fmin_out[0]
            end = self._neg_log_marginal_likelihood_given_hyperparameters(self.hyperparameters)

    def _update_hyperparameters_via_marginal_BFGS(self, hyperparameter_grad_table=None):
        """Update via scipy.optimize.optimize.fmin_bfgs

        References:
            * http://www.scipy.org/doc/api_docs/SciPy.optimize.optimize.html#fmin_bfgs
            * http://en.wikipedia.org/wiki/BFGS_method
            * follows chapter 5 of RW (namely 5.9)
        """
        fmin_out = fmin_bfgs(
                self._neg_log_marginal_likelihood_given_hyperparameters,
                numpy.array(self.hyperparameters),
                fprime=self._gradient_of_marginal_wrt_hyperparameters,
                full_output=1,
                )
        self.hyperparameters = fmin_out[0]

    def _update_hyperparameters_via_LOO_CV_BFGS(self, hyperparameter_grad_table=None):
        """Update via scipy.optimize.optimize.fmin_bfgs

        References:
            * follows chapter 5 of RW (namely 5.4.2)
        """
        raise(NotImplementedError)

    def update_hyperparameters(self, GP, update_type=BFGS_MARGINAL_UPDATER, **kwargs):
        # TODO fix for new cov function
        self.GP = GP

        if update_type == BFGS_LOO_CV_UPDATER:
            self._update_hyperparameters_via_LOO_CV_BFGS(**kwargs)

        elif update_type == FMIN_MARGINAL_UPDATER:
            self._update_hyperparameters_via_marginal_fmin(**kwargs)

        elif update_type == BFGS_MARGINAL_UPDATER:
            self._update_hyperparameters_via_marginal_BFGS(**kwargs)
