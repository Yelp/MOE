# -*- coding: utf-8 -*-

import collections
import logging
import numpy

import moe.build.GPP as C_GP
import moe.optimal_learning.python.cpp_wrappers.cpp_utils as cpp_utils
import moe.optimal_learning.python.cpp_wrappers.domain as cpp_domain
from moe.optimal_learning.python.cpp_wrappers.covariance import SquareExponential
from moe.optimal_learning.python.cpp_wrappers.gaussian_process import GaussianProcess
import moe.optimal_learning.python.cpp_wrappers.expected_improvement as cpp_ei
import moe.optimal_learning.python.cpp_wrappers.log_likelihood as cpp_log_likelihood
from moe.optimal_learning.python.cpp_wrappers.optimization import GradientDescentParameters, GradientDescentOptimizer
from moe.optimal_learning.python.data_containers import SamplePoint, HistoricalData
from moe.optimal_learning.python.models.optimal_gaussian_process import OptimalGaussianProcess


logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

class OptimalGaussianProcessLinkedCpp(OptimalGaussianProcess):
    """Overrides methods in OptimalGaussianProcess with calls to C++ code in src/cpp/GPP_python.cpp via boost
    """
    def __init__(self, max_num_threads=1, *args, **kwargs):
        """Initializes max_num_threads (which should be constant for any given instantiation of this class)
        Creates a singleton randomness member (C++ RandomnessSourceContainer object) and initializes seeds randomness
        sources based on system time (+ potentially other less repeatable quantities) and thread id (as appropriate)
        """
        super(OptimalGaussianProcessLinkedCpp, self).__init__(*args, **kwargs)
        self.max_num_threads = max_num_threads
        self.randomness = C_GP.RandomnessSourceContainer(self.max_num_threads)
        self.randomness.SetRandomizedUniformGeneratorSeed(0) # set seed based on less repeatable factors (e.g,. time)
        self.randomness.SetRandomizedNormalRNGSeed(0) # set seed baesd on thread id & less repeatable factors (e.g,. time)

    def _cppify_hyperparameters(self):
        """C++ interface expects hyperparameters in a list, where:
        hyperparameters[0]: double = \alpha (\sigma_f^2, signal variance)
        hyperparameters[1]: list = length scales (len = dim, one length per spatial dimension)
        """
        # HACK
        alpha = self.cop.hyperparameters[0]
        return [numpy.float64(alpha), self._cppify_length()]

    def _cppify_extend_scalar_to_list(self, scalar, size):
        """Python internally stores some vector values as scalars when they are constant.
        C++ internals do not support this, so we need to convert the scalar to an explicit vector
        """
        return cpp_utils.cppify(numpy.ones(size, dtype=numpy.float64)*numpy.float64(scalar))

    def _cppify_length(self):
        """C++ interface expects length as a list of size dim=len(self.domain).
        Expands constant length values (e.g., = 1.0) to a list (e.g., = [1.0, 1.0, 1.0]).
        """
        cop_length = self.cop.hyperparameters[1:]
        if len(cop_length) != len(self.domain):
            length = numpy.ones(len(self.domain), dtype=numpy.float64) * numpy.float64(cop_length[0])
        else:
            length = cop_length
        return cpp_utils.cppify(length)

    def _build_new_environment(self):
        cop_length = self.cop.hyperparameters[1:]
        if len(cop_length) != len(self.domain):
            length = numpy.ones(len(self.domain), dtype=numpy.float64) * numpy.float64(cop_length[0])
        else:
            length = cop_length
        hyper = [self.cop.hyperparameters[0]]
        hyper.extend(length)
        cov = SquareExponential(hyper)

        history = HistoricalData(len(self.domain), self.points_sampled)

        return cov, history

    def _build_cpp_gaussian_process(self):
        cov, history = self._build_new_environment()
        return GaussianProcess(cov, history)

    def multistart_expected_improvement_optimization(self, optimizer_type, optimization_parameters, domain_type, num_random_samples, num_samples_to_generate, domain=None, points_being_sampled=numpy.array([]), mc_iterations=1000, status=None):
        """Calls into multistart_expected_improvement_optimization_wrapper in cpp/GPP_python.cpp (solving q,p-EI)
        """

        if domain is None:
            domain = self.domain

        gaussian_process = self._build_cpp_gaussian_process()
        ei_evaluator = cpp_ei.ExpectedImprovement(gaussian_process, numpy.array([]), points_to_sample=points_being_sampled, num_mc_iterations=mc_iterations, randomness=self.randomness)
        if domain_type == C_GP.DomainTypes.tensor_product:
            new_domain = cpp_domain.TensorProductDomain(domain)
        else:
            new_domain = cpp_domain.SimplexIntersectTensorProductDomain(domain)

        ei_optimizer = optimizer_type(new_domain, ei_evaluator, optimization_parameters, num_random_samples=num_random_samples)

        return cpp_ei.multistart_expected_improvement_optimization(ei_optimizer, None, num_samples_to_generate, randomness=self.randomness, max_num_threads=self.max_num_threads, status=status)

    def _heuristic_expected_improvement_optimization(self, optimizer_type, optimization_parameters, domain_type, num_random_samples, num_samples_to_generate, estimation_policy, domain=None, status=None):
        """
        Calls into heuristic_expected_improvement_optimization_wrapper in cpp/GPP_python.cpp

        Requires estimation_policy, a subclass of ObjectiveEstimationPolicyInterface (C++ pure abstract); examples include
        ConstantLiarEstimationPolicy and KrigingBelieverEstimationPolicy.
        """
        if domain is None:
            domain = self.domain

        gaussian_process = self._build_cpp_gaussian_process()
        ei_evaluator = cpp_ei.ExpectedImprovement(gaussian_process, numpy.array([]), num_mc_iterations=0, randomness=self.randomness)
        if domain_type == C_GP.DomainTypes.tensor_product:
            new_domain = cpp_domain.TensorProductDomain(domain)
        else:
            new_domain = cpp_domain.SimplexIntersectTensorProductDomain(domain)

        ei_optimizer = optimizer_type(new_domain, ei_evaluator, optimization_parameters, num_random_samples=num_random_samples)

        return cpp_ei._heuristic_expected_improvement_optimization(ei_optimizer, None, num_samples_to_generate, estimation_policy, randomness=self.randomness, max_num_threads=self.max_num_threads, status=status)

    def constant_liar_expected_improvement_optimization(self, optimizer_type, optimization_parameters, domain_type, num_random_samples, num_samples_to_generate, lie_value, lie_noise_variance=0.0, domain=None, status=None):
        """
        Calls into heuristic_expected_improvement_optimization_wrapper in cpp/GPP_python.cpp (solving q,0-EI)
        with the ConstantLiarEstimationPolicy.

        double lie_value: the "constant lie" that this estimator should return
        double lie_noise_variance: the noise_variance to associate to the lie_value (MUST be >= 0.0)
        """
        estimation_policy = C_GP.ConstantLiarEstimationPolicy(lie_value, lie_noise_variance)
        return self._heuristic_expected_improvement_optimization(optimizer_type, optimization_parameters, domain_type, num_random_samples, num_samples_to_generate, estimation_policy, domain, status)

    def kriging_believer_expected_improvement_optimization(self, optimizer_type, optimization_parameters, domain_type, num_random_samples, num_samples_to_generate, std_deviation_coef=0.0, kriging_noise_variance=0.0, domain=None, status=None):
        """
        Calls into heuristic_expected_improvement_optimization_wrapper in cpp/GPP_python.cpp (solving q,0-EI)
        with the KrigingBelieverEstimationPolicy.

        double std_deviation_coef: the relative amount of bias (in units of GP std deviation) to introduce into the GP mean
        double kriging_noise_variance: the noise_variance to associate to each function value estimate (MUST be >= 0.0)
        """
        estimation_policy = C_GP.KrigingBelieverEstimationPolicy(std_deviation_coef, kriging_noise_variance)
        return self._heuristic_expected_improvement_optimization(optimizer_type, optimization_parameters, domain_type, num_random_samples, num_samples_to_generate, estimation_policy, domain, status)

    # TODO(eliu): this call is DEPRECATED; use multistart_expected_improvement_optimization instead!
    # not deleting yet in case this screws up inheritance (since this overrides superclass member functions)
    def get_multistart_best(self, starting_points=None, points_being_sampled=numpy.array([]), gamma=0.9, gd_iterations=1000, mc_iterations=1000, num_multistarts=5, max_num_restarts=3, max_relative_change=1.0, tolerance=1.0e-7, status=None):
        """Wrapper for multistart_expected_improvement_optimization
        """
        optimization_parameters = GradientDescentParameters(
            num_multistarts, # num_multistarts
            gd_iterations, # max_num_steps
            max_num_restarts, # max_num_restarts
            gamma, # gamma,
            1.0, # pre_mult
            max_relative_change, # max_relative_change
            tolerance, #tolerance
        )

        optimizer_type = GradientDescentOptimizer
        num_random_samples = 0

        domain_type = C_GP.DomainTypes.tensor_product
        num_samples_to_generate = 1 # the deprecated form of this was written with only the 1 sample to generate case in mind
        # uncppify b/c users of get_multistart_best expect a list of size len(self.domain)
        # with the coordinates
        # multistart_expected_improvement_optimization will return a list of num_samples_to_generate
        # lists, each of size len(self.domain)
        return cpp_utils.uncppify(self.multistart_expected_improvement_optimization(optimizer_type, optimization_parameters, domain_type, num_random_samples, num_samples_to_generate, domain=None, points_being_sampled=points_being_sampled, mc_iterations=mc_iterations, status=status), len(self.domain))

    def evaluate_expected_improvement_at_point_list(self, points_to_evaluate, points_being_sampled=numpy.array([]), mc_iterations=1000, status=None):
        """Calls into evaluate_EI_at_point_list_wrapper() in src/cpp/GPP_python.cpp
        """
        gaussian_process = self._build_cpp_gaussian_process()
        ei_evaluator = cpp_ei.ExpectedImprovement(gaussian_process, numpy.array([]), points_to_sample=points_being_sampled, num_mc_iterations=mc_iterations, randomness=self.randomness)
        return cpp_ei.evaluate_expected_improvement_at_point_list(ei_evaluator, points_to_evaluate, randomness=self.randomness, max_num_threads=self.max_num_threads, status=status)

    def get_grad_mu(self, points_to_sample):
        """Calls into get_grad_mean_wrapper in src/cpp/GPP_python.cpp
        """
        gaussian_process = self._build_cpp_gaussian_process()
        return gaussian_process.compute_grad_mean_of_points(numpy.asarray(points_to_sample))

    def cholesky_decomp_and_grad(self, points_to_sample, var_of_grad=0):
        """Calls into get_chol_var and get_grad_var in src/cpp/GPP_python.cpp
        """
        gaussian_process = self._build_cpp_gaussian_process()
        points_to_sample = numpy.asarray(points_to_sample)
        python_cholesky_var = gaussian_process.compute_cholesky_variance_of_points(points_to_sample)
        # python_grad_cholesky_var = gaussian_process.compute_grad_cholesky_variance_of_points(points_to_sample, var_of_grad)

        num_to_sample = points_to_sample.shape[0]
        grad_chol_decomp = C_GP.get_grad_chol_var(
            gaussian_process._gaussian_process,
            cpp_utils.cppify(points_to_sample),
            num_to_sample,
            var_of_grad,
        )
        python_grad_cholesky_var = cpp_utils.uncppify(grad_chol_decomp, (num_to_sample, num_to_sample, gaussian_process.dim))

        return python_cholesky_var, python_grad_cholesky_var

    def compute_expected_improvement(self, points_to_sample, force_monte_carlo=False, mc_iterations=1000):
        """Compute expected improvement. Calls into src/cpp/GPP_python.cpp

        Automatically selects analytic evaluators when they are available (for performance/accuracy).
        Set "force_monte_carlo" to True to force monte-carlo evaluation even if analytic is available.
        (This is probably only useful for testing.)
        """
        gaussian_process = self._build_cpp_gaussian_process()

        current_point = numpy.array(points_to_sample[-1])
        points_to_sample_temp = numpy.array(points_to_sample[:-1])
        ei_evaluator = cpp_ei.ExpectedImprovement(gaussian_process, current_point, points_to_sample=points_to_sample_temp, num_mc_iterations=mc_iterations, randomness=self.randomness)
        return ei_evaluator.compute_expected_improvement(
            force_monte_carlo=force_monte_carlo,
        )

    def compute_grad_expected_improvement(self, points_to_sample, force_monte_carlo=False, mc_iterations=1000):
        """Compute spatial gradient of expected improvement. Calls into src/cpp/GPP_python.cpp

        Automatically selects analytic evaluators when they are available (for performance/accuracy).
        Set "force_monte_carlo" to True to force monte-carlo evaluation even if analytic is available.
        (This is probably only useful for testing.)
        """
        gaussian_process = self._build_cpp_gaussian_process()

        # current point being sampled is the last point by convention
        current_point = numpy.array(points_to_sample[-1])
        # remaining points represented as concurrent experiments
        points_to_sample_temp = numpy.array(points_to_sample[:-1])
        ei_evaluator = cpp_ei.ExpectedImprovement(gaussian_process, current_point, points_to_sample=points_to_sample_temp, num_mc_iterations=mc_iterations, randomness=self.randomness)
        return ei_evaluator.compute_grad_expected_improvement(
            force_monte_carlo=force_monte_carlo,
        )[0]

    # TODO(eliu): this call is DEPRECATED; use compute_grad_expected_improvement instead!
    # not deleting yet in case this screws up inheritance (since this overrides superclass member functions)
    def get_1D_analytic_grad_EI(self, point_to_sample):
        return self.compute_grad_expected_improvement(
            [point_to_sample], # callsite expects a list of points, so we need to wrap
            force_monte_carlo=False,
            mc_iterations=0,
        )

    # TODO(eliu): this call is DEPRECATED; use compute_expected_improvement instead!
    # not deleting yet in case this screws up inheritance (since this overrides superclass member functions)
    def get_1D_analytic_expected_improvement(self, point_to_sample):
        return self.compute_expected_improvement(
            [point_to_sample], # callsite expects a list of points, so we need to wrap
            force_monte_carlo=False,
            mc_iterations=0,
        )

    # TODO(eliu): this call is DEPRECATED; use compute_expected_improvement instead!
    # not deleting yet in case this screws up inheritance (since this overrides superclass member functions)
    def get_expected_improvement(self, points_to_sample, mc_iterations=1000):
        return self.compute_expected_improvement(
            points_to_sample,
            force_monte_carlo=True,
            mc_iterations=mc_iterations,
        )

    # TODO(eliu): this call is DEPRECATED; use compute_grad_expected_improvement instead!
    # not deleting yet in case this screws up inheritance (since this overrides superclass member functions)
    def get_grad_expected_improvement(self, points_to_sample, mc_iterations=1000):
        return self.compute_grad_expected_improvement(
            points_to_sample,
            force_monte_carlo=True,
            mc_iterations=mc_iterations,
        )

    def get_mean_and_var_of_points(self, points_to_sample):
        """Calls into get_mean and get_var wrapper in src/cpp/GPP_python.cpp
        """
        gaussian_process = self._build_cpp_gaussian_process()
        python_mu = gaussian_process.compute_mean_of_points(numpy.asarray(points_to_sample))
        python_var = gaussian_process.compute_variance_of_points(numpy.asarray(points_to_sample))
        return python_mu, python_var

    def compute_log_likelihood(self, objective_type=C_GP.LogLikelihoodTypes.log_marginal_likelihood):
        """Calls into compute_log_likelihood in cpp/GPP_python.cpp to compute
        the requested log_likelihood measure (e.g., log marginal or leave one out)
        """
        cov, history = self._build_new_environment()
        log_likelihood_evaluator = cpp_log_likelihood.GaussianProcessLogLikelihood(cov, history, log_likelihood_type=objective_type)
        return log_likelihood_evaluator.compute_log_likelihood()

    def compute_hyperparam_grad_log_likelihood(self, objective_type=C_GP.LogLikelihoodTypes.log_marginal_likelihood):
        """Calls into compute_hyperparam_grad_log_likelihood in cpp/GPP_python.cpp to compute
        the gradient of the requested log_likelihood measure (e.g., log marginal or leave one out) wrt the hyperparameters
        """
        cov, history = self._build_new_environment()
        log_likelihood_evaluator = cpp_log_likelihood.GaussianProcessLogLikelihood(cov, history, log_likelihood_type=objective_type)
        return log_likelihood_evaluator.compute_grad_log_likelihood()

    # TODO(eliu): this call is DEPRECATED; use compute_log_likelihood instead!
    # not deleting yet in case this screws up inheritance (since this overrides superclass member functions)
    def get_log_marginal_likelihood(self):
        """Wrapper for compute_log_likelihood
        """
        return self.compute_log_likelihood(objective_type=C_GP.LogLikelihoodTypes.log_marginal_likelihood)

    # TODO(eliu): this call is DEPRECATED; use compute_log_likelihood instead!
    # not deleting yet in case this screws up inheritance (since this overrides superclass member functions)
    def get_hyperparam_grad_log_marginal_likelihood(self):
        """Wrapper for compute_hyperparam_grad_log_likelihood
        """
        return self.compute_hyperparam_grad_log_likelihood(objective_type=C_GP.LogLikelihoodTypes.log_marginal_likelihood)

    def multistart_hyperparameter_optimization(self, optimizer_type, optimization_parameters, hyperparameter_domain=None, status=None):
        """ Optimizes hyperparameters based on maximizing the log likelihood-like measures using a multistart optimizer:
        log_marginal_likelihood     : log marginal likelihood (that the data comes from the model, p(y | X, \theta).
        leave_one_out_log_likelihood: leave-one-out log pseudo-likelihood, which evaluates the ability of the
        model to predict each member of its training set:
        \sum_{i=1}^n log p(y_i | X_{-i}, y_{-i}, \theta), where X_{-i}, y_{-i} denotes the parent set with i-th member removed

        Optimizers are: null ('dumb' search), gradient descent, newton
        'dumb' search means this will just evaluate the objective log likelihood measure at num_multistarts 'points'
        (hyperparameters) in the domain, uniformly sampled using latin hypercube sampling.

        See gpp_python.cpp for C++ enum declarations laying out the options for objective and optimizer types.
        """
        cov, history = self._build_new_environment()
        log_likelihood_evaluator = cpp_log_likelihood.GaussianProcessLogLikelihood(cov, history, log_likelihood_type=hyperparameter_optimization_parameters.objective_type)
        # Guess "reasonable" hyperparameter constraints if none are given.
        if not hyperparameter_domain:
            hyperparameter_domain = cpp_domain.TensorProductDomain([ClosedInterval(0.01, 10.0)] * log_likelihood_evaluator.num_hyperparameters)

        num_random_samples = 0
        log_likelihood_optimizer = optimizer_type(hyperparameter_domain, log_likelihood_evaluator, optimization_parameters)

        return cpp_log_likelihood.multistart_hyperparameter_optimization(log_likelihood_optimizer, optimization_parameters, num_random_samples, hyperparameter_domain=hyperparameter_domain, randomness=self.randomness, max_num_threads=self.max_num_threads, status=status)

    def evaluate_log_likelihood_at_hyperparameter_list(self, hyperparameters_to_evaluate, objective_type=C_GP.LogLikelihoodTypes.log_marginal_likelihood):
        """ Calls into evaluate_log_likelihood_at_hyperparameter_list_wrapper() in src/cpp/GPP_python.cpp
        """
        cov, history = self._build_new_environment()
        log_likelihood_evaluator = cpp_log_likelihood.GaussianProcessLogLikelihood(cov, history, log_likelihood_type=objective_type)
        return cpp_log_likelihood.evaluate_log_likelihood_at_hyperparameter_list(log_likelihood_evaluator, hyperparameters_to_evaluate, max_num_threads=self.max_num_threads)
