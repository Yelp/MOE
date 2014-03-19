# -*- coding: utf-8 -*-

import collections
import logging
import numpy

import build.GPP as C_GP
from optimal_learning.EPI.src.python.models.optimal_gaussian_process import OptimalGaussianProcess


logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# namedtuple to track a closed interval [min, max]
class ClosedInterval(collections.namedtuple('ClosedInterval', ['min', 'max'])):
	__slots__ = ()

	@staticmethod
	def build_closed_intervals_from_list(domain):
		result = []
		for i, bounds in enumerate(domain):
			result.append(ClosedInterval(bounds[0], bounds[1]))

		return result

class HyperparameterOptimizationParameters(object):

        """Container for parameters that specify the behavior of a hyperparameter optimizer.

        We use slots to enforce a "type." Typo'ing a member name will error, not add a new field.
        This class is passed to C++, so it is convenient to be strict about its structure.

        Attributes:
        objective_type: which log likelihood measure to use as the metric of model quality
          e.g., log marginal likelihood, leave one out cross validation log pseudo-likelihood
        optimizer_type: which optimizer to use (e.g., dumb search, gradient dsecent, Newton)
        num_random_samples: number of samples to try if using 'dumb' search
        optimizer_parameters: parameters to control derviative-based optimizers, e.g.,
          step size control, number of steps tolerance, etc.
        NOTE: this MUST be a C++ object whose type matches objective_type. e.g., if objective_type
        is kNewton, then this must be built via C_GP.NewtonParameters()

        """

        __slots__ = ('objective_type', 'optimizer_type', 'num_random_samples', 'optimizer_parameters', )
        def __init__(self, objective_type=None, optimizer_type=None, num_random_samples=None, optimizer_parameters=None):
                # see gpp_python.cpp for .*_type enum definitions. .*_type variables must be from those enums (NOT integers)
                self.objective_type = objective_type
                self.optimizer_type = optimizer_type
                self.num_random_samples = num_random_samples # number of samples to 'dumb' search over
                self.optimizer_parameters = optimizer_parameters # must match the optimizer_type

class ExpectedImprovementOptimizationParameters(object):

        """Container for parameters that specify the behavior of a expected improvement optimizer.

        We use slots to enforce a "type." Typo'ing a member name will error, not add a new field.
        This class is passed to C++, so it is convenient to be strict about its structure.

        Attributes:
        domain_type: type of domain that we are optimizing expected improvement over (e.g., tensor, simplex)
        optimizer_type: which optimizer to use (e.g., dumb search, gradient dsecent)
        num_random_samples: number of samples to try if using 'dumb' search or if generating more
          than one simultaneous sample with dumb search fallback enabled
        optimizer_parameters: parameters to control derviative-based optimizers, e.g.,
          step size control, number of steps tolerance, etc.
        NOTE: this MUST be a C++ object whose type matches objective_type. e.g., if objective_type
        is kGradientDescent, then this must be built via C_GP.GradientDescentParameters object

        """

	__slots__ = ('domain_type', 'optimizer_type', 'num_random_samples', 'optimizer_parameters', )
	def __init__(self, domain_type=None, optimizer_type=None, num_random_samples=None, optimizer_parameters=None):
		# see gpp_python.cpp for .*_type enum definitions. .*_type variables must be from those enums (NOT integers)
		self.domain_type = domain_type
		self.optimizer_type = optimizer_type
		self.num_random_samples = num_random_samples # number of samples to 'dumb' search over
		self.optimizer_parameters = optimizer_parameters # must match the optimizer_type
		# NOTE: need both num_random_samples AND optimizer_parameters if generating > 1 sample
		# using gradient descent optimization

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
		return self._cppify(numpy.ones(size, dtype=numpy.float64)*numpy.float64(scalar))

	def _cppify(self, array):
		"""flattens an array or list and rewraps it in a list for cpp to consume
		"""
		return list(numpy.array(array).flatten())

	def _uncppify(self, array, expected_shape):
		"""un-flattens an array and makes it the expected shape
		"""
		return numpy.reshape(numpy.array(array), expected_shape)

	def _cppify_length(self):
		"""C++ interface expects length as a list of size dim=len(self.domain).
		Expands constant length values (e.g., = 1.0) to a list (e.g., = [1.0, 1.0, 1.0]).
		"""
		cop_length = self.cop.hyperparameters[1:]
		if len(cop_length) != len(self.domain):
			length = numpy.ones(len(self.domain), dtype=numpy.float64) * numpy.float64(cop_length[0])
		else:
			length = cop_length
		return self._cppify(length)

	@staticmethod
	def _build_newton_parameters(num_multistarts, max_num_steps, gamma, time_factor, max_relative_change, tolerance):
		"""Calls NewtonParameters' C++ constructor; this object specifies multistarted Newton behavior
		"""
		# details on gamma, pre-mult:
                # on i-th newton iteration, we add 1/(time_factor*gamma^(i+1)) * I to the Hessian to improve robustness
		newton_data = C_GP.NewtonParameters(
			num_multistarts, # number of initial guesses in multistarting
			max_num_steps, # maximum number of newton iterations per multistart
			numpy.float64(gamma), # exponent controlling rate of time_factor increase
			numpy.float64(time_factor), # initial amount of diagonal dominance
			numpy.float64(max_relative_change), # max relative change allowed per iteration of gradient descent
			numpy.float64(tolerance), # stop when the norm of gradient drops below this
			)
		return newton_data

	@staticmethod
	def _build_gradient_descent_parameters(num_multistarts, max_num_steps, max_num_restarts, gamma, pre_mult, max_relative_change, tolerance):
		"""Calls GradientDescentParameters' C++ constructor; this object specifies GD behavior
		"""
		# details on gamma, pre-mult:
		# GD may be implemented using a learning rate: pre_mult * (i+1)^{-\gamma}, where i is the current iteration
		gd_data = C_GP.GradientDescentParameters(
			num_multistarts, # number of initial guesses in multistarting
			max_num_steps, # maximum number of gradient descent iterations
			max_num_restarts, # maximum number of times we are allowed to call gradient descent.  Should be >= 2 as a minimum.
			numpy.float64(gamma), # exponent controlling rate of step size decrease
			numpy.float64(pre_mult), # scaling factor for step size
			numpy.float64(max_relative_change), # max relative change allowed per iteration of gradient descent
			numpy.float64(tolerance), # if gradient drops below this or we cannot move farther than this dist, stop
			)
		return gd_data

	def _build_cpp_gaussian_process(self):
		"""Calls GaussianProcess's C++ constructor, using parameters from the current state of this object
		"""
		size_of_domain = len(self.domain)

		# build a C++ GaussianProcess object
		gaussian_process = C_GP.GaussianProcess(
				self._cppify_hyperparameters(), # hyperparameters, e.g., [signal variance, [length scales]]; see _cppify_hyperparameter docs, C++ python interface docs
				self._cppify([p.point for p in self.points_sampled]), # points already sampled
				self._cppify(self.values_of_samples), # objective value at each sampled point
				self._cppify(self.sample_variance_of_samples), # noise variance, one value per sampled point
				size_of_domain, # dimension of parameter space
				len(self.points_sampled), # number of points already sampled
				)
		return gaussian_process

	def multistart_expected_improvement_optimization(self, ei_optimization_parameters, num_samples_to_generate, domain=None, points_being_sampled=None, mc_iterations=1000, status=None):
		"""Calls into multistart_expected_improvement_optimization_wrapper in EPI/src/cpp/GPP_python.cpp (solving q,p-EI)
		"""

		gaussian_process = self._build_cpp_gaussian_process()

		if domain is None:
			domain = self.domain

		if points_being_sampled is None:
			points_being_sampled = numpy.array([])

		if status is None:
			status = {}

		best_points_to_sample = C_GP.multistart_expected_improvement_optimization(
			ei_optimization_parameters, # ExpectedImprovementOptimizationParameters object (see MOE_driver.py)
			gaussian_process, # C++ GaussianProcess object (e.g., from self._build_cpp_gaussian_process())
			self._cppify(self.domain), # [lower, upper] bound pairs for each dimension
			self._cppify(points_being_sampled), # points to sample
			len(points_being_sampled), # number of points to sample
			num_samples_to_generate, # how many simultaneous experiments you would like to run
			self.best_so_far, # best known value of objective so far
			mc_iterations, # number of MC integration points in EI
			self.max_num_threads,
			self.randomness, # C++ RandomnessSourceContainer that holds enough randomness sources for multithreading
			status,
		)

		# reform output to be a list of dim-dimensional points, dim = len(self.domain)
		return self._uncppify(best_points_to_sample, (num_samples_to_generate, len(self.domain)))

	def _heuristic_expected_improvement_optimization(self, ei_optimization_parameters, num_samples_to_generate, estimation_policy, domain=None, status=None):
		"""
		Calls into heuristic_expected_improvement_optimization_wrapper in EPI/src/cpp/GPP_python.cpp

		Requires estimation_policy, a subclass of ObjectiveEstimationPolicyInterface (C++ pure abstract); examples include
		ConstantLiarEstimationPolicy and KrigingBelieverEstimationPolicy.
		"""

		gaussian_process = self._build_cpp_gaussian_process()

		if domain is None:
			domain = self.domain

		if status is None:
			status = {}

		best_points_to_sample = C_GP.heuristic_expected_improvement_optimization(
			ei_optimization_parameters, # ExpectedImprovementOptimizationParameters object (see MOE_driver.py)
			gaussian_process, # C++ GaussianProcess object (e.g., from self._build_cpp_gaussian_process())
			self._cppify(self.domain), # [lower, upper] bound pairs for each dimension
			estimation_policy, # estimation policy to use for guessing objective function values (e.g., ConstantLiar, KrigingBeliever)
			num_samples_to_generate, # how many simultaneous experiments you would like to run
			self.best_so_far, # best known value of objective so far
			self.max_num_threads,
			self.randomness, # C++ RandomnessSourceContainer that holds enough randomness sources for multithreading
			status,
		)

		# reform output to be a list of dim-dimensional points, dim = len(self.domain)
		return self._uncppify(best_points_to_sample, (num_samples_to_generate, len(self.domain)))

	def constant_liar_expected_improvement_optimization(self, ei_optimization_parameters, num_samples_to_generate, lie_value, lie_noise_variance=0.0, domain=None, status=None):
		"""
		Calls into heuristic_expected_improvement_optimization_wrapper in EPI/src/cpp/GPP_python.cpp (solving q,0-EI)
		with the ConstantLiarEstimationPolicy.

		double lie_value: the "constant lie" that this estimator should return
		double lie_noise_variance: the noise_variance to associate to the lie_value (MUST be >= 0.0)
		"""

		estimation_policy = C_GP.ConstantLiarEstimationPolicy(lie_value, lie_noise_variance)
		return self._heuristic_expected_improvement_optimization(ei_optimization_parameters, num_samples_to_generate, estimation_policy, domain, status)

	def kriging_believer_expected_improvement_optimization(self, ei_optimization_parameters, num_samples_to_generate, std_deviation_coef=0.0, kriging_noise_variance=0.0, domain=None, status=None):
		"""
		Calls into heuristic_expected_improvement_optimization_wrapper in EPI/src/cpp/GPP_python.cpp (solving q,0-EI)
		with the KrigingBelieverEstimationPolicy.

		double std_deviation_coef: the relative amount of bias (in units of GP std deviation) to introduce into the GP mean
		double kriging_noise_variance: the noise_variance to associate to each function value estimate (MUST be >= 0.0)
		"""

		estimation_policy = C_GP.KrigingBelieverEstimationPolicy(std_deviation_coef, kriging_noise_variance)
		return self._heuristic_expected_improvement_optimization(ei_optimization_parameters, num_samples_to_generate, estimation_policy, domain, status)

	# TODO(eliu): this call is DEPRECATED; use multistart_expected_improvement_optimization instead!
	# not deleting yet in case this screws up inheritance (since this overrides superclass member functions)
	def get_multistart_best(self, starting_points=None, points_being_sampled=None, gamma=0.9, gd_iterations=1000, mc_iterations=1000, num_multistarts=5, max_num_restarts=3, max_relative_change=1.0, tolerance=1.0e-7, status=None):
		"""Wrapper for multistart_expected_improvement_optimization
		"""

		ei_gradient_descent_optimization_parameters = ExpectedImprovementOptimizationParameters()
		ei_gradient_descent_optimization_parameters.domain_type = C_GP.DomainTypes.tensor_product
		ei_gradient_descent_optimization_parameters.optimizer_type = C_GP.OptimizerTypes.gradient_descent
		ei_gradient_descent_optimization_parameters.num_random_samples = 0
		ei_gradient_descent_optimization_parameters.optimizer_parameters = self._build_gradient_descent_parameters(
			num_multistarts, # num_multistarts
			gd_iterations, # max_num_steps
			max_num_restarts, # max_num_restarts
			gamma, # gamma,
			1.0, # pre_mult
			max_relative_change, # max_relative_change
			tolerance, #tolerance
		)

		num_samples_to_generate = 1 # the deprecated form of this was written with only the 1 sample to generate case in mind
		# uncppify b/c users of get_multistart_best expect a list of size len(self.domain)
		# with the coordinates
		# multistart_expected_improvement_optimization will return a list of num_samples_to_generate
		# lists, each of size len(self.domain)
		return self._uncppify(self.multistart_expected_improvement_optimization(ei_gradient_descent_optimization_parameters, num_samples_to_generate, domain=None, points_being_sampled=points_being_sampled, mc_iterations=mc_iterations, status=status), len(self.domain))

	def evaluate_expected_improvement_at_point_list(self, points_to_evaluate, points_being_sampled=None, mc_iterations=1000, status=None):
		"""Calls into evaluate_EI_at_point_list_wrapper() in src/cpp/GPP_python.cpp
		"""

		gaussian_process = self._build_cpp_gaussian_process()

		if points_being_sampled is None:
			points_being_sampled = numpy.array([])

		if status is None:
			status = {}

		return C_GP.evaluate_EI_at_point_list(
				gaussian_process, # C++ GaussianProcess object (e.g., from self._build_cpp_gaussian_process())
				self._cppify(points_to_evaluate), # points at which to evaluate EI
				self._cppify(points_being_sampled), # points to sample
				len(points_to_evaluate), # number of points to evaluate
				len(points_being_sampled), # number of points to sample
				self.best_so_far, # best known value of objective so far
				mc_iterations, # number of MC integration points in EI
				self.max_num_threads,
				self.randomness, # C++ RandomnessSourceContainer that holds enough randomness sources for multithreading
				status,
				)

	def get_grad_mu(self, points_to_sample):
		"""Calls into get_grad_mean_wrapper in src/cpp/GPP_python.cpp
		"""
		# hit the cpp with a grad_mu request
		num_points_to_sample = len(points_to_sample)
		size_of_domain = len(self.domain)

		gaussian_process = self._build_cpp_gaussian_process()

		grad_mu = C_GP.get_grad_mean(
				gaussian_process, # C++ GaussianProcess object (e.g., from self._build_cpp_gaussian_process())
				self._cppify(points_to_sample), # points to sample
				num_points_to_sample, # number of points to sample
				)
		# reform it to match the overidden method
		return self._uncppify(grad_mu, (num_points_to_sample, size_of_domain))

	def cholesky_decomp_and_grad(self, points_to_sample, var_of_grad=0):
		"""Calls into get_chol_var and get_grad_var in src/cpp/GPP_python.cpp
		"""
		num_points_to_sample = len(points_to_sample)
		size_of_domain = len(self.domain)

		gaussian_process = self._build_cpp_gaussian_process()

		cholesky_var = C_GP.get_chol_var(
				gaussian_process, # C++ GaussianProcess object (e.g., from self._build_cpp_gaussian_process())
				self._cppify(points_to_sample), # points to sample
				num_points_to_sample, # number of points to sample
				)
		# reform it to match the overidden method
		python_cholesky_var = self._uncppify(cholesky_var, (num_points_to_sample, num_points_to_sample))

		grad_cholesky_var = C_GP.get_grad_var(
				gaussian_process, # C++ GaussianProcess object (e.g., from self._build_cpp_gaussian_process())
				self._cppify(points_to_sample), # points to sample
				num_points_to_sample, # number of points to sample
				var_of_grad, # dimension to differentiate in
				)
		# reform it to match the overidden method
		python_grad_cholesky_var = self._uncppify(grad_cholesky_var, (num_points_to_sample, num_points_to_sample, size_of_domain))

		return python_cholesky_var, python_grad_cholesky_var

	def compute_expected_improvement(self, points_to_sample, force_monte_carlo=False, mc_iterations=1000):
		"""Compute expected improvement. Calls into src/cpp/GPP_python.cpp

		Automatically selects analytic evaluators when they are available (for performance/accuracy).
		Set "force_monte_carlo" to True to force monte-carlo evaluation even if analytic is available.
		(This is probably only useful for testing.)
		"""
		gaussian_process = self._build_cpp_gaussian_process()

		return C_GP.compute_expected_improvement(
			gaussian_process, # C++ GaussianProcess object (e.g., from self._build_cpp_gaussian_process())
			self._cppify(points_to_sample), # points to sample
			len(points_to_sample), # number of points to sample
			mc_iterations,
			self.best_so_far, # best known value of objective so far
			force_monte_carlo,
			self.randomness,
		)

	def compute_grad_expected_improvement(self, points_to_sample, force_monte_carlo=False, mc_iterations=1000):
		"""Compute spatial gradient of expected improvement. Calls into src/cpp/GPP_python.cpp

		Automatically selects analytic evaluators when they are available (for performance/accuracy).
		Set "force_monte_carlo" to True to force monte-carlo evaluation even if analytic is available.
		(This is probably only useful for testing.)
		"""
		gaussian_process = self._build_cpp_gaussian_process()

		current_point = points_to_sample[-1]
		points_to_sample_temp = points_to_sample[:-1]
		num_points_to_sample = len(points_to_sample_temp)

		grad_EI = C_GP.compute_grad_expected_improvement(
			gaussian_process, # C++ GaussianProcess object (e.g., from self._build_cpp_gaussian_process())
			self._cppify([points_to_sample_temp]), # points to sample
			num_points_to_sample, # number of points to sample
			mc_iterations,
			self.best_so_far, # best known value of objective so far
			force_monte_carlo,
			self.randomness,
			self._cppify(current_point),
		)
		# TODO(eliu): sclark(?) why is this wrapped in extra [] brackets?
		return numpy.array([[grad_EI]])

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
		num_points_to_sample = len(points_to_sample)

		gaussian_process = self._build_cpp_gaussian_process()

		# hit the cpp with a get_mean request
		mu = C_GP.get_mean(
				gaussian_process, # C++ GaussianProcess object (e.g., from self._build_cpp_gaussian_process())
				self._cppify(points_to_sample), # points to sample
				num_points_to_sample, # number of points to sample
				)

		# hit the cpp with a get_var request
		var = C_GP.get_var(
				gaussian_process, # C++ GaussianProcess object (e.g., from self._build_cpp_gaussian_process())
				self._cppify(points_to_sample), # points to sample
				num_points_to_sample, # number of points to sample
				)

		python_mu = numpy.array(mu)
		python_var = self._uncppify(var, (num_points_to_sample, num_points_to_sample))
		return python_mu, python_var

	def compute_log_likelihood(self, objective_type=C_GP.LogLikelihoodTypes.log_marginal_likelihood):
		"""Calls into compute_log_likelihood in EPI/src/cpp/GPP_python.cpp to compute
		the requested log_likelihood measure (e.g., log marginal or leave one out)
		"""
		return C_GP.compute_log_likelihood(
				self._cppify([p.point for p in self.points_sampled]), # points already sampled
				self._cppify(self.values_of_samples), # objective value at each sampled point
				len(self.domain), # dimension of parameter space
				len(self.points_sampled), # number of points already sampled
				objective_type, # log likelihood measure to eval (e.g., LogLikelihoodTypes.log_marginal_likelihood, see gpp_python.cpp for enum declaration)
				self._cppify_hyperparameters(), # hyperparameters, e.g., [signal variance, [length scales]]; see _cppify_hyperparameter docs, C++ python interface docs
				self._cppify(self.sample_variance_of_samples), # noise variance, one value per sampled point
				)

	def compute_hyperparam_grad_log_likelihood(self, objective_type=C_GP.LogLikelihoodTypes.log_marginal_likelihood):
		"""Calls into compute_hyperparam_grad_log_likelihood in EPI/src/cpp/GPP_python.cpp to compute
		the gradient of the requested log_likelihood measure (e.g., log marginal or leave one out) wrt the hyperparameters
		"""
		grad_log_marginal = C_GP.compute_hyperparameter_grad_log_likelihood(
				self._cppify([p.point for p in self.points_sampled]), # points already sampled
				self._cppify(self.values_of_samples), # objective value at each sampled point
				len(self.domain), # dimension of parameter space
				len(self.points_sampled), # number of points already sampled
				objective_type, # log likelihood measure to eval (e.g., LogLikelihoodTypes.log_marginal_likelihood, see gpp_python.cpp for enum declaration)
				self._cppify_hyperparameters(), # hyperparameters, e.g., [signal variance, [length scales]]; see _cppify_hyperparameter docs, C++ python interface docs
				self._cppify(self.sample_variance_of_samples), # noise variance, one value per sampled point
				)
		return numpy.array(grad_log_marginal)

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

	def multistart_hyperparameter_optimization(self, hyperparameter_optimization_parameters, hyperparameter_domain=None, status=None):
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
		# default domain is [0.01, 10] for every hyperparameter
		if not hyperparameter_domain:
			hyperparameter_domain = []
			for i in range(0, 1 + len(self.domain)):
				hyperparameter_domain[2*i + 0] = -2.0
				hyperparameter_domain[2*i + 1] = 1.0

		if status is None:
			status = {} # must be an initialized dict for the call to C++

		return C_GP.multistart_hyperparameter_optimization(
				hyperparameter_optimization_parameters, # HyperparameterOptimizationParameters object (see MOE_driver)
				self._cppify(hyperparameter_domain), # domain of hyperparameters in LOG-10 SPACE
				self._cppify([p.point for p in self.points_sampled]), # points already sampled
				self._cppify(self.values_of_samples), # objective value at each sampled point
				len(self.domain), # dimension of parameter space
				len(self.points_sampled), # number of points already sampled
				self._cppify_hyperparameters(), # hyperparameters, e.g., [signal variance, [length scales]]; see _cppify_hyperparameter docs, C++ python interface docs
				self._cppify(self.sample_variance_of_samples), # noise variance, one value per sampled point
				self.max_num_threads,
				self.randomness, # C++ RandomnessSourceContainer that holds a UniformRandomGenerator object
				status, # status report on optimizer success, etc.
				)

	def evaluate_log_likelihood_at_hyperparameter_list(self, hyperparameters_to_evaluate, objective_type=C_GP.LogLikelihoodTypes.log_marginal_likelihood):
		""" Calls into evaluate_log_likelihood_at_hyperparameter_list_wrapper() in src/cpp/GPP_python.cpp
		"""
		return C_GP.evaluate_log_likelihood_at_hyperparameter_list(
				self._cppify(hyperparameters_to_evaluate), # hyperparameters at which to compute log likelihood
				self._cppify([p.point for p in self.points_sampled]), # points already sampled
				self._cppify(self.values_of_samples), # objective value at each sampled point
				len(self.domain), # dimension of parameter space
				len(self.points_sampled), # number of points already sampled
				objective_type, # log likelihood measure to eval (e.g., LogLikelihoodTypes.log_marginal_likelihood, see gpp_python.cpp for enum declaration)
				self._cppify_hyperparameters(), # hyperparameters, e.g., [signal variance, [length scales]]; see _cppify_hyperparameter docs, C++ python interface docs
				self._cppify(self.sample_variance_of_samples), # noise variance, one value per sampled point
				len(hyperparameters_to_evaluate), # number of hyperparameter points to evaluate
				self.max_num_threads,
				)
