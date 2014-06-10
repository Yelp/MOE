# -*- coding: utf-8 -*-
# In principle you can replace cpp_wrappers with python_version and things should still work
import numpy

import moe.build.GPP as cpp_optimal_learning
from moe.optimal_learning.python import constant
from moe.optimal_learning.python import data_containers
from moe.optimal_learning.python import gaussian_process_test_utils
from moe.optimal_learning.python import geometry_utils
import moe.optimal_learning.python.cpp_wrappers as optimal_learning_base
import moe.optimal_learning.python.cpp_wrappers.covariance
import moe.optimal_learning.python.cpp_wrappers.expected_improvement
import moe.optimal_learning.python.cpp_wrappers.domain
import moe.optimal_learning.python.cpp_wrappers.gaussian_process
import moe.optimal_learning.python.cpp_wrappers.log_likelihood
import moe.optimal_learning.python.cpp_wrappers.optimization


def build_gaussian_process_example_data(covariance_class, num_hyperparameters, domain, num_sampled, noise_variance_base):
    hyperparameter_interval = geometry_utils.ClosedInterval(0.3, 2.5)
    covariance_generation = gaussian_process_test_utils.fill_random_covariance_hyperparameters(
        hyperparameter_interval,
        num_hyperparameters,
        covariance_type=covariance_class,
    )

    noise_variance = numpy.full(num_sampled, noise_variance_base)
    points_sampled = geometry_utils.generate_latin_hypercube_points(num_sampled, domain._domain_bounds)
    gaussian_process = gaussian_process_test_utils.build_random_gaussian_process(
        points_sampled,
        covariance_generation,
        noise_variance=noise_variance,
        gaussian_process_type=optimal_learning_base.gaussian_process.GaussianProcess,
    )
    return gaussian_process._historical_data


def demo():
    # INPUTS
    # 1) Core Properties
    # specifies the domain of each independent variable in (min, max) pairs
    domain_bounds = geometry_utils.ClosedInterval.build_closed_intervals_from_list(
        [[-2.5, 1.0], [-0.1, 1.1], [-0.5, 0.5]]
    )
    domain = optimal_learning_base.domain.TensorProductDomain(domain_bounds)

    # number of concurrent samples running alongside the optimization
    num_to_sample = 0  # >= 0

    # covariance selection
    covariance_class = optimal_learning_base.covariance.SquareExponential
    # hyperparameters (arbitrary if using model selection, important if not)
    hyperparameters_initial = numpy.full(domain.dim + 1, 1.0)  # all 1.0 for now
    covariance = covariance_class(hyperparameters_initial)

    # 2) Core Data
    # number of points that we have already sampled; i.e., size of the training set
    num_sampled = 23  # >= 0

    # At the end of this, you should have a complete HistoricalData object. You can use
    # whatever data you want; here is an exmaple generating the data randomly.
    numpy.random.seed(8962)
    noise_variance_base = 1.0e-2
    historical_data = build_gaussian_process_example_data(covariance_class, covariance.num_hyperparameters, domain, num_sampled, noise_variance_base)

    # 3) Hyperparameter Optimization Parameters
    # hyperparameter domain
    hyperparameter_domain_bounds = geometry_utils.ClosedInterval.build_closed_intervals_from_list(
        [geometry_utils.ClosedInterval(1.0e-5, 1.0e3)] * covariance.num_hyperparameters
    )
    hyperparameter_domain = optimal_learning_base.domain.TensorProductDomain(hyperparameter_domain_bounds)

    # Suggestion for Newton
    # TODO(eliu): verify these
    num_multistarts_newton = 200
    max_num_steps_newton = 100
    gamma_newton = 1.05
    time_factor_newton = 1.0e-2
    max_relative_change_newton = 1.0
    tolerance_newton = 1.0e-9
    hyperopt_newton_parameters = optimal_learning_base.optimization.NewtonParameters(
        num_multistarts_newton,
        max_num_steps_newton,
        gamma_newton,
        time_factor_newton,
        max_relative_change_newton,
        tolerance_newton,
    )

    # Suggestion for Gradient Descent
    # Note: if using python_version.optimization.GradientDescentOptimizer, these parameters won't be the best
    # TODO(eliu): verify these
    num_multistarts_gd = 300
    max_num_steps_gd = 400
    max_num_restarts_gd = 10
    gamma_gd = 0.7
    pre_mult_gd = 0.4
    max_relative_change_gd = 0.1
    tolerance_gd = 1.0e-6
    hyperopt_gradient_descent_parameters = optimal_learning_base.optimization.GradientDescentParameters(
        num_multistarts_gd,
        max_num_steps_gd,
        max_num_restarts_gd,
        gamma_gd,
        pre_mult_gd,
        max_relative_change_gd,
        tolerance_gd,
    )

    # 4) EI Optimization Parameters
    # Could also use constant.default_ei_optimization_parameters
    # Although we should update those defaults and REMOVE num_mc_iterations from that list
    num_multistarts_ei = 200  # this will be REALLY slow if you hit monte-carlo
    max_num_steps_ei = 400
    max_num_restarts_ei = 10
    gamma_ei = 0.7
    pre_mult_ei = 1.0
    max_relative_change_ei = 0.9
    tolerance_ei = 1.0e-7
    ei_gradient_descent_parameters = optimal_learning_base.optimization.GradientDescentParameters(
        num_multistarts_ei,
        max_num_steps_ei,
        max_num_restarts_ei,
        gamma_ei,
        pre_mult_ei,
        max_relative_change_ei,
        tolerance_ei,
    )
    num_random_samples_ei = constant.default_num_random_samples  # 4000

    # 5) EI Parameters
    num_mc_iterations = constant.default_expected_improvement_parameters.mc_iterations  # 100000

    # 5.5) C++ randomness
    num_threads = 8
    # Set randomness to None if don't care; I'm setting it here to make things repeatable
    # Reasons to set randomness explicitly:
    # 1) repeatability (for tests)
    # 2) consistent RNG state across calls to C++
    # Not sure if this matters/is practical across the REST interface
    randomness = cpp_optimal_learning.RandomnessSourceContainer(num_threads)
    randomness.SetExplicitUniformGeneratorSeed(314)
    randomness.SetExplicitNormalRNGSeed(314)

    # 6) Optimize hyperparameters
    # You can skip this; the required "output" from this phase is a set of hyperparameters
    log_likelihood_eval = optimal_learning_base.log_likelihood.GaussianProcessLogLikelihood(
        covariance,
        historical_data,
    )
    log_likelihood_optimizer = optimal_learning_base.optimization.NewtonOptimizer(
        hyperparameter_domain,
        log_likelihood_eval,
        hyperopt_newton_parameters,
        num_random_samples=0,  # hyperopt doesn't use dumb search if optimization fails
    )
    # GD is an option too, but slower
    # log_likelihood_optimizer = optimal_learning_base.optimization.GradientDescentOptimizer(
    #     hyperparameter_domain,
    #     log_likelihood_eval,
    #     hyperopt_gradient_descent_parameters,
    #     num_random_samples=0,  # hyperopt doesn't use dumb search if optimization fails
    # )

    hyperopt_status = {}
    optimized_hyperparameters = optimal_learning_base.log_likelihood.multistart_hyperparameter_optimization(
        log_likelihood_optimizer,
        hyperopt_newton_parameters.num_multistarts,
        randomness=randomness,
        max_num_threads=num_threads,
        status=hyperopt_status,
    )
    log_likelihood_eval.set_current_point(optimized_hyperparameters)
    print 'log likelihood = %s' % log_likelihood_eval.compute_log_likelihood()
    print 'grad log likelihood = %s' % log_likelihood_eval.compute_grad_log_likelihood()
    print 'hyperparameters     = %s' % optimized_hyperparameters
    print hyperopt_status

    # 7) Construct GP
    covariance_optimized = covariance_class(optimized_hyperparameters)
    gaussian_process = optimal_learning_base.gaussian_process.GaussianProcess(
        covariance_optimized,
        historical_data,
    )

    # 8) Optimize EI
    # set points_to_sample (normally this comes from the input; here we generate randomly)
    points_to_sample = geometry_utils.generate_latin_hypercube_points(num_to_sample, domain._domain_bounds)
    current_point = [0.0] * domain.dim  # arbitrary
    ei_eval = optimal_learning_base.expected_improvement.ExpectedImprovement(
        gaussian_process,
        current_point,
        points_to_sample=points_to_sample,
        num_mc_iterations=num_mc_iterations,
    )
    ei_optimizer = optimal_learning_base.optimization.GradientDescentOptimizer(
        domain,
        ei_eval,
        ei_gradient_descent_parameters,
        num_random_samples=num_random_samples_ei,
    )

    # number of samples you want to optimize
    num_samples_to_generate = 1
    ei_status = {}
    new_points = optimal_learning_base.expected_improvement.multistart_expected_improvement_optimization(
        ei_optimizer,
        ei_gradient_descent_parameters.num_multistarts,
        num_samples_to_generate,
        randomness=randomness,
        max_num_threads = num_threads,
        status=ei_status,
    )
    ei_eval.set_current_point(new_points)
    print ' ei        = %s' % ei_eval.compute_expected_improvement()
    print 'grad ei    = %s' % ei_eval.compute_grad_expected_improvement()
    print 'new points = %s' % new_points
    print ei_status

