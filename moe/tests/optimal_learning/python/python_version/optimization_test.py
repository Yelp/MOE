# -*- coding: utf-8 -*-
"""Tests for the Python optimization module (null, gradient descent, and multistarting) using a simple polynomial objective."""
import numpy

import testify as T

from moe.optimal_learning.python.geometry_utils import ClosedInterval
from moe.optimal_learning.python.interfaces.optimization_interface import OptimizableInterface
from moe.optimal_learning.python.python_version.domain import TensorProductDomain
from moe.optimal_learning.python.python_version.optimization import multistart_optimize, LBFGSBParameters, GradientDescentParameters, NullOptimizer, GradientDescentOptimizer, MultistartOptimizer, LBFGSBOptimizer, ConstrainedDFOOptimizer, ConstrainedDFOParameters
from moe.tests.optimal_learning.python.optimal_learning_test_case import OptimalLearningTestCase


class QuadraticFunction(OptimizableInterface):

    r"""Class to evaluate the function f(x_1,...,x_{dim}) = -\sum_i (x_i - s_i)^2, i = 1..dim.

    This is a simple quadratic form with maxima at (s_1, ..., s_{dim}).

    """

    def __init__(self, maxima_point, current_point):
        """Create an instance of QuadraticFunction with the specified maxima."""
        self._maxima_point = numpy.copy(maxima_point)
        self._current_point = numpy.copy(current_point)

    @property
    def dim(self):
        """Return the number of spatial dimensions."""
        return self._maxima_point.size

    @property
    def problem_size(self):
        """Return the number of independent parameters to optimize."""
        return self.dim

    @property
    def optimum_value(self):
        """Return max_x f(x), the global maximum value of this function."""
        return 0.0

    @property
    def optimum_point(self):
        """Return the argmax_x (f(x)), the point at which the global maximum occurs."""
        return numpy.copy(self._maxima_point)

    def get_current_point(self):
        """Get the current_point (array of float64 with shape (problem_size)) at which this object is evaluating the objective function, ``f(x)``."""
        return numpy.copy(self._current_point)

    def set_current_point(self, current_point):
        """Set current_point to the specified point; ordering must match.

        :param current_point: current_point at which to evaluate the objective function, ``f(x)``
        :type current_point: array of float64 with shape (problem_size)

        """
        self._current_point = numpy.copy(current_point)

    current_point = property(get_current_point, set_current_point)

    def compute_objective_function(self, **kwargs):
        r"""Compute ``f(current_point)``.

        :return: value of objective function evaluated at ``current_point``
        :rtype: float64

        """
        temp = self._current_point - self._maxima_point
        temp *= temp
        return -temp.sum()

    def compute_grad_objective_function(self, **kwargs):
        r"""Compute the gradient of ``f(current_point)`` wrt ``current_point``.

        :return: gradient of the objective, i-th entry is ``\pderiv{f(x)}{x_i}``
        :rtype: array of float64 with shape (problem_size)

        """
        return -2.0 * (self._current_point - self._maxima_point)

    def compute_hessian_objective_function(self, **kwargs):
        r"""Compute the hessian matrix of ``f(current_point)`` wrt ``current_point``.

        This matrix is symmetric as long as the mixed second derivatives of f(x) are continuous: Clairaut's Theorem.
        http://en.wikipedia.org/wiki/Symmetry_of_second_derivatives

        :return: hessian of the objective, (i,j)th entry is ``\mixpderiv{f(x)}{x_i}{x_j}``
        :rtype: array of float64 with shape (problem_size, problem_size)

        """
        return numpy.diag(numpy.full(self.dim, -2.0))


class NullOptimizerTest(OptimalLearningTestCase):

    """Test the NullOptimizer on a simple objective.

    NullOptimizer should do nothing.
    Multistarting it should be the same as a 'dumb' search over points.

    """

    @T.class_setup
    def base_setup(self):
        """Set up a test case for optimizing a simple quadratic polynomial."""
        self.dim = 3
        domain_bounds = [ClosedInterval(-1.0, 1.0)] * self.dim
        self.domain = TensorProductDomain(domain_bounds)

        maxima_point = numpy.full(self.dim, 0.5)
        current_point = numpy.zeros(self.dim)
        self.polynomial = QuadraticFunction(maxima_point, current_point)
        self.null_optimizer = NullOptimizer(self.domain, self.polynomial)

    def test_null_optimizer(self):
        """Test that null optimizer does not change current_point."""
        current_point_old = self.null_optimizer.objective_function.current_point
        self.null_optimizer.optimize()
        current_point_new = self.null_optimizer.objective_function.current_point
        self.assert_vector_within_relative(current_point_old, current_point_new, 0.0)

    def test_multistarted_null_optimizer(self):
        """Test that multistarting null optimizer just evalutes the function and indentifies the max."""
        num_points = 15
        points = self.domain.generate_uniform_random_points_in_domain(num_points)

        truth = numpy.empty(num_points)
        for i, point in enumerate(points):
            self.null_optimizer.objective_function.current_point = point
            truth[i] = self.null_optimizer.objective_function.compute_objective_function()

        best_index = numpy.argmax(truth)
        truth_best_point = points[best_index, ...]

        test_best_point, test_values = multistart_optimize(self.null_optimizer, starting_points=points)

        self.assert_vector_within_relative(test_best_point, truth_best_point, 0.0)
        self.assert_vector_within_relative(test_values, truth, 0.0)


class GradientDescentOptimizerTest(OptimalLearningTestCase):

    r"""Test Gradient Descent on a simple quadratic objective.

    We check GD in an unconstrained setting, a constrained setting, and we test multistarting it.

    We don't test the stochastic averaging option meaningfully. We check that the optimizer will average
    the number of steps specified by input. We also check that the simple unconstrained case can also be solved
    with averaging on\*.

    \* This is not much of a test. The problem is convex and isotropic so GD will take a more or less straight
    path to the maxima. Averaging can only reduce the accuracy of the solve.

    TODO(GH-179): Build a simple stochastic objective and test the stochastic component fully.

    """

    @T.class_setup
    def base_setup(self):
        """Set up a test case for optimizing a simple quadratic polynomial."""
        self.dim = 3
        domain_bounds = [ClosedInterval(-1.0, 1.0)] * self.dim
        self.domain = TensorProductDomain(domain_bounds)

        maxima_point = numpy.full(self.dim, 0.5)
        current_point = numpy.zeros(self.dim)
        self.polynomial = QuadraticFunction(maxima_point, current_point)

        max_num_steps = 250
        max_num_restarts = 10
        num_steps_averaged = 0
        gamma = 0.7  # smaller gamma would lead to faster convergence, but we don't want to make the problem too easy
        pre_mult = 1.0
        max_relative_change = 0.8
        tolerance = 1.0e-12
        self.gd_parameters = GradientDescentParameters(
            max_num_steps,
            max_num_restarts,
            num_steps_averaged,
            gamma,
            pre_mult,
            max_relative_change,
            tolerance,
        )

        approx_grad = False
        max_func_evals = 150000
        max_metric_correc = 10
        factr = 1000.0
        pgtol = 1e-10
        epsilon = 1e-8
        self.BFGS_parameters = LBFGSBParameters(
            approx_grad,
            max_func_evals,
            max_metric_correc,
            factr,
            pgtol,
            epsilon,
        )

        maxfun = 1000
        rhobeg = 1.0
        rhoend = 1.0e-13
        catol = 2.0e-13
        self.COBYLA_parameters = ConstrainedDFOParameters(
            rhobeg,
            rhoend,
            maxfun,
            catol,
        )

    def test_gradient_descent_optimizer(self):
        """Check that gradient descent can find the optimum of the quadratic test objective."""
        # Check the claimed optima is an optima
        optimum_point = self.polynomial.optimum_point
        self.polynomial.current_point = optimum_point
        gradient = self.polynomial.compute_grad_objective_function()
        self.assert_vector_within_relative(gradient, numpy.zeros(self.polynomial.dim), 0.0)

        # Verify that gradient descent does not move from the optima if we start it there.
        gradient_descent_optimizer = GradientDescentOptimizer(self.domain, self.polynomial, self.gd_parameters)
        gradient_descent_optimizer.optimize()
        output = gradient_descent_optimizer.objective_function.current_point
        self.assert_vector_within_relative(output, optimum_point, 0.0)

        # Start at a wrong point and check optimization
        tolerance = 2.0e-13
        initial_guess = numpy.full(self.polynomial.dim, 0.2)
        gradient_descent_optimizer.objective_function.current_point = initial_guess
        gradient_descent_optimizer.optimize()
        output = gradient_descent_optimizer.objective_function.current_point
        # Verify coordinates
        self.assert_vector_within_relative(output, optimum_point, tolerance)

        # Verify function value
        value = self.polynomial.compute_objective_function()
        self.assert_scalar_within_relative(value, self.polynomial.optimum_value, tolerance)

        # Verify derivative
        gradient = self.polynomial.compute_grad_objective_function()
        self.assert_vector_within_relative(gradient, numpy.zeros(self.polynomial.dim), tolerance)

    def test_get_averaging_range(self):
        """Test the method used to produce what interval to average over in Polyak-Ruppert averaging."""
        num_steps_total = 250
        end = num_steps_total + 1
        num_steps_averaged_input_list = [-1, 0, 1, 20, 100, 249, 250, 251, 10000]
        truth_list = [(1, end), (250, end), (250, end), (231, end), (151, end), (2, end), (1, end), (1, end), (1, end)]

        for i, truth in enumerate(truth_list):
            start, end = GradientDescentOptimizer._get_averaging_range(num_steps_averaged_input_list[i], num_steps_total)
            T.assert_equal(start, truth[0])
            T.assert_equal(end, truth[1])

    def test_gradient_descent_optimizer_with_averaging(self):
        """Check that gradient descent can find the optimum of the quadratic test objective with averaging on.

        This test doesn't exercise the purpose of averaging (i.e., this objective isn't stochastic), but it does
        check that it at least runs.

        """
        num_steps_averaged = self.gd_parameters.max_num_steps * 3 / 4
        gd_parameters_averaging = GradientDescentParameters(
            self.gd_parameters.max_num_steps,
            self.gd_parameters.max_num_restarts,
            num_steps_averaged,
            self.gd_parameters.gamma,
            self.gd_parameters.pre_mult,
            self.gd_parameters.max_relative_change,
            self.gd_parameters.tolerance,
        )
        # Check the claimed optima is an optima
        optimum_point = self.polynomial.optimum_point
        self.polynomial.current_point = optimum_point
        gradient = self.polynomial.compute_grad_objective_function()
        self.assert_vector_within_relative(gradient, numpy.zeros(self.polynomial.dim), 0.0)

        # Verify that gradient descent does not move from the optima if we start it there.
        gradient_descent_optimizer = GradientDescentOptimizer(self.domain, self.polynomial, gd_parameters_averaging)
        gradient_descent_optimizer.optimize()
        output = gradient_descent_optimizer.objective_function.current_point
        self.assert_vector_within_relative(output, optimum_point, 0.0)

        # Start at a wrong point and check optimization
        tolerance = 2.0e-10
        initial_guess = numpy.full(self.polynomial.dim, 0.2)
        gradient_descent_optimizer.objective_function.current_point = initial_guess
        gradient_descent_optimizer.optimize()
        output = gradient_descent_optimizer.objective_function.current_point
        # Verify coordinates
        self.assert_vector_within_relative(output, optimum_point, tolerance)

        # Verify function value
        value = self.polynomial.compute_objective_function()
        self.assert_scalar_within_relative(value, self.polynomial.optimum_value, tolerance)

        # Verify derivative
        gradient = self.polynomial.compute_grad_objective_function()
        self.assert_vector_within_relative(gradient, numpy.zeros(self.polynomial.dim), tolerance)

    def test_gradient_descent_optimizer_constrained(self):
        """Check that gradient descent can find the global optimum (in a domain) when the true optimum is outside."""
        # Domain where the optimum, (0.5, 0.5, 0.5), lies outside the domain
        domain_bounds = [ClosedInterval(0.05, 0.32), ClosedInterval(0.05, 0.6), ClosedInterval(0.05, 0.32)]
        domain = TensorProductDomain(domain_bounds)
        gradient_descent_optimizer = GradientDescentOptimizer(domain, self.polynomial, self.gd_parameters)

        # Work out what the maxima point woudl be given the domain constraints (i.e., project to the nearest point on domain)
        constrained_optimum_point = self.polynomial.optimum_point
        for i, bounds in enumerate(domain_bounds):
            if constrained_optimum_point[i] > bounds.max:
                constrained_optimum_point[i] = bounds.max
            elif constrained_optimum_point[i] < bounds.min:
                constrained_optimum_point[i] = bounds.min

        tolerance = 2.0e-13
        initial_guess = numpy.full(self.polynomial.dim, 0.2)
        gradient_descent_optimizer.objective_function.current_point = initial_guess
        initial_value = gradient_descent_optimizer.objective_function.compute_objective_function()
        gradient_descent_optimizer.optimize()
        output = gradient_descent_optimizer.objective_function.current_point
        # Verify coordinates
        self.assert_vector_within_relative(output, constrained_optimum_point, tolerance)

        # Verify optimized value is better than initial guess
        final_value = self.polynomial.compute_objective_function()
        T.assert_gt(final_value, initial_value)

        # Verify derivative: only get 0 derivative if the coordinate lies inside domain boundaries
        gradient = self.polynomial.compute_grad_objective_function()
        for i, bounds in enumerate(domain_bounds):
            if bounds.is_inside(self.polynomial.optimum_point[i]):
                self.assert_scalar_within_relative(gradient[i], 0.0, tolerance)

    def test_multistarted_gradient_descent_optimizer_crippled_start(self):
        """Check that multistarted GD is finding the best result from GD."""
        # Only allow 1 GD iteration.
        gd_parameters_crippled = GradientDescentParameters(
            1,
            1,
            self.gd_parameters.num_steps_averaged,
            self.gd_parameters.gamma,
            self.gd_parameters.pre_mult,
            self.gd_parameters.max_relative_change,
            self.gd_parameters.tolerance,
        )
        gradient_descent_optimizer_crippled = GradientDescentOptimizer(self.domain, self.polynomial, gd_parameters_crippled)

        num_points = 15
        points = self.domain.generate_uniform_random_points_in_domain(num_points)

        multistart_optimizer = MultistartOptimizer(gradient_descent_optimizer_crippled, num_points)
        test_best_point, _ = multistart_optimizer.optimize(random_starts=points)
        # This point set won't include the optimum so multistart GD won't find it.
        for value in (test_best_point - self.polynomial.optimum_point):
            T.assert_not_equal(value, 0.0)

        points_with_opt = numpy.append(points, self.polynomial.optimum_point.reshape((1, self.polynomial.dim)), axis=0)
        test_best_point, _ = multistart_optimizer.optimize(random_starts=points_with_opt)
        # This point set will include the optimum so multistart GD will find it.
        for value in (test_best_point - self.polynomial.optimum_point):
            T.assert_equal(value, 0.0)

    def test_multistarted_gradient_descent_optimizer(self):
        """Check that multistarted GD can find the optimum in a 'very' large domain."""
        # Set a large domain: a single GD run is unlikely to reach the optimum
        domain_bounds = [ClosedInterval(-10.0, 10.0)] * self.dim
        domain = TensorProductDomain(domain_bounds)

        tolerance = 2.0e-10
        num_points = 10
        gradient_descent_optimizer = GradientDescentOptimizer(domain, self.polynomial, self.gd_parameters)
        multistart_optimizer = MultistartOptimizer(gradient_descent_optimizer, num_points)

        output, _ = multistart_optimizer.optimize()
        # Verify coordinates
        self.assert_vector_within_relative(output, self.polynomial.optimum_point, tolerance)

        # Verify function value
        value = self.polynomial.compute_objective_function()
        self.assert_scalar_within_relative(value, self.polynomial.optimum_value, tolerance)

        # Verify derivative
        gradient = self.polynomial.compute_grad_objective_function()
        self.assert_vector_within_relative(gradient, numpy.zeros(self.polynomial.dim), tolerance)

    def test_bfgs_optimizer(self):
        """Check that BFGS can find the optimum of the quadratic test objective."""
        # Check the claimed optima is an optima
        optimum_point = self.polynomial.optimum_point
        self.polynomial.current_point = optimum_point
        gradient = self.polynomial.compute_grad_objective_function()
        self.assert_vector_within_relative(gradient, numpy.zeros(self.polynomial.dim), 0.0)

        # Verify that BFGS does not move from the optima if we start it there.
        bfgs_optimizer = LBFGSBOptimizer(self.domain, self.polynomial, self.BFGS_parameters)
        bfgs_optimizer.optimize()
        output = bfgs_optimizer.objective_function.current_point
        self.assert_vector_within_relative(output, optimum_point, 0.0)

        # Start at a wrong point and check optimization
        tolerance = 2.0e-13
        initial_guess = numpy.full(self.polynomial.dim, 0.2)
        bfgs_optimizer.objective_function.current_point = initial_guess
        bfgs_optimizer.optimize()
        output = bfgs_optimizer.objective_function.current_point
        # Verify coordinates
        self.assert_vector_within_relative(output, optimum_point, tolerance)

        # Verify function value
        value = self.polynomial.compute_objective_function()
        self.assert_scalar_within_relative(value, self.polynomial.optimum_value, tolerance)

        # Verify derivative
        gradient = self.polynomial.compute_grad_objective_function()
        self.assert_vector_within_relative(gradient, numpy.zeros(self.polynomial.dim), tolerance)

    def test_multistarted_bfgs_optimizer(self):
        """Check that multistarted BFGS can find the optimum in a 'very' large domain."""
        # Set a large domain: a single BFGS run is unlikely to reach the optimum
        domain_bounds = [ClosedInterval(-10.0, 10.0)] * self.dim
        domain = TensorProductDomain(domain_bounds)

        tolerance = 2.0e-10
        num_points = 10
        bfgs_optimizer = LBFGSBOptimizer(domain, self.polynomial, self.BFGS_parameters)
        multistart_optimizer = MultistartOptimizer(bfgs_optimizer, num_points)

        output, _ = multistart_optimizer.optimize()
        # Verify coordinates
        self.assert_vector_within_relative(output, self.polynomial.optimum_point, tolerance)

        # Verify function value
        value = self.polynomial.compute_objective_function()
        self.assert_scalar_within_relative(value, self.polynomial.optimum_value, tolerance)

        # Verify derivative
        gradient = self.polynomial.compute_grad_objective_function()
        self.assert_vector_within_relative(gradient, numpy.zeros(self.polynomial.dim), tolerance)

    def test_cobyla_optimizer(self):
        """Check that COBYLA can find the optimum of the quadratic test objective."""
        # Check the claimed optima is an optima
        optimum_point = self.polynomial.optimum_point
        self.polynomial.current_point = optimum_point
        gradient = self.polynomial.compute_grad_objective_function()
        self.assert_vector_within_relative(gradient, numpy.zeros(self.polynomial.dim), 0.0)

        # Verify that COBYLA does not move from the optima if we start it there.
        tolerance = 2.0e-13
        constrained_optimizer = ConstrainedDFOOptimizer(self.domain, self.polynomial, self.COBYLA_parameters)
        constrained_optimizer.optimize()
        output = constrained_optimizer.objective_function.current_point
        self.assert_vector_within_relative(output, optimum_point, tolerance)

        # Start at a wrong point and check optimization
        initial_guess = numpy.full(self.polynomial.dim, 0.2)
        constrained_optimizer.objective_function.current_point = initial_guess
        constrained_optimizer.optimize()
        output = constrained_optimizer.objective_function.current_point
        # Verify coordinates
        self.assert_vector_within_relative(output, optimum_point, tolerance)

        # Verify function value
        value = self.polynomial.compute_objective_function()
        self.assert_scalar_within_relative(value, self.polynomial.optimum_value, tolerance)

        # Verify derivative
        gradient = self.polynomial.compute_grad_objective_function()
        self.assert_vector_within_relative(gradient, numpy.zeros(self.polynomial.dim), tolerance)

    def test_multistarted_cobyla_optimizer(self):
        """Check that multistarted COBYLA can find the optimum in a 'very' large domain."""
        # Set a large domain: a single COBYLA run is unlikely to reach the optimum
        domain_bounds = [ClosedInterval(-10.0, 10.0)] * self.dim
        domain = TensorProductDomain(domain_bounds)

        tolerance = 2.0e-10
        num_points = 10
        constrained_optimizer = ConstrainedDFOOptimizer(domain, self.polynomial, self.COBYLA_parameters)
        multistart_optimizer = MultistartOptimizer(constrained_optimizer, num_points)

        output, _ = multistart_optimizer.optimize()
        # Verify coordinates
        self.assert_vector_within_relative(output, self.polynomial.optimum_point, tolerance)

        # Verify function value
        value = self.polynomial.compute_objective_function()
        self.assert_scalar_within_relative(value, self.polynomial.optimum_value, tolerance)

        # Verify derivative
        gradient = self.polynomial.compute_grad_objective_function()
        self.assert_vector_within_relative(gradient, numpy.zeros(self.polynomial.dim), tolerance)
