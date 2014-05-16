# -*- coding: utf-8 -*-
"""Tests for the functions/classes in geometry_utils."""
import numpy
import testify as T

from moe.optimal_learning.python.geometry_utils import ClosedInterval, generate_grid_points, generate_latin_hypercube_points
from moe.optimal_learning.python.python_version.domain import TensorProductDomain
from moe.tests.optimal_learning.python.optimal_learning_test_case import OptimalLearningTestCase


class LatinHypercubeRandomPointGenerationTest(OptimalLearningTestCase):

    """Test moe.optimal_learning.python.geometry_utils.generate_latin_hypercube_points.

    http://en.wikipedia.org/wiki/Latin_hypercube_sampling

    From wikipedia:
        In the context of statistical sampling, a square grid containing sample positions
        is a Latin square if (and only if) there is only one sample in each row and each column.
        A Latin hypercube is the generalisation of this concept to an arbitrary number of dimensions,
        whereby each sample is the only one in each axis-aligned hyperplane containing it.

        When sampling a function of N variables, the range of each variable is divided into
        M equally probable intervals. M sample points are then placed to satisfy the Latin hypercube requirements;
        note that this forces the number of divisions, M, to be equal for each variable.
        Also note that this sampling scheme does not require more samples for more dimensions (variables);
        this independence is one of the main advantages of this sampling scheme.
        Another advantage is that random samples can be taken one at a time,
        remembering which samples were taken so far.

    """

    @T.class_setup
    def base_setup(self):
        """Set up parameters for test cases."""
        domain_bounds_to_test = [
            ClosedInterval.build_closed_intervals_from_list([[-1.0, 1.0]]),
            ClosedInterval.build_closed_intervals_from_list([[-10.0, 10.0]]),
            ClosedInterval.build_closed_intervals_from_list([[-500.0, -490.0]]),
            ClosedInterval.build_closed_intervals_from_list([[6000.0, 6000.001]]),
            ClosedInterval.build_closed_intervals_from_list([[-1.0, 1.0], [-1.0, 1.0]]),
            ClosedInterval.build_closed_intervals_from_list([[-1.0, 1.0], [-1.0, 1.0], [-1.0, 1.0]]),
            ClosedInterval.build_closed_intervals_from_list([[-7000.0, 10000.0], [-8000.0, -7999.0], [10000.06, 10000.0601]]),
        ]

        self.domains_to_test = [TensorProductDomain(domain_bounds) for domain_bounds in domain_bounds_to_test]
        self.num_points_to_test = [1, 2, 5, 10, 20]

    def test_latin_hypercube_within_domain(self):
        """Test that generate_latin_hypercube_points returns points within the domain."""
        for domain in self.domains_to_test:
            for num_points in self.num_points_to_test:
                points = generate_latin_hypercube_points(num_points, domain._domain_bounds)

                for point in points:
                    T.assert_equal(domain.check_point_inside(point), True)

    def test_make_rand_point_within_domain(self):
        """Test that domain.generate_random_point_in_domain returns a point in the domain."""
        for domain in self.domains_to_test:
            for _ in xrange(10):
                point = domain.generate_random_point_in_domain()
                T.assert_equal(domain.check_point_inside(point), True)

    def test_latin_hypercube_equally_spaced(self):
        """Test that generate_latin_hypercube_points returns properly spaced points.

        Sampling from a latin hypercube results in a set of points that in each dimension are drawn
        uniformly from sub-intervals of the domain this tests that every sub-interval in each dimension
        contains exactly one point.

        """
        for domain in self.domains_to_test:
            for num_points in self.num_points_to_test:
                domain_bounds = domain._domain_bounds
                points = generate_latin_hypercube_points(num_points, domain_bounds)

                for dim in xrange(domain.dim):
                    # This size of each slice
                    sub_domain_width = domain_bounds[dim].length / float(num_points)
                    # Sort in dim dimension
                    points = sorted(points, key=lambda points: points[dim])
                    for i, point in enumerate(points):
                        # This point must fall somewhere within the slice
                        min_val = domain_bounds[dim].min + sub_domain_width * i
                        max_val = min_val + sub_domain_width
                        T.assert_gte(point[dim], min_val)
                        T.assert_lte(point[dim], max_val)


class GridPointGenerationTest(OptimalLearningTestCase):

    """Test the generation of an evenly spaced, axis-aligned grid on a hypercube."""

    def test_grid_generation(self):
        """Test that ``generate_grid_points`` generates a uniform grid.

        Test makes assumptions about the ordering of the output that may be invalidated by
        changes to numpy.meshgrid.

        """
        domain_bounds = ClosedInterval.build_closed_intervals_from_list([[0.0, 1.0], [-2.0, 3.0], [2.71, 3.14]])
        points_per_dimension = [7, 11, 8]

        # Test that all points are present
        grid = generate_grid_points(points_per_dimension, domain_bounds)

        per_axis_grid = [numpy.linspace(bounds.min, bounds.max, points_per_dimension[i])
                         for i, bounds in enumerate(domain_bounds)]

        # Loop ordering assumes the output is ordered a certain way.
        for i, y_coord in enumerate(per_axis_grid[1]):
            for j, x_coord in enumerate(per_axis_grid[0]):
                for k, z_coord in enumerate(per_axis_grid[2]):
                    coordinates_truth = numpy.array([x_coord, y_coord, z_coord])
                    index = i * len(per_axis_grid[2]) * len(per_axis_grid[0]) + j * len(per_axis_grid[2]) + k
                    test = grid[index, ...]
                    self.assert_vector_within_relative(test, coordinates_truth, 0.0)

        # Also test that scalar points_per_dimension works
        points_per_dimension = [5, 5, 5]
        grid_truth = generate_grid_points(points_per_dimension, domain_bounds)

        points_per_dimension = 5
        grid_test = generate_grid_points(points_per_dimension, domain_bounds)

        self.assert_vector_within_relative(grid_test, grid_truth, 0.0)


class ClosedIntervalTest(OptimalLearningTestCase):

    """Tests for ClosedInterval's member functions."""

    @T.class_setup
    def base_setup(self):
        """Set up test cases (described inline)."""
        self.test_cases = [
            ClosedInterval(9.378, 9.378),    # min == max
            ClosedInterval(-2.71, 3.14),     # min < max
            ClosedInterval(-2.71, -3.14),    # min > max
            ClosedInterval(0.0, numpy.inf),  # infinte range
        ]

        self.points_to_check = numpy.empty((len(self.test_cases), 5))
        for i, case in enumerate(self.test_cases):
            self.points_to_check[i, 0] = (case.min + case.max) * 0.5  # midpoint
            self.points_to_check[i, 1] = case.min                     # left boundary
            self.points_to_check[i, 2] = case.max                     # right boundary
            self.points_to_check[i, 3] = case.min - 0.5               # outside on the left
            self.points_to_check[i, 4] = case.max + 0.5               # outside on the right

    def test_length(self):
        """Check that length works."""
        expected_length_list = [0.0, self.test_cases[1].max - self.test_cases[1].min, self.test_cases[2].max - self.test_cases[2].min, numpy.inf]
        for i, case in enumerate(self.test_cases):
            T.assert_equal(case.length, expected_length_list[i])

    def test_is_inside(self):
        """Check that is_inside works."""
        points_inside_test_interval_0_truth = [True, True, True, False, False]
        case = 0
        for j, value in enumerate(self.points_to_check[case, ...]):
            T.assert_equal(self.test_cases[case].is_inside(value), points_inside_test_interval_0_truth[j])

        points_inside_test_interval_1_truth = [True, True, True, False, False]
        case = 1
        for j, value in enumerate(self.points_to_check[case, ...]):
            T.assert_equal(self.test_cases[case].is_inside(value), points_inside_test_interval_1_truth[j])

        points_inside_test_interval_2_truth = [False, False, False, False, False]
        case = 2
        for j, value in enumerate(self.points_to_check[case, ...]):
            T.assert_equal(self.test_cases[case].is_inside(value), points_inside_test_interval_2_truth[j])

        points_inside_test_interval_3_truth = [True, True, True, False, True]
        case = 3
        for j, value in enumerate(self.points_to_check[case, ...]):
            T.assert_equal(self.test_cases[case].is_inside(value), points_inside_test_interval_3_truth[j])

    def test_is_empty(self):
        """Check that is_empty works."""
        test_interval_empty_truth = [False, False, True, False]
        for i, case in enumerate(self.test_cases):
            T.assert_equal(case.is_empty(), test_interval_empty_truth[i])


if __name__ == "__main__":
    T.run()
