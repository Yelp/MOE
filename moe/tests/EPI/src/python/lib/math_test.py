# -*- coding: utf-8 -*-

import testify as T
import numpy

import moe.optimal_learning.EPI.src.python.lib.math


MACHINE_PRECISION = 1e-13

class PointGenerationTest(T.TestCase):
    """Test optimal_learning.EPI.src.python.lib.math.get_latin_hypercube_points(num_points, domains)
    also tests optimal_learning.EPI.src.python.lib.math.not_in_domain(point, domain)

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

    domains_to_test = [
            [[-1, 1]],
            [[-10, 10]],
            [[-500, -490]],
            [[6000, 6000.001]],
            [[-1, 1], [-1, 1]],
            [[-1, 1], [-1, 1], [-1, 1]],
            [[-7000, 10000], [-8000, -7999], [10000.06, 10000.0601]],
            ]

    num_points_to_test = [1, 2, 5, 10, 20]

    def _point_in_domain(self, point, domain):
        """Returns true if a point is in a specified domain
        """
        for i, sub_domain in enumerate(domain):
            if point[i] > sub_domain[1]:
                return False
            elif point[i] < sub_domain[0]:
                return False
        return True

    def test_latin_hypercube_within_domain(self):
        """Tests that get_latin_hypercube_points returns points within the domain
        """
        for domain in self.domains_to_test:
            for num_points in self.num_points_to_test:
                points = moe.optimal_learning.EPI.src.python.lib.math.get_latin_hypercube_points(num_points, domain)

                for point in points:
                    T.assert_equal(self._point_in_domain(point, domain), True)
                    T.assert_equal(moe.optimal_learning.EPI.src.python.lib.math.not_in_domain(point, domain), False)

    def test_make_rand_point_within_domain(self):
        """Tests that make_rand_point(domain) returns a point in the domain
        """
        for domain in self.domains_to_test:
            point = moe.optimal_learning.EPI.src.python.lib.math.make_rand_point(domain)

            T.assert_equal(self._point_in_domain(point, domain), True)
            T.assert_equal(moe.optimal_learning.EPI.src.python.lib.math.not_in_domain(point, domain), False)

    def test_latin_hypercube_equally_spaced(self):
        """Sampling from a latin hypercube results in a set of points
        that in each dimension are drawn uniformly from sub-intervals of the domain
        this tests that every sub-interval in each dimension contains exactly one point
        """
        for domain in self.domains_to_test:
            for num_points in self.num_points_to_test:
                points = moe.optimal_learning.EPI.src.python.lib.math.get_latin_hypercube_points(num_points, domain)

                for dim in range(len(domain)):
                    # This size of each slice
                    sub_domain_width = (domain[dim][1] - domain[dim][0])/(1.0 * num_points) # Weak types suck sometimes
                    # Sort in dim dimension
                    points = sorted(points, key=lambda points: points[dim])
                    for i, point in enumerate(points):
                        # This point must fall somewhere within the slice
                        min_val = domain[dim][0] + sub_domain_width*i
                        max_val = min_val + sub_domain_width
                        T.assert_gte(point[dim], min_val)
                        T.assert_lte(point[dim], max_val)

class MatrixOpsTest(T.TestCase):
    """Test the various matrix ops in optimal_learning.EPI.src.python.lib.math

    Tests:
        vector_diag_vector_product
        inverse_via_backwards_sub
        cholesky_decomp
        inverse_via_cholesky
    """

    def _assert_matrix_equal(self, matrix, expected_matrix):
        """Asserts a matrix is within MACHINE_PRECISION of the identity matrix
        """
        size_of_matrix = len(matrix)
        for i in range(size_of_matrix):
            for j in range(size_of_matrix):
                T.assert_lte(
                        numpy.abs(matrix[i][j] - expected_matrix[i][j]),
                        MACHINE_PRECISION
                        )

    def test_cholesky_decomp(self):
        """Tests cholesky_decomp
        """
        for size_of_matrix in range(2,10):
            true_cholesky = numpy.tril(
                    numpy.random.uniform(1, 2, size=(size_of_matrix,size_of_matrix))
                    )
            matrix = numpy.dot(true_cholesky, true_cholesky.T)
            calculated_cholesky = moe.optimal_learning.EPI.src.python.lib.math.cholesky_decomp(matrix)
            self._assert_matrix_equal(
                    calculated_cholesky - true_cholesky,
                    numpy.zeros(shape=(size_of_matrix, size_of_matrix))
                    )


    def test_inverse_via_cholesky(self):
        """This tests inverse_via_cholesky and cholesky_decomp
        """
        for size_of_matrix in range(2,10):
            true_cholesky = numpy.tril(
                    numpy.random.uniform(1, 2, size=(size_of_matrix, size_of_matrix))
                    )
            A = numpy.dot(true_cholesky, true_cholesky.T)
            A_inv = moe.optimal_learning.EPI.src.python.lib.math.inverse_via_cholesky(A)
            self._assert_matrix_equal(
                    numpy.dot(A, A_inv),
                    numpy.diag(numpy.ones(size_of_matrix))
                    )

    def test_inverse_via_backwards_sub(self):
        """Tests inverse_via_backwards_sub
        """
        for size_of_matrix in range(2,10):
            A = numpy.tril(numpy.random.uniform(1, 2, size=(size_of_matrix,size_of_matrix)))
            A_inv = moe.optimal_learning.EPI.src.python.lib.math.inverse_via_backwards_sub(A)
            self._assert_matrix_equal(
                    numpy.dot(A, A_inv),
                    numpy.diag(numpy.ones(size_of_matrix))
                    )

    def test_vector_diag_vector_product(self):
        """Tests test_vector_diag_vector_product by doing the full matrix ops
        """
        sizes_to_test = [2, 5, 10, 20]

        for size in sizes_to_test:
            rand_vec_one = numpy.random.normal(size=size)
            rand_vec_two = numpy.random.normal(size=size)
            rand_diag_vec = numpy.random.normal(size=size)
            rand_diag_matrix = numpy.diag(rand_diag_vec)

            method_result = moe.optimal_learning.EPI.src.python.lib.math.vector_diag_vector_product(
                    rand_vec_one,
                    rand_diag_vec,
                    rand_vec_two
                    )
            numpy_result = numpy.dot(
                    (rand_vec_one - rand_vec_two),
                    numpy.dot(
                        rand_diag_matrix,
                        (rand_vec_one - rand_vec_two).T
                        )
                    )

            T.assert_lte(
                    (method_result - numpy_result)/numpy_result,
                    MACHINE_PRECISION
                    )

if __name__ == "__main__":
    T.run()
