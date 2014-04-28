# -*- coding: utf-8 -*-

import numpy # for sci comp
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

def vector_diag_vector_product(vec_one, diag_vec, vec_two):
    """Computes (vec_one - vec_two) * diag_matrix * (vec_one - vec_two).T
    """
    result = 0.0
    diag_val = False
    if len(vec_one) > 1 and len(diag_vec) == 1:
        diag_val = diag_vec[0]
    for i, vec_one_point in enumerate(vec_one):
        vec_two_point = vec_two[i]
        if diag_val:
            result += (vec_one_point - vec_two_point) * (vec_one_point - vec_two_point) * diag_val
        else:
            result += (vec_one_point - vec_two_point) * (vec_one_point - vec_two_point) * diag_vec[i]
    return result

def get_latin_hypercube_points(num_points, domains):
    """Compute points to sample from a latin hypercube

    :args:
        - num_points: number of samples
        - domains: a list of tuples from which to sample ex. [(0,1),(0,1)] (the unit square)

    :Returns: list of numpy arrays corresponding to points to be sampled"""

    points = numpy.zeros((num_points, len(domains)), dtype=numpy.float64)

    for domain_on, domain in enumerate(domains):
        # The range we are conerned with in this subspace
        assert(len(domain) == 2)

        # Cut the range into num_points slices
        hypercube_domain_width = (domain[1] - domain[0]) / float(num_points)

        # Create random ordering for slices
        ordering = numpy.arange(num_points)
        numpy.random.shuffle(ordering)

        for point_on in range(num_points):
            point_base = domain[0] + (hypercube_domain_width * ordering[point_on])
            random_point = point_base + numpy.random.uniform(0, hypercube_domain_width)
            points[point_on][domain_on] = random_point

    logging.debug("Generated %i latin hypercube points from %s:\n%s" % (num_points, domains, points))

    return points

def not_in_domain(point, domain):
    """Return true if *point* is not in the *domain*

    :Returns: Boolean
    """
    for i, component in enumerate(point):
        if component < numpy.min(domain[i]):
            return True
        if component > numpy.max(domain[i]):
            return True
    return False

def make_empty_2D_list(size_one, size_two):
    """Make a two (3) dim empty list"""
    the_list = []
    for i in range(size_one):
        the_list.append([])
        for j in range(size_two):
            the_list[-1].append([])
    return the_list

def make_empty_1D_list(size_one):
    """Make a 1 (2) dim empty list"""
    the_list = []
    for i in range(size_one):
        the_list.append([])
    return the_list

def matrix_vector_multiply(matrix, vector):
    """Multiply nested arrays matrix_one and matrix_two"""
    answer = []
    for row in matrix:
        answer_row = numpy.zeros(len(row[0]))
        for i, element in enumerate(row):
            answer_row += element*vector[i]
        answer.append(answer_row)
    return numpy.array(answer)

def make_rand_point(domain):
    """Returns a random point in *domain*

    :Returns: 1-D numpy array
    """
    assert(len(domain) > 0)
    point = []

    try:
        len(domain[0])
        # multi-D
        for component in domain:
            point.append(numpy.random.uniform(low = numpy.min(component), high = numpy.max(component)))
    except: # 1-D
        point.append(numpy.random.uniform(low = numpy.min(domain), high = numpy.max(domain)))

    return numpy.array(point)

def inverse_via_backwards_sub(matrix):
    """Get the inverse of a triangular matrix via backwards substitution O(n^3)

    via Cormen (CLRS)

    :Return: n-D numpy array
    """
    mat_size = len(matrix)
    inv_mat = numpy.zeros(shape=(mat_size, mat_size))
    for inv_col in range(mat_size):
        for inv_row in range(mat_size):
            if inv_col == inv_row:
                inv_mat[inv_row][inv_col] = 1.0
            else:
                inv_mat[inv_row][inv_col] = 0.0
            for back_col in range(inv_row):
                inv_mat[inv_row][inv_col] -= matrix[inv_row][back_col]*inv_mat[back_col][inv_col]
            inv_mat[inv_row][inv_col] /= matrix[inv_row][inv_row]
    return inv_mat

def cholesky_decomp(matrix, eps=0.000001):
    """Get the cholesky decomposition of the variance

    see Smith 1995 or Trefethen, Bau 1997 or Golub, Van Loan 1983

    :Returns: 2-D numpy array
    """
    cholesky_decomp = matrix.copy() # Just to start!

    # TODO make more pythonic
    # zero out the upper half of the matrix
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            if i < j:
                cholesky_decomp[i][j] = 0.0

    # Step 2 of Appendix 2
    for k in range(len(matrix)):
        if cholesky_decomp[k][k] > eps:
            cholesky_decomp[k][k] = numpy.sqrt(abs(cholesky_decomp[k][k]))
            for j in range(k+1, len(matrix)):
                cholesky_decomp[j][k] = cholesky_decomp[j][k]/cholesky_decomp[k][k]
            for j in range(k+1, len(matrix)):
                for i in range(j, len(matrix)):
                    cholesky_decomp[i][j] = cholesky_decomp[i][j] - cholesky_decomp[i][k]*cholesky_decomp[j][k]

    return cholesky_decomp

def inverse_via_cholesky(matrix, eps=0.000001):
    """Get the inverse of a pos, semi-def matrix via a cholesky decomp O(n^3)

    :Returns: 2-D numpy array
    """
    L = cholesky_decomp(matrix)
    L_inv = inverse_via_backwards_sub(L)
    return numpy.dot(L_inv.T, L_inv)

def main():
    pass

if __name__ == '__main__':
    main()
