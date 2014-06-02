# -*- coding: utf-8 -*-
r"""Testing code for the (Python) optimal_learning library.

Testing is done via the Testify package:
https://github.com/Yelp/Testify

This package includes:

* Test cases/test setup files
* Tests for optimal_learning/python/\*py utils
* Tests for optimal_learning/python_version
* Tests for optimal_learning/cpp_wrappers

**Files in this package**

* optimal_learning_test_case.py: test case with extra asserts for checking relative differences of floats (scalar, vector)
* gaussian_process_test_case.py: test case with extra logic to construct random gaussian process priors; meant to provide
  a well-behaved source of random data to unit tests.
* geometry_utils_test.py: tests for geometry_utils

**Files in this package**

* optimal_learning_test_case.py: base test case for optimal_learning tests with some extra asserts for checking relative differences of floats (scalar, vector)
* gaussian_process_test_case.py: test case for tests that manipulate GPs, includes extra
  logic to construct random gaussian process priors; meant to provide
  a well-behaved source of random data to unit tests.
* gaussian_process_test_utils.py: utilities for constructing a random domain, covariance, and GaussianProcess
* geometry_utils_test.py: tests for optimal_learning.python.geometry_utils

**Subpackages**

* python_version: tests for the Python implementation of optimal_learning. These include some manual checks, ping tests,
  and some high level integration tests. Python testing is currently relatively sparse; we rely heavily on
  the C++ comparison.
* cpp_wrappers: tests that check the equivalence of the C++ implementation and the Python implementation of
  optimal_learning (where applicable). Also runs the C++ unit tests.

"""
