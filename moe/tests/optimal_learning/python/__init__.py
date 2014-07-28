# -*- coding: utf-8 -*-
r"""Testing code for the (Python) optimal_learning library.

Testing is done via the Testify package:
https://github.com/Yelp/Testify

This package includes:

* Test cases/test setup files
* Tests for classes and utils in :mod:`moe.optimal_learning.python`
* Tests for classes and functions in :mod:`moe.optimal_learning.python.python_version`
* Tests for classes and functions in :mod:`moe.optimal_learning.python.cpp_wrappers`

**Files in this package**

* :mod:`moe.tests.optimal_learning.python.optimal_learning_test_case`: base test case for optimal_learning tests with some extra asserts for checking relative differences of floats (scalar, vector)
* :mod:`moe.tests.optimal_learning.python.gaussian_process_test_case`: test case for tests that manipulate GPs, includes extra
  logic to construct random gaussian process priors; meant to provide
  a well-behaved source of random data to unit tests.
* :mod:`moe.tests.optimal_learning.python.gaussian_process_test_utils`: utilities for constructing a random domain, covariance, and GaussianProcess
* :mod:`moe.tests.optimal_learning.python.geometry_utils_test`: tests for :mod:`moe.optimal_learning.python.geometry_utils`

**Subpackages**

* :mod:`moe.tests.optimal_learning.python.python_version`: tests for the Python implementation of optimal_learning. These include some manual checks, ping tests,
  and some high level integration tests. Python testing is currently relatively sparse; we rely heavily on
  the C++ comparison.
* :mod:`moe.tests.optimal_learning.python.cpp_wrappers`: tests that check the equivalence of the C++ implementation and the Python implementation of
  optimal_learning (where applicable). Also runs the C++ unit tests.

"""
