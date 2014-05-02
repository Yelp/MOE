# -*- coding: utf-8 -*-
"""Python unit tests, including straight Python tests and comparisons against the C++ implementation.

Major members:
* optimal_learning_test_case.py: test case with extra asserts for checking relative differences of floats (scalar, vector)
* gaussian_process_test_case.py: test case with extra logic to construct random gaussian process priors; meant to provide
  a well-behaved source of random data to unit tests.
* geometry_utils_test.py: tests for geometry_utils

Major sub-packages:
* cpp_wrappers: unit tests that compare the C++ and Python implementations of equivalent features. Also invokes all of
  the C++ unit tests.

* python_version: unit tests for the Python implementation of optimal_learning. Unfortunately, these are not very thorough,
  although we do verify the major Python routines against the C++; see package comments for further details.

"""
