# -*- coding: utf-8 -*-
r"""Tests checking that results (e.g., log likelihood, expected improvement, gradients thereof) computed by Python and C++ are the same.

These tests are just meant as an extra check for equivalence since we have two "identical" implementations of optimal_learning.
The C++ is independently tested (see ``moe/optimal_learning/cpp/*test.cpp`` files) as is the Python\*
(see moe/tests/optimal_learning/python/python_version), so the tests in this package generally are not very exhaustive.

\* The C++ tests are much more extensive than the Python tests, which still need substantial development.

"""
