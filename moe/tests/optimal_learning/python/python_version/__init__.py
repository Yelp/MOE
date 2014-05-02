# -*- coding: utf-8 -*-
"""Test suite for the Python implementation of optimal_learning.

* Lower level functions (e.g., covariance) are generally tested with a combination of manual verification and derivative pinging.
* Mid-level level functions (e.g., log likelihood) are mostly tested with derivative pinging.
* High-level functions (e.g., optimization of EI or log likelihood) are only loosely tested, only checking that outputs
are valid (vs trying to verify them).

Note that the Python implementation is additionally tested against the C++ (same inputs, same results for the various
optimal_learning features) implementation (see moe/tests/optimal_learning/python/cpp_wrappers).

TODO(eliu): in general, the Python test suite is lacking and we rely on comparison against the more extensively tested
C++ implementation to check the Python.

"""
