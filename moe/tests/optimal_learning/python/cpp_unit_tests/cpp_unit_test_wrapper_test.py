# -*- coding: utf-8 -*-
"""Dummy test case that invokes all of the C++ unit tests."""
from builtins import object
import moe.build.GPP as C_GP


class TestCppUnitTestWrapper(object):

    """Calls a C++ function that runs all C++ unit tests.

    TODO(GH-115): Remove/fix this once C++ gets a proper unit testing framework.

    """

    def test_run_cpp_unit_tests(self):
        """Call C++ function that runs all C++ unit tests and assert 0 errors."""
        number_of_cpp_test_errors = C_GP.run_cpp_tests()
        assert number_of_cpp_test_errors == 0
