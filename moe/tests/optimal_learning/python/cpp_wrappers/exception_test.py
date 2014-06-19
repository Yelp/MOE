# -*- coding: utf-8 -*-
"""Tests for the C++-defined Python exception type objects."""
import testify as T

import moe.build.GPP as C_GP


class ExceptionStructureTest(T.TestCase):

    """Tests for the C++-defined Python exception type objects."""

    def test_exception_class_hierarchy(self):
        """Test that the C++-defined Python exception type objects have the right class hiearchy."""
        # Base class inherits from Exception
        T.assert_true(issubclass(C_GP.OptimalLearningException, Exception))

        type_objects = (C_GP.BoundsException, C_GP.InvalidValueException, C_GP.SingularMatrixException)
        for type_object in type_objects:
            T.assert_true(issubclass(type_object, C_GP.OptimalLearningException))

    def test_exception_thrown_from_cpp(self):
        """Test that a C++ interface function throws the expected type."""
        T.assert_raises(C_GP.BoundsException, C_GP.GaussianProcess, [-1.0, [1.0]], [], [], [], 1, 0)
