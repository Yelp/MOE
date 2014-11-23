# -*- coding: utf-8 -*-
"""Tests for the ``cpp_wrappers.optimization`` module.

Currently these objects either inherit from Boost Python objects or are very thin data containers. So tests
just verify that objects are created & behave as expected.
MOE does not yet support the ability to optimize arbitrary Python functions through C++-coded optimizers.

"""
import copy

import pytest

from moe.optimal_learning.python.cpp_wrappers.optimization import NewtonParameters, GradientDescentParameters


class TestOptimizerParameters(object):

    """Test that the various optimizer parameter classes (wrapping C++ objects) work."""

    @classmethod
    @pytest.fixture(autouse=True, scope='class')
    def base_setup(cls):
        """Set up dummy parameters for testing optimization parameter structs."""
        cls.newton_param_dict = {
            'num_multistarts': 10,
            'max_num_steps': 20,
            'gamma': 1.05,
            'time_factor': 1.0e-5,
            'max_relative_change': 0.8,
            'tolerance': 3.7e-8,
        }

        # new object b/c we want to modify it
        cls.gd_param_dict = copy.deepcopy(cls.newton_param_dict)
        del cls.gd_param_dict['time_factor']
        cls.gd_param_dict.update({
            'max_num_restarts': 50,
            'num_steps_averaged': 11,
            'pre_mult': 0.45,
        })

    @staticmethod
    def _parameter_test_core(param_type, param_dict):
        """Test param struct construction, member read/write, and equality check."""
        # Check construction
        params = param_type(**param_dict)

        # Check that the internal state matches the input
        test_params_dict = dict(params._get_member_dict())
        assert test_params_dict == param_dict

        # New object is equal to old when params match
        params_other = param_type(**param_dict)
        assert params_other == params

        # Inequality when we change a param
        params_other.gamma += 1.2
        assert params_other != params

    def test_newton_parameters(self):
        """Test that ``NewtonParameters`` is created correctly and comparison works."""
        self._parameter_test_core(NewtonParameters, self.newton_param_dict)

    def test_gradient_descent_parameters(self):
        """Test that ``GradientDescentParameters`` is created correctly and comparison works."""
        self._parameter_test_core(GradientDescentParameters, self.gd_param_dict)
