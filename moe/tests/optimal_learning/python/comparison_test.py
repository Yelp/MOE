# -*- coding: utf-8 -*-
"""Tests for the functions/classes in ``comparison.py``."""
import copy

import pytest

from moe.optimal_learning.python.comparison import EqualityComparisonMixin


class NotComparableObject(object):

    """Object with == and != disabled."""

    def __eq__(self, other):
        """Disable __eq__."""
        return NotImplemented

    def __ne__(self, other):
        """Disable __ne__."""
        return NotImplemented


class ComparableTestObject(EqualityComparisonMixin):

    """Object for testing equality comparisons."""

    def __init__(self, args, property_offset=0, function_offset=0):
        """Construct ComparableTestObject.

        :param args: attributes to set
        :type args: list of (attr_name, attr_value) tuples

        """
        for arg in args:
            setattr(self, arg[0], copy.deepcopy(arg[1]))

        # Private "property_offset" variable for property-test; this will be ignored
        # by EqualityComparisonMixin if it skips properties (as expected)
        self.__property_offset__ = copy.deepcopy(property_offset)
        self.__function_offset__ = copy.deepcopy(function_offset)

    @property
    def some_property(self):
        """Some property; will be picked up in the comparison."""
        return self.__property_offset__

    def some_function(self):
        """Some function; will not be picked up in the comparison."""
        return self.__function_offset__


class TestEqualityComparisonMixin(object):

    """Test the mixin features of ``EqualityComparisonMixin``."""

    @classmethod
    @pytest.fixture(autouse=True, scope='class')
    def base_setup(cls):
        """Set up test cases for ``EqualityComparisonMixin``."""
        # attributes that will be picked up in comparison
        compared_attributes_group0 = [
            ('public_param', 1),
            ('_private_param', {'hi': 'bye'}),
            ('__super_private_param', 'wee'),
            ('callable', list),
            ('function', setattr),
        ]
        # includes non-compared attributes and not-comparable attributes
        full_attributes_group0_0 = compared_attributes_group0 + [
            ('__system_param__', 3.14),
            ('__not_comparable__', NotComparableObject()),
        ]
        full_attributes_group0_1 = compared_attributes_group0 + [
            ('__system_param__', 2.78),
            ('__not_comparable__', NotComparableObject()),
        ]
        # only members of the same group should compare equal
        cls.comparable_object_group0_member0 = ComparableTestObject(compared_attributes_group0, property_offset=0, function_offset=0)
        # different function_offset
        cls.comparable_object_group0_member1 = ComparableTestObject(compared_attributes_group0, property_offset=0, function_offset=1)
        # includes non-comparable attributes
        cls.comparable_object_group0_member2 = ComparableTestObject(full_attributes_group0_0, property_offset=0, function_offset=1)
        # includes different non-comparable attributes
        cls.comparable_object_group0_member3 = ComparableTestObject(full_attributes_group0_1, property_offset=0, function_offset=1)

        full_attributes_group1 = copy.copy(full_attributes_group0_0)
        full_attributes_group1[0] = (full_attributes_group1[0][0], full_attributes_group1[0][1] + 3)
        # different comparable values from group 0
        cls.comparable_object_group1_member0 = ComparableTestObject(full_attributes_group1, property_offset=0, function_offset=0)

        # different value in the @property from group 0
        cls.comparable_object_group2_member0 = ComparableTestObject(full_attributes_group1, property_offset=2, function_offset=0)

        # extra comparable value
        cls.comparable_object_group3_member0 = ComparableTestObject(full_attributes_group1 + [('other_param', 8)], property_offset=2, function_offset=0)

        not_comparable_attributes = [
            ('public_param', NotComparableObject()),
            ('_private_param', {'hi': 'bye'}),
            ('__super_private_param', 'wee'),
            ('callable', list),
            ('function', setattr),
        ]
        cls.not_comparable_object0 = ComparableTestObject(not_comparable_attributes)
        cls.not_comparable_object1 = ComparableTestObject(not_comparable_attributes)

    @staticmethod
    def _test_equals(obj1, obj2):
        assert obj1 == obj2
        assert not obj1 != obj2

    @staticmethod
    def _test_not_equals(obj1, obj2):
        assert obj1 != obj2
        assert not obj1 == obj2

    def test_eq(self):
        """Test __eq__ and __ne__ operators."""
        # mismatched types will not compare
        self._test_not_equals(self.comparable_object_group0_member0, [])

        # compatible types will compare but come out false
        self._test_not_equals(self.comparable_object_group0_member0, object())

        self._test_equals(self.comparable_object_group0_member0, self.comparable_object_group0_member0)
        self._test_equals(self.comparable_object_group0_member0, self.comparable_object_group0_member1)
        self._test_equals(self.comparable_object_group0_member0, self.comparable_object_group0_member2)
        self._test_equals(self.comparable_object_group0_member0, self.comparable_object_group0_member3)

        # check non-comparable attributes aren't being compared
        self._test_equals(self.comparable_object_group0_member2, self.comparable_object_group0_member3)

        self._test_not_equals(self.comparable_object_group0_member0, self.comparable_object_group1_member0)
        self._test_not_equals(self.comparable_object_group0_member0, self.comparable_object_group2_member0)
        self._test_not_equals(self.comparable_object_group0_member0, self.comparable_object_group3_member0)

        # object with a member that is not comparable will fail
        assert self.comparable_object_group0_member0 != self.not_comparable_object0

        # identity is equality
        assert self.not_comparable_object1 == self.not_comparable_object1
        # objects with a non-comparable component cannot be equal
        assert self.not_comparable_object0 != self.not_comparable_object1

    def test_hash(self):
        """Verify that hashing will fail."""
        with pytest.raises(TypeError):
            hash(self.comparable_object_group0_member0)
