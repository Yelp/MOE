# -*- coding: utf-8 -*-
"""Comparison mixins to help devs generate comparison operations for their classes.

Consider combining with tools like ``functools.total_ordering``:
https://docs.python.org/2/library/functools.html#functools.total_ordering
to fill out additional comparison functionality.

"""
from builtins import object
import inspect


class EqualityComparisonMixin(object):

    """Mixin class to autogenerate __eq__ (from instance members), __ne__, and __repr__ and disable __hash__.

    Adds no names to the class's public namespace (names without pre/postceding underscores).

    Sources:
    http://stackoverflow.com/questions/390250/elegant-ways-to-support-equivalence-equality-in-python-classes
    http://stackoverflow.com/questions/9058305/getting-attributes-of-a-class

    Be careful with NaN!
    This object uses dict comparison (which amounts to sorted tuple comparison).
    Sorted tuple comparison in Python *IS NOT* calling __eq__ on every test pair.

    In particular, Python short-circuits comparison when object ids are the same. Identity is equality.
    However, NaN makes it impossible to be consistent here, since NaN != NaN by definition.

    So::

      >> a = {'k': float('nan')}  # Note: float('nan') is not interned (no floats are)
      >> float('nan') == float('nan')  # False: by definition in IEEE754
      >> a == a  # True: short-circuits b/c both floats have the same object id
      >> a == copy.deepcopy(a)  # True: WHOA!

    WHOA: You might think deepcopy would produce *different* object ids, but it doesn't. Instead of interning,
    ``float`` have a short cache for already-created objects (the "free-list") so a new ``float`` is NOT created.
    This is generally not crippling because::

      >> b = copy.deepcopy(a)
      >> b['k'] = float('nan')  # Force a new float object
      >> a == b  # False

    Note: ``numpy.nan`` is a Python float and has the same issue. ``numpy.float64(numpy.nan)`` however is
    a ``numpy.float64`` which does not appear to undergo any kind of caching/interning.

    """

    def _get_member_dict(self):
        """Return a new ``dict`` that maps field names to their values, similar to ``namedtuple._asdict()``.

        .. Note:: this is *different* than ``_asdict()`` for a ``namedtuple`` which returns an
          ``collections.OrderedDict``. The name is also different so objects that inherit from this
          and ``namedtuple`` do not have ``_asdict`` overriden.

        :return: ``dict`` mapping non-private (see ``_get_comparable_members``) field names to their values
        :rtype: ``dict``

        """
        return dict(self._get_comparable_members())

    def _get_comparable_members(self):
        r"""Return a list of fields to use in comparing two ``EqualityComparisonMixin`` subclasses for equality.

        Ignores fields that look like "__\(.*\)__".

        Ignores routines but NOT @property decorated functions, b/c this decorator converts
        the decorated to an attribute.

        :return: (field, value) pairs representing the group of fields to compare on
        :rtype: list

        """
        members = inspect.getmembers(self, lambda field: not(inspect.isroutine(field)))
        return [(field, value) for field, value in members if not(field.startswith('__') and field.endswith('__'))]

    def __repr__(self):
        """Return a nicely formatted representation string: ``ClassName(arg1=value1, arg2=value2, ...)``.

        Included here b/c classes for which Python's default ``__eq__`` is useless typically ``__repr__``
        to unhelpful memory-address strings like: "<foo.Bar object at 0x2aba1c0cf890>"

        :return: formatted string showing class name and members used in comparison.
        :rtype: str

        """
        return '{0:s}({1:s})'.format(
            self.__class__.__name__,
            ', '.join(['{0}={1}'.format(pair[0], pair[1]) for pair in self._get_comparable_members()]),
        )

    def __eq__(self, other):
        """Check if two objects are both NewtonParameters; if so, check if they hold the same parameter values."""
        if type(other) is type(self):
            return self._get_member_dict() == other._get_member_dict()
        return False

    def __ne__(self, other):
        """Not ``__eq__``; see docstring for :func:`~moe.optimal_learning.python.cpp_wrappers.optimization.EqualityComparisonMixin.__eq__`."""
        return not self.__eq__(other)

    __hash__ = None
