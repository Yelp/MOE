# -*- coding: utf-8 -*-
"""Data containers convenient for/used to interact with bandit members."""
import pprint

import numpy


class SampleArm(object):

    """An arm (name, win, loss, total) sampled from the objective function we are modeling/optimizing.

    This class is a representation of a "Sample Arm," which is defined by the three data members listed here.
    SampleArm is a convenient way of communicating data to the rest of the bandit library (via the
    HistoricalData container); it also provides a convenient grouping for interactive introspection.

    Users are not required to use SampleArm, iterables with the same data layout will suffice.

    :ivar win: (*float64 >= 0.0*) The amount won from playing this arm
    :ivar loss: (*float64 >= 0.0*) The amount loss from playing this arm
    :ivar total: (*int >= 0*) The number of times we have played this arm

    """

    __slots__ = ('_win', '_loss', '_total')

    def __init__(self, win=0.0, loss=0.0, total=0):
        """Allocate and construct a new instance with the specified data fields; see class docstring for input descriptions."""
        self._win = win
        self._loss = loss
        self._total = total
        self.validate()

    def __str__(self):
        """Pretty print this object as a dict."""
        return pprint.pformat(dict(self._asdict()))

    def __add__(self, arm):
        """Overload Add operator to add sampled arm results to this arm.

        :param arm: arm samples to add to this arm
        :type arm: a SampleArm object
        """
        self._win += arm.win
        self._loss += arm.loss
        self._total += arm.total

    def json_payload(self):
        """Convert the sample_arm into a dict to be consumed by json for a REST request."""
        return {
                'win': self.win,
                'loss': self.loss,
                'total': self.total,
                }

    def validate(self):
        """Check this SampleArm passes basic validity checks: all values are finite.

        :raises ValueError: if any member data is non-finite or out of range

        """
        # check that all values are finite
        if self.win < 0.0 or not numpy.isfinite(self.win):
            raise ValueError('win = {0} is non-finite or negative!'.format(self.win))
        if self.loss < 0.0 or not numpy.isfinite(self.loss):
            raise ValueError('loss = {0} is non-finite or negative!'.format(self.loss))
        if self.total < 0 or not numpy.isfinite(self.total):
            raise ValueError('total = {0} is non-finite or negative!'.format(self.total))
        if self.total == 0 and not (self.win == 0.0 and self.loss == 0.0):
            raise ValueError('win or loss is not 0 when total is 0!')

    @property
    def win(self):
        """Return the amount win, always greater than or equal to zero."""
        return self._win

    @property
    def loss(self):
        """Return the amount loss, always greater than or equal to zero."""
        return self._loss

    @property
    def total(self):
        """Return the total number of tries, always a non-negative integer."""
        return self._total


class HistoricalData(object):

    """A data container for storing the historical data from an entire experiment in a layout convenient for this library.

    Users will likely find it most convenient to store experiment historical data of arms in "tuples" of
    (win, loss, total); for example, these could be the columns of a database row, part of an ORM, etc.
    The SampleArm class (above) provides a convenient representation of this input format, but users are *not* required
    to use it.

    :ivar _arms_sampled: (*dict*) mapping of arm names to already-sampled arms

    """

    __slots__ = ('_arms_sampled')

    def __init__(self, sample_arms=None, validate=True):
        """Create a HistoricalData object tracking the state of an experiment (already-sampled arms).

        :param sample_arms: the already-sampled arms: names, wins, losses, and totals
        :type sample_arms: a dictionary of (arm name, SampleArm) key-value pairs
        :param validate: whether to sanity-check the input sample_arms
        :type validate: boolean

        """
        if sample_arms is None:
            sample_arms = {}

        if validate:
            self.validate_sample_arms(sample_arms)

        self._arms_sampled = sample_arms

    def __str__(self, pretty_print=True):
        """String representation of this HistoricalData object.

        pretty-print'ing produces output that is easily read by humans.
        Disabling it prints the member arrays to the screen in full precision; this is convenient for
        pasting into C++ or other debugging purposes.

        :param pretty_print: enable pretty-printing for formatted, human-readable output
        :type pretty_print: bool
        :return: string representation
        :rtype: string

        """
        if pretty_print:
            return pprint.pformat(self._arms_sampled)
        else:
            return repr(self._arms_sampled)

    def json_payload(self):
        """Construct a json serializeable and MOE REST recognizeable dictionary of the historical data."""
        json_arms_sampled = {}
        for name, arm in self._arms_sampled.iteritems():
            json_arms_sampled[name] = arm.json_payload()
        return {'arms_sampled': json_arms_sampled}

    @staticmethod
    def validate_sample_arms(sample_arms):
        """Check that sample_arms passes basic validity checks: all values are finite.

        :param sample_arms: the already-sampled arms: names, wins, losses, and totals
        :type sample_arms: a dictionary of  (arm name, SampleArm) key-value pairs
        :return: True if inputs are valid
        :rtype: boolean

        """
        if sample_arms:
            for arm in sample_arms.itervalues():
                arm.validate()

    @staticmethod
    def validate_historical_data(arms_sampled):
        """Check that the historical data components (wins, losses, and total) all have finite values.

        :param arms_sampled: already-sampled arms
        :type arms_sampled: a dictionary of  (arm name, SampleArm) key-value pairs
        :return: True if inputs are valid
        :rtype: boolean

        """
        HistoricalData.validate_sample_arms(arms_sampled)

    def append_sample_arms(self, sample_arms, validate=True):
        """Append the contents of ``sample_arms`` to the data members of this class.

        This method first validates the arms and then updates the historical data.
        The result of combining two valid arms is always a valid arm.

        :param sample_arms: the already-sampled arms: wins, losses, and totals
        :type sample_arms: a dictionary of  (arm name, SampleArm) key-value pairs
        :param validate: whether to sanity-check the input sample_arms
        :type validate: boolean

        """
        if not sample_arms:
            return

        if validate:
            self.validate_sample_arms(sample_arms)

        self._update_historical_data(sample_arms)

    def _update_historical_data(self, sample_arms):
        """Add arm sampled ersults from ``sample_arms`` into this object's data member.

        :param sample_arms: the already-sampled arms: wins, losses, and totals
        :type sample_arms: dictionary of (arm name, SampleArm) key-value pairs
        """
        for name, arm in sample_arms.iteritems():
            if name in self._arms_sampled:
                self._arms_sampled[name] += arm
            else:
                self._arms_sampled[name] = arm

    @property
    def num_arms(self):
        """Return the number of sampled arms."""
        return len(self._arms_sampled)

    @property
    def arms_sampled(self):
        """Return the arms_sampled, a dictionary of (arm name, SampleArm) key-value pairs."""
        return self._arms_sampled
