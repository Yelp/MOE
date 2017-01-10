# -*- coding: utf-8 -*-
"""Data containers convenient for/used to interact with bandit members."""
import pprint

import numpy


class SampleArm(object):

    """An arm (name, win, loss, total, variance) sampled from the objective function we are modeling/optimizing.

    This class is a representation of a "Sample Arm," which is defined by the four data members listed here.
    SampleArm is a convenient way of communicating data to the rest of the bandit library (via the
    HistoricalData container); it also provides a convenient grouping for interactive introspection.

    Users are not required to use SampleArm, iterables with the same data layout will suffice.

    :ivar win: (*float64 >= 0.0*) The amount won from playing this arm
    :ivar loss: (*float64 >= 0.0*) The amount loss from playing this arm
    :ivar total: (*int >= 0*) The number of times we have played this arm
    :ivar variance: (*float >= 0.0*) The variance of this arm, if there is no variance it is equal to None

    """

    __slots__ = ('_win', '_loss', '_total', '_variance')

    def __init__(self, win=0.0, loss=0.0, total=0, variance=None):
        """Allocate and construct a new instance with the specified data fields; see class docstring for input descriptions."""
        self._win = win
        self._loss = loss
        self._total = total
        self._variance = variance
        self.validate()

    def __str__(self):
        """Pretty print this object as a dict."""
        return pprint.pformat(self.json_payload())

    def __add__(self, sample_arm_to_add):
        """Overload Add operator to add ``sample_arm_to_add`` sampled arm results to this arm.

        :param sample_arm_to_add: arm samples to add to this arm
        :type sample_arm_to_add: a SampleArm object
        :return: new SampleArm that is a result of adding two arms
        :rtype: SampleArm
        :raise: ValueError when ``arm.variance`` or self.variance is not None.

        """
        if self._variance is not None or sample_arm_to_add.variance is not None:
            raise ValueError('Cannot add arms when variance is not None! Please combine arms manually.')
        result = SampleArm(win=self._win + sample_arm_to_add.win, loss=self._loss + sample_arm_to_add.loss, total=self._total + sample_arm_to_add.total)
        result.validate()
        return result

    __radd__ = __add__

    def __iadd__(self, sample_arm_to_add):
        """Overload in-place Add operator to add ``sample_arm_to_add`` sampled arm results to this arm in-place.

        :param sample_arm_to_add: arm samples to add to this arm
        :type sample_arm_to_add: a SampleArm object
        :return: this arm after adding ``sample_arm_to_add``
        :rtype: SampleArm
        :raise: ValueError when ``arm.variance`` or self.variance is not None.

        """
        if self._variance is not None or sample_arm_to_add.variance is not None:
            raise ValueError('Cannot add arms when variance is not None! Please combine arms manually.')
        self._win += sample_arm_to_add.win
        self._loss += sample_arm_to_add.loss
        self._total += sample_arm_to_add.total
        self.validate()
        return self

    def json_payload(self):
        """Convert the sample_arm into a dict to be consumed by json for a REST request."""
        return {
                'win': self.win,
                'loss': self.loss,
                'total': self.total,
                'variance': self.variance,
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
        if self.variance is not None and (self.variance < 0.0 or not numpy.isfinite(self.variance)):
            raise ValueError('variance = {0} is non-finite or negative!'.format(self.variance))
        if self.total == 0 and not (self.win == 0.0 and self.loss == 0.0):
            raise ValueError('win or loss is not 0 when total is 0!')
        if self.variance is None and self.win > self.total:
            raise ValueError('win cannot be greater than total when default variance computation is used! Please specify variance.')

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

    @property
    def variance(self):
        """Return the variance of sampled tries, always greater than or equal to zero, if there is no variance it is equal to None."""
        return self._variance


class BernoulliArm(SampleArm):

    """A Bernoulli arm (name, win, loss, total, variance) sampled from the objective function we are modeling/optimizing.

    A Bernoulli arm has payoff 1 for a success and 0 for a failure.
    See more details on Bernoulli distribution at http://en.wikipedia.org/wiki/Bernoulli_distribution

    See superclass :class:`~moe.bandit.data_containers.SampleArm` for more details.

    """

    def validate(self):
        """Check this Bernoulli arm is a valid Bernoulli arm. Also check that this BernoulliArm passes basic validity checks: all values are finite.

        A Bernoulli arm has payoff 1 for a success and 0 for a failure.
        See more details on Bernoulli distribution at http://en.wikipedia.org/wiki/Bernoulli_distribution

        :raises ValueError: if any member data is non-finite or out of range or the arm is not a valid Bernoulli arm

        """
        super(BernoulliArm, self).validate()
        if self.loss != 0.0:
            raise ValueError('loss = {0} is not zero! This is not a Bernoulli arm'.format(self.loss))
        if self.win > self.total:
            raise ValueError('win = {0} > total = {1}! This is not a Bernoulli arm'.format(self.win, self.total))


class HistoricalData(object):

    """A data container for storing the historical data from an entire experiment in a layout convenient for this library.

    Users will likely find it most convenient to store experiment historical data of arms in tuples of
    (win, loss, total, variance); for example, these could be the columns of a database row, part of an ORM, etc.
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
            return pprint.pformat(self.json_payload())
        else:
            return repr(self.json_payload())

    def json_payload(self):
        """Construct a json serializeable and MOE REST recognizeable dictionary of the historical data."""
        json_arms_sampled = {}
        for name, arm in self._arms_sampled.items():
            json_arms_sampled[name] = arm.json_payload()
        return {'arms_sampled': json_arms_sampled}

    @staticmethod
    def validate_sample_arms(sample_arms):
        """Check that sample_arms passes basic validity checks: all values are finite.

        :param sample_arms: already-sampled arms: names, wins, losses, and totals
        :type sample_arms: a dictionary of  (arm name, SampleArm) key-value pairs
        :return: True if inputs are valid
        :rtype: boolean

        """
        if sample_arms:
            for arm in sample_arms.values():
                arm.validate()

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
        """Add arm sampled results from ``sample_arms`` into this object's data member.

        :param sample_arms: the already-sampled arms: wins, losses, and totals
        :type sample_arms: dictionary of (arm name, SampleArm) key-value pairs
        """
        for name, arm in sample_arms.items():
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
