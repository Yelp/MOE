# -*- coding: utf-8 -*-
"""Data containers convenient for/used to interact with optimal_learning.python members."""
import collections
import pprint

import numpy


# See SamplePoint (below) for docstring.
_BaseSamplePoint = collections.namedtuple('_BaseSamplePoint', [
    'point',
    'value',
    'noise_variance',
])


class SamplePoint(_BaseSamplePoint):

    """A point (coordinates, function value, noise variance) sampled from the objective function we are modeling/optimizing.

    This class is a representation of a "Sample Point," which is defined by the three data members listed here.
    SamplePoint is a convenient way of communicating data to the rest of the optimal_learning library (via the
    HistoricalData container); it also provides a convenient grouping for interactive introspection.

    Users are not required to use SamplePoint--iterables with the same data layout will suffice.

    :ivar point: (*iterable of dim float64*) The point sampled (in the domain of the function)
    :ivar value: (*float64*) The value returned by the function
    :ivar noise_variance: (*float64 >= 0.0*) The noise/measurement variance (if any) associated with :attr`value`

    """

    __slots__ = ()

    def __new__(cls, point, value, noise_variance=0.0):
        """Allocate and construct a new instance with the specified data fields; see class docstring for input descriptions."""
        if noise_variance >= 0.0 and numpy.isfinite(noise_variance):
            return super(SamplePoint, cls).__new__(cls, point, value, noise_variance)
        else:
            raise ValueError('noise_variance = {0} must be positive and finite!'.format(noise_variance))

    def __str__(self):
        """Pretty print this object as a dict."""
        return pprint.pformat(dict(self._asdict()))

    def json_payload(self):
        """Convert the sample_point into a dict to be consumed by json for a REST request."""
        return {
                'point': list(self.point),  # json needs a list (e.g., this may be a ndarray)
                'value': self.value,
                'value_var': self.noise_variance,
                }

    def validate(self, dim=None):
        """Check this SamplePoint passes basic validity checks: dimension is expected, all values are finite.

        :param dim: number of (expected) spatial dimensions; None to skip check
        :type dim: int > 0
        :raises ValueError: self.point does not have exactly dim entries
        :raises ValueError: if any member data is non-finite or out of range

        """
        # check that dim of points matches specified dimension
        if dim is not None and len(self.point) != dim:
            raise ValueError('Input dim = {0:d} and point dimension {1:d} do not match!'.format(dim, len(self.point)))

        # check that all values are finite
        if not numpy.isfinite(self.point).all():
            raise ValueError('point = {0} contains non-finite values!'.format(self.point))
        if not numpy.isfinite(self.value):
            raise ValueError('value = {0} is non-finite!'.format(self.value))
        if not numpy.isfinite(self.noise_variance) or self.noise_variance < 0.0:
            raise ValueError('value = {0} is non-finite or negative!'.format(self.noise_variance))


class HistoricalData(object):

    """A data container for storing the historical data from an entire experiment in a layout convenient for this library.

    Users will likely find it most convenient to store experiment historical data in tuples of
    (coordinates, value, noise); for example, these could be the columns of a database row, part of an ORM, etc.
    The :class:`moe.optimal_learning.python.SamplePoint` class (above) provides a convenient representation of this input format, but users are *not* required
    to use it.

    But the internals of optimal_learning will generally do computations on all coordinates at once, all values at once,
    and/or all noise measurements at once. So this object reads the input data and "transposes" the ordering so that
    we have a matrix of coordinates and vectors of values and noises. Compared to storing a list of :class:`moe.optimal_learning.python.SamplePoint`,
    these internals save on redundant data transformations and improve locality.

    Note that the points in HistoricalData are *not* associated to any particular domain. HistoricalData could be (and is)
    used for model selection as well as Gaussian Process manipulation, Expected Improvement optimization, etc. In the former,
    the point-domain has no meaning (as opposed to the hyperparameter domain). In the latter, users could perform multiple
    optimization runs with slightly different domains (e.g., differing levels of exploration) without changing
    HistoricalData. Users may also optimize within a subdomain of the points already sampled. Thus, we are not including
    domain in HistoricalData so as to place no restriction on how users can use optimal_learning and think about their
    experiments.

    :ivar _points_sampled: (*array of float64 with shape (self.num_sampled, self.dim)*) already-sampled points
    :ivar _points_sampled_value: (*array of float64 with shape (self.num_sampled)*) function value measured at each point
    :ivar _points_sampled_noise_variance: (*array of float64 with shape (self.num_sampled)*) noise variance associated with ``points_sampled_value``

    """

    __slots__ = ('_dim', '_points_sampled', '_points_sampled_value', '_points_sampled_noise_variance')

    def __init__(self, dim, sample_points=None, validate=False):
        """Create a HistoricalData object tracking the state of an experiment (already-sampled points, values, and noise).

        :param dim: number of spatial dimensions; must line up with len(sample_points[0]) if sample_points is empty
        :type dim: int > 0
        :param sample_points: the already-sampled points: coordinates, objective function values, and noise variance
        :type sample_points: iterable of iterables with the same structure as a list of :class:`moe.optimal_learning.python.SamplePoint`
        :param validate: whether to sanity-check the input sample_points
        :type validate: boolean

        """
        if sample_points is None:
            sample_points = []

        num_sampled = len(sample_points)
        self._dim = dim
        if validate:
            self.validate_sample_points(dim, sample_points)

        self._points_sampled = numpy.empty((num_sampled, self.dim))
        self._points_sampled_value = numpy.empty(num_sampled)
        self._points_sampled_noise_variance = numpy.empty(num_sampled)

        self._update_historical_data(0, sample_points)

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
            sample_point_list = self.to_list_of_sample_points()
            return pprint.pformat(sample_point_list)
        else:
            out_string = repr(self._points_sampled) + '\n'
            out_string += repr(self._points_sampled_value) + '\n'
            out_string += repr(self._points_sampled_noise_variance)
            return out_string

    def json_payload(self):
        """Construct a json serializeable and MOE REST recognizeable dictionary of the historical data."""
        json_points_sampled = [point.json_payload() for point in self.to_list_of_sample_points()]
        return {'points_sampled': json_points_sampled}

    @staticmethod
    def validate_sample_points(dim, sample_points):
        """Check that sample_points passes basic validity checks: dimension is the same, all values are finite.

        :param dim: number of (expected) spatial dimensions
        :type dim: int > 0
        :param sample_points: the already-sampled points: coordinates, objective function values, and noise variance
        :type sample_points: iterable of iterables with the same structure as a list of :class:`moe.optimal_learning.python.SamplePoint`
        :return: True if inputs are valid
        :rtype: boolean

        """
        num_sampled = len(sample_points)
        if dim <= 0:
            raise ValueError('Input dim = {0:d} is non-positive.'.format(dim))

        if num_sampled > 0:
            for sample_point in sample_points:
                sample_point.validate(dim=dim)

    @staticmethod
    def validate_historical_data(dim, points_sampled, points_sampled_value, points_sampled_noise_variance):
        """Check that the historical data components (dim, coordinates, values, noises) are consistent in dimension and all have finite values.

        :param dim: number of (expected) spatial dimensions
        :type dim: int > 0
        :param points_sampled: already-sampled points
        :type points_sampled: array of float64 with shape (num_sampled, dim)
        :param points_sampled_value: function value measured at each point
        :type points_sampled_value: array of float64 with shape (num_sampled)
        :param points_sampled_noise_variance: noise variance associated with ``points_sampled_value``
        :type points_sampled_noise_variance: array of float64 with shape (num_sampled)
        :return: True if inputs are valid
        :rtype: boolean

        """
        if dim <= 0:
            raise ValueError('Input dim = {0:d} is non-positive.'.format(dim))

        # Check that all array leading dimensions are the same
        if points_sampled.shape[0] != points_sampled_value.size or points_sampled.shape[0] != points_sampled_noise_variance.size:
            raise ValueError('Input arrays do not have the same leading dimension: (points_sampled, value, noise) = ({0:d}, {1:d}, {2:d})'.format(points_sampled.shape[0], points_sampled_value.size, points_sampled_noise_variance.size))

        if points_sampled.shape[0] > 0:
            for i in xrange(points_sampled.shape[0]):
                temp = SamplePoint(points_sampled[i], points_sampled_value[i], points_sampled_noise_variance[i])
                temp.validate(dim=dim)

    def append_sample_points(self, sample_points, validate=False):
        """Append the contents of ``sample_points`` to the data members of this class.

        :param sample_points: the already-sampled points: coordinates, objective function values, and noise variance
        :type sample_points: iterable of iterables with the same structure as a list of :class:`moe.optimal_learning.python.SamplePoint`
        :param validate: whether to sanity-check the input sample_points
        :type validate: boolean

        """
        if len(sample_points) == 0:
            return

        if validate:
            self.validate_sample_points(self.dim, sample_points)

        offset = self.num_sampled
        num_sampled = self.num_sampled + len(sample_points)
        self._points_sampled.resize((num_sampled, self.dim))
        self._points_sampled_value.resize(num_sampled)
        self._points_sampled_noise_variance.resize(num_sampled)

        self._update_historical_data(offset, sample_points)

    def append_historical_data(self, points_sampled, points_sampled_value, points_sampled_noise_variance, validate=False):
        """Append lists of points_sampled, their values, and their noise variances to the data members of this class.

        This class (see class docstring) stores its data members as numpy arrays; this method provides a way for users
        who already have data in this format to append directly instead of creating an intermediate :class:`moe.optimal_learning.python.SamplePoint` list.

        :param points_sampled: already-sampled points
        :type points_sampled: array of float64 with shape (num_sampled, dim)
        :param points_sampled_value: function value measured at each point
        :type points_sampled_value: array of float64 with shape (num_sampled)
        :param points_sampled_noise_variance: noise variance associated with ``points_sampled_value``
        :type points_sampled_noise_variance: array of float64 with shape (num_sampled)
        :param validate: whether to sanity-check the input sample_points
        :type validate: boolean

        """
        if points_sampled.size == 0:
            return

        if validate:
            self.validate_historical_data(self.dim, points_sampled, points_sampled_value, points_sampled_noise_variance)

        self._points_sampled = numpy.append(self._points_sampled, points_sampled, axis=0)
        self._points_sampled_value = numpy.append(self._points_sampled_value, points_sampled_value)
        self._points_sampled_noise_variance = numpy.append(self._points_sampled_noise_variance, points_sampled_noise_variance)

    def to_list_of_sample_points(self):
        """Convert this HistoricalData into a list of SamplePoint.

        The list of SamplePoint format is more convenient for human consumption/introspection.

        :return: list where i-th SamplePoint has data from the i-th entry of each self.points_sampled* member.
        :rtype: list of :class:`moe.optimal_learning.python.SamplePoint`

        """
        return [SamplePoint(numpy.copy(self._points_sampled[i]), self._points_sampled_value[i], noise_variance=self._points_sampled_noise_variance[i])
                for i in xrange(self.num_sampled)]

    def _update_historical_data(self, offset, sample_points):
        """Copy (in "transposed" order) data from ``sample_points`` into this object's data members, starting at index ``offset``.

        :param offset: the index offset to the internal arrays at which to copy in sample_points
        :type offset: int >= 0
        :param sample_points: the already-sampled points: coordinates, objective function values, and noise variance
        :type sample_points: iterable of iterables with the same structure as a list of :class:`moe.optimal_learning.python.SamplePoint`

        """
        for i, sample_point in enumerate(sample_points):
            point, value, noise_variance = sample_point
            numpy.copyto(self._points_sampled[offset + i], point)
            self._points_sampled_value[offset + i] = value
            self._points_sampled_noise_variance[offset + i] = noise_variance

    @property
    def dim(self):
        """Return the number of spatial dimensions of a point in ``self.points_sampled``."""
        return self._dim

    @property
    def num_sampled(self):
        """Return the number of sampled points."""
        return self._points_sampled.shape[0]

    @property
    def points_sampled(self):
        """Return the coordinates of the points_sampled, array of float64 with shape (self.num_sampled, self.dim)."""
        return self._points_sampled

    @property
    def points_sampled_value(self):
        """Return the objective function values measured at each of ``self.points_sampled``, array of floa664 with shape (self.num_sampled)."""
        return self._points_sampled_value

    @property
    def points_sampled_noise_variance(self):
        """Return the noise variances associated with function values measured at each of ``self.points_sampled``, array of floa664 with shape (self.num_sampled)."""
        return self._points_sampled_noise_variance
