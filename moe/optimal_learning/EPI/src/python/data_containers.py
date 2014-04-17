# -*- coding: utf-8 -*-
"""Data containers convenient for/used to interact with optimal_learning.EPI.src.python members."""
import collections
import logging
import numpy
import pprint

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


class SamplePoint(collections.namedtuple('SamplePoint', ['point', 'value', 'noise_variance'])):

    """A point (coordinates, function value, noise variance) sampled from the objective function we are modeling/optimizing.

    This class is a representation of a "Sample Point," which is defined by the three data members listed here.
    SamplePoint is a convenient way of communicating data to the rest of the optimal_learning library (via the
    HistoricalData container); it also provides a convenient grouping for interactive introspection.

    Users are not required to use SamplePoint--iterables with the same data layout will suffice.

    :ivar point: (*iterable of dim float64*) The point sampled (in the domain of the function)
    :ivar value: (*float64*) The value returned by the function
    :ivar noise_variance: (*float64*) The noise/measurement variance (if any) associated with ``value``

    """

    __slots__ = ()

    def __new__(cls, point, value, noise_variance=0.0):
        """Allocate and construct a new class instance with the specified data fields.

        Checks for data validity.
        See class docs for input descriptions.

        """
        if noise_variance < 0.0:
            raise ValueError('noise_variance = %f must be positive!' % noise_variance)
        else:
            return super(SamplePoint, cls).__new__(cls, point, value, noise_variance)

    def __str__(self):
        """Pretty print this object as a dict."""
        return pprint.pformat(dict(self._asdict()))


class HistoricalData(object):

    """A data container for storing the historical data from an entire experiment in a layout convenient for this library.

    Users will likely find it most convenient to store experiment historical data in "tuples" of
    (coordinates, value, noise); for example, these could be the columns of a database row, part of an ORM, etc.
    The SamplePoint class (above) provides a convenient representation of this input format, but users are *not* required
    to use it.

    But the internals of optimal_learning will generally do computations on all coordinates at once, all values at once,
    and/or all noise measurements at once. So this object reads the input data data and "transposes" the ordering so that
    we have a matrix of coordinates and vectors of values and noises. Compared to storing a list of SampledPoint,
    these internals save on redundant data transformations and improve locality.

    Note that the points in HistoricalData are *not* associated to any particular domain. HistoricalData could be (and is)
    used for model selection as well as Gaussian Process manipulation, Expected Improvement optimization, etc. In the former,
    the point-domain has no meaning (as opposed to the hyperparameter domain). In the latter, users could perform multiple
    optimization runs with slightly different domains (e.g., differing levels of exploration) without changing
    HistoricalData. Users may also optimize within a subdomain of the points already sampled. Thus, we are not including
    domain in HistoricalData so as to place no restriction on how users can use optimal_learning and think about their
    experiments.

    TODO(eliu): it'd be cool if this inherited from namedtuple (since adding new points makes sense, but changing dim or
    arbitrarily changing the referrant of noise_variance does not). Not 100% sure how to do this since __new__ and
    __init__ would need to take different arguments.

    :ivar _points_sampled: (*array of float64 with shape (self.num_sampled, self.dim)*) already-sampled points
    :ivar _points_sampled_value: (*array of float64 with shape (self.num_sampled)*) function value measured at each point
    :ivar _points_sampled_noise_variance: (*array of float64 with shape (self.num_sampled)*) noise variance associated with ``points_sampled_value``

    """

    __slots__ = ('_dim', '_points_sampled', '_points_sampled_value', '_points_sampled_noise_variance')

    def __init__(self, dim, sample_points=[], validate=False):
        """Create a HistoricalData object tracking the state of an experiment (already-sampled points, values, and noise).

        :param dim: number of spatial dimensions; must line up with len(sample_points[0]) if sample_points is empty
        :type dim: int > 0
        :param sample_points: the already-sampled points: coordinates, objective function values, and noise variance
        :type sample_points: iterable of iterables with the same structure as a list of SamplePoint
        :param validate: whether to sanity-check the input sample_points
        :type validate: boolean

        """
        num_sampled = len(sample_points)
        self._dim = dim
        if validate:
            self.validate_sample_points(dim, sample_points)

        self._points_sampled = numpy.empty((num_sampled, self.dim))
        self._points_sampled_value = numpy.empty(num_sampled)
        self._points_sampled_noise_variance = numpy.empty(num_sampled)

        self._update_historical_data(0, sample_points)

    def __str__(self):
        """Pretty-printed string representation of this HistoricalData object.

        See self.__repr__() if you want the more traditional __str__ behavior.

        """
        sample_point_list = self.to_list_of_sample_points()
        return pprint.pformat(sample_point_list)

    def __repr__(self):
        """String (high precision) representation of this HistoricalData object's data members."""
        out_string = repr(self._points_sampled) + '\n'
        out_string += repr(self._points_sampled_value) + '\n'
        out_string += repr(self._points_sampled_noise_variance)
        return out_string

    @staticmethod
    def validate_sample_points(dim, sample_points):
        """Check that sample_points passes basic validity checks: dimension is the same, all values are finite.

        :param dim: number of (expected) spatial dimensions
        :type dim: int > 0
        :param sample_points: the already-sampled points: coordinates, objective function values, and noise variance
        :type sample_points: iterable of iterables with the same structure as a list of SamplePoint
        :return: True if inputs are valid
        :rtype: boolean

        """
        num_sampled = len(sample_points)
        if dim <= 0:
            raise ValueError('Input dim = %d is non-positive.' % dim)

        if num_sampled > 0:
            for sample_point in sample_points:
                point, value, noise_variance = sample_point
                # check that dim of points matches specified dimension
                if len(point) != dim:
                    raise ValueError('Input dim = %d and point dimension %d do not match!' % (dim, len(point)))
                # check that all values are finite
                if not numpy.isfinite(point).all():
                    raise ValueError('point = %s contains non-finite values!' % point)
                if not numpy.isfinite(value):
                    raise ValueError('value = %f is non-finite!' % value)
                if not numpy.isfinite(noise_variance) and noise_variance < 0.0:
                    raise ValueError('value = %f is non-finite or negative!' % noise_variance)

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
            raise ValueError('Input dim = %d is non-positive.' % dim)

        # Check that all array leading dimensions are the same
        if points_sampled.shape[0] != points_sampled_value.size or points_sampled.shape[0] != points_sampled_noise_variance.size:
            raise ValueError('Input arrays do not have the same leading dimension: (points_sampled, value, noise) = (%d, %d, %d)' % (points_sampled.shape[0], points_sampled_value.size, points_sampled_noise_variance.size))

        if points_sampled.shape[0] > 0:
            # check that dim of points matches specified dimension
            if points_sampled.shape[1] != dim:
                raise ValueError('Input dim = %d and point dimension %d do not match!' % (dim, points_sampled.shape[1]))

            # check that all values are finite
            if not numpy.isfinite(points_sampled).all():
                raise ValueError('points_sampled contains non-finite values!')
            if not numpy.isfinite(points_sampled_value).all():
                raise ValueError('points_sampled_value contains non-finite values!')
            if not numpy.isfinite(points_sampled_noise_variance).all():
                raise ValueError('points_sampled_noise_variance contains non-finite values!')

    def append_sample_points(self, sample_points, validate=False):
        """Append the contents of ``sample_points`` to the data members of this class.

        :param sample_points: the already-sampled points: coordinates, objective function values, and noise variance
        :type sample_points: iterable of iterables with the same structure as a list of SamplePoint
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
        who already have data in this format to append directly instead of creating an intermediate SamplePoint list.

        :param points_sampled: already-sampled points
        :type points_sampled: array of float64 with shape (num_sampled, dim)
        :param points_sampled_value: function value measured at each point
        :type points_sampled_value: array of float64 with shape (num_sampled)
        :param points_sampled_noise_variance: noise variance associated with ``points_sampled_value``
        :type points_sampled_noise_variance: array of float64 with shape (num_sampled)
        :param validate: whether to sanity-check the input sample_points
        :type validate: boolean

        """
        if points_sampled.shape[0] == 0:
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
        :rtype: list of SamplePoint

        """
        out = []
        for i in range(self.num_sampled):
            out.append(SamplePoint(
                numpy.copy(self._points_sampled[i]),
                self._points_sampled_value[i],
                noise_variance=self._points_sampled_noise_variance[i],
            ))
        return out

    def _update_historical_data(self, offset, sample_points):
        """Copy (in "transposed" order) data from ``sample_points`` into this object's data members, starting at index ``offset``.

        :param offset: the index offset to the internal arrays at which to copy in sample_points
        :type offset: int >= 0
        :param sample_points: the already-sampled points: coordinates, objective function values, and noise variance
        :type sample_points: iterable of iterables with the same structure as a list of SamplePoint

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
