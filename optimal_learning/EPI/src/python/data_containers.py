# -*- coding: utf-8 -*-
"""Data containers convenient for/used to interact with optimal_learning.EPI.src.python members."""
import collections
import logging
import numpy

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


class SamplePoint(collections.namedtuple('SamplePoint', ['point', 'value', 'noise_variance'])):

    """A point (coordinates, function value, noise variance) sampled from the objective function we are modeling/optimizing.

    :ivar point: (*iterable of dim double*) The point sampled (in the domain of the function)
    :ivar value: (*double*) The value returned by the function
    :ivar noise_variance: The noise/measurement variance (if any) associated with ``value``, double

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
        """Pretty print this object."""
        return "SamplePoint: f(%s) = %f +/- %f" % (str(self.point), self.value, self.noise_variance)


class HistoricalData(object):

    """A data container for storing the historical data from an entire experiment in a layout convenient for this library.

    Users will likely find it most convenient to store experiment historical data in "tuples" of
    (coordinates, value, noise); for example, these could be the columns of a database row, part of an ORM, etc.
    The SamplePoint class (above) provides a convenient representation of this input format, but users are *not* required
    to use it.

    But the internals of optimal_learning will generally do computations on all coordinates at once, all values at once,
    and/or all noise measurements at once. So this object reads the input data data and "transposes" the ordering so that
    we have a matrix of coordinates and vectors of values and noises.

    TODO(eliu): it'd be cool if this inherited from namedtuple (since adding new points makes sense, but changing dim or
    arbitrarily changing the referrant of noise_variance does not). Not 100% sure how to do this since __new__ and
    __init__ would need to take different arguments.

    :ivar points_sampled: already-sampled points, 2d array[self.num_sampled][self.dim] of double
    :ivar points_sampled_value: function value measured at each point, 1d array[self.num_sampled] of double
    :ivar noise_variance: noise variance associated with ``points_sampled_value``, 1d array[self.num_sampled] of double

    """

    __slots__ = ('_dim', 'points_sampled', 'points_sampled_value', 'noise_variance')

    def __init__(self, dim, sample_points=[]):
        """Create a HistoricalData object tracking the state of an experiment (already-sampled points, values, and noise).

        :param dim: number of spatial dimensions; must line up with len(sample_points[0]) if sample_points is empty
        :type dim: int > 0
        :param sample_points: the already-sampled points: coordinates, objective function values, and noise variance
        :type sample_points: iterable of iterables with the same structure as a list of SamplePoint

        """
        num_sampled = len(sample_points)
        self._dim = dim
        if num_sampled > 0 and dim != len(sample_points[0][0]):
            raise ValueError('Input dim = %d and point dimension %d do not match!' % (dim, len(sample_points[0][0])))            

        self.points_sampled = numpy.empty((num_sampled, self.dim))
        self.points_sampled_value = numpy.empty(num_sampled)
        self.noise_variance = numpy.empty(num_sampled)

        self._copy_sample_points(0, sample_points)

    def append(self, sample_points):
        """Append the contents of ``sample_points`` to the data members of this class.

        :param sample_points: the already-sampled points: coordinates, objective function values, and noise variance
        :type sample_points: iterable of iterables with the same structure as a list of SamplePoint

        """
        if len(sample_points) == 0:
            return

        offset = self.num_sampled
        num_sampled = self.num_sampled + len(sample_points)
        self.points_sampled.resize((num_sampled, self.dim))
        self.points_sampled_value.resize(num_sampled)
        self.noise_variance.resize(num_sampled)

        self._copy_sample_points(offset, sample_points)

    def _copy_sample_points(self, offset, sample_points):
        """Copy (in "transposed" order) data from ``sample_points`` into this object's data members, starting at index ``offset``.

        :param offset: 
        :type offset: int >= 0
        :param sample_points: the already-sampled points: coordinates, objective function values, and noise variance
        :type sample_points: iterable of iterables with the same structure as a list of SamplePoint

        """
        for i, sample_point in enumerate(sample_points):
            point, value, noise_variance = sample_point
            numpy.copyto(self.points_sampled[offset + i], point)
            self.points_sampled_value[offset + i] = value
            self.noise_variance[offset + i] = noise_variance        

    @property
    def dim(self):
        """Return the number of spatial dimensions."""
        return self._dim

    @property
    def num_sampled(self):
        """Return the number of sampled points."""
        return self.points_sampled.shape[0]
