# -*- coding: utf-8 -*-
"""Base test case for tests that manipulate Gaussian Process data and supporting structures."""
import pytest

import collections

import numpy

from moe.optimal_learning.python.geometry_utils import ClosedInterval
from moe.optimal_learning.python.python_version.covariance import SquareExponential
from moe.optimal_learning.python.python_version.domain import TensorProductDomain
from moe.optimal_learning.python.python_version.gaussian_process import GaussianProcess
import moe.tests.optimal_learning.python.gaussian_process_test_utils as gp_utils
from moe.tests.optimal_learning.python.optimal_learning_test_case import OptimalLearningTestCase


# See GaussianProcessTestEnvironment (below) for docstring.
_BaseGaussianProcessTestEnvironment = collections.namedtuple('GaussianProcessTestEnvironment', [
    'domain',
    'gaussian_process',
])


class GaussianProcessTestEnvironment(_BaseGaussianProcessTestEnvironment):

    """An object for representing a (randomly generated) Gaussian Process.

    :ivar domain: (*interfaces.domain_interface.DomainInterface subclass*) domain the GP was built on
    :ivar gaussian_process: (*interfaces.gaussian_process_interface.GaussianProcessInterface subclass*) the constructed GP

    """

    __slots__ = ()


class GaussianProcessTestEnvironmentInput(object):

    """A test environment for constructing randomly generated Gaussian Process priors within GaussianProcessTestCase.

    This is used only in testing. The intended use-case is that subclasses of GaussianProcessTestCase (below) will
    define one of these objects, and then GaussianProcessTestCase has some simple logic to precompute the requested
    GaussianProcess-derived test case(s).

    """

    def __init__(
            self,
            dim,
            num_hyperparameters,
            num_sampled,
            noise_variance_base=0.0,
            hyperparameter_interval=ClosedInterval(0.2, 1.3),
            lower_bound_interval=ClosedInterval(-2.0, 0.5),
            upper_bound_interval=ClosedInterval(2.0, 3.5),
            covariance_class=SquareExponential,
            spatial_domain_class=TensorProductDomain,
            hyperparameter_domain_class=TensorProductDomain,
            gaussian_process_class=GaussianProcess,
    ):
        """Create a test environment: object with enough info to construct a Gaussian Process prior from repeated random draws.

        :param dim: number of (expected) spatial dimensions; None to skip check
        :type dim: int > 0
        :param num_hyperparameters: number of hyperparemeters of the covariance function
        :type num_hyperparameters: int > 0
        :param num_sampled: number of ``points_sampled`` to generate from the GP prior
        :type num_sampled: int > 0
        :param noise_variance_base: noise variance to associate with each sampled point
        :type noise_variance_base: float64 >= 0.0
        :param hyperparameter_interval: interval from which to draw hyperparameters (uniform random)
        :type hyperparameter_interval: non-empty ClosedInterval
        :param lower_bound_interval: interval from which to draw domain lower bounds (uniform random)
        :type lower_bound_interval: non-empty ClosedInterval; cannot overlap with upper_bound_interval
        :param upper_bound_interval: interval from which to draw domain upper bounds (uniform random)
        :type upper_bound_interval: non-empty ClosedInterval; cannot overlap with lower_bound_interval
        :param covariance_class: the type of covariance to use when building the GP
        :type covariance_class: type object of covariance_interface.CovarianceInterface (or one of its subclasses)
        :param spatial_domain_class: the type of the domain that the GP lives in
        :type spatial_domain_class: type object of domain_interface.DomainInterface (or one of its subclasses)
        :param hyperparameter_domain_class: the type of the domain that the hyperparameters live in
        :type hyperparameter_domain_class: type object of domain_interface.DomainInterface (or one of its subclasses)
        :param gaussian_process_class: the type of the Gaussian Process to draw from
        :type gaussian_process_class: type object of gaussian_process_interface.GaussianProcessInterface (or one of its subclasses)

        """
        self.dim = dim
        self.num_hyperparameters = num_hyperparameters

        self.noise_variance_base = noise_variance_base
        self.num_sampled = num_sampled

        self.hyperparameter_interval = hyperparameter_interval
        self.lower_bound_interval = lower_bound_interval
        self.upper_bound_interval = upper_bound_interval

        self.covariance_class = covariance_class
        self.spatial_domain_class = spatial_domain_class
        self.hyperparameter_domain_class = hyperparameter_domain_class
        self.gaussian_process_class = gaussian_process_class

    @property
    def num_sampled(self):
        """Return the number of ``points_sampled`` that test GPs should be built with."""
        return self._num_sampled

    @num_sampled.setter
    def num_sampled(self, value):
        """Set num_sampled and resize dependent quantities (e.g., noise_variance)."""
        self._num_sampled = value
        self.noise_variance = numpy.full(self.num_sampled, self.noise_variance_base)


class GaussianProcessTestCase(OptimalLearningTestCase):

    """Base test case for tests that want to use random data generated from Gaussian Process(es).

    Users are required to set the *class variable* ``precompute_gaussian_process_data`` flag and define *class variables*:
    ``gp_test_environment_input`` and ``num_sampled_list`` (see base_setup() docstring).

    Using that info, base_setup will create the required test cases (in ``gp_test_environments``) for use in testing.

    The idea is that base_setup is run once per test class, so the (expensive) work of building GPs can be shared across
    numerous individual tests.

    """

    precompute_gaussian_process_data = False

    noise_variance_base = 0.0002
    dim = 3
    num_hyperparameters = dim + 1

    gp_test_environment_input = GaussianProcessTestEnvironmentInput(
        dim,
        num_hyperparameters,
        0,
        noise_variance_base=noise_variance_base,
        hyperparameter_interval=ClosedInterval(0.1, 1.3),
        lower_bound_interval=ClosedInterval(-2.0, 0.5),
        upper_bound_interval=ClosedInterval(2.0, 3.5),
        covariance_class=SquareExponential,
        spatial_domain_class=TensorProductDomain,
        hyperparameter_domain_class=TensorProductDomain,
        gaussian_process_class=GaussianProcess,
    )

    num_sampled_list = (1, 2, 3, 5, 10, 16, 20, 42)
    num_to_sample_list = (1, 2, 3, 8)

    @classmethod
    @pytest.fixture(autouse=True, scope='class')
    def base_setup(cls):
        """Build a Gaussian Process prior for each problem size in ``cls.num_sampled_list`` if precomputation is desired.

        **Requires**

        * cls.num_sampled_list: (*list of int*) problem sizes to consider
        * cls.gp_test_environment_input: (*GaussianProcessTestEnvironmentInput*) specification of how to build the
          gaussian process prior

        **Outputs**

        * cls.gp_test_environments: (*list of GaussianProcessTestEnvironment*) gaussian process data for each of the
          specified problem sizes (``cls.num_sampled_list``)

        """
        if cls.precompute_gaussian_process_data:
            cls.gp_test_environments = []
            for num_sampled in cls.num_sampled_list:
                cls.gp_test_environment_input.num_sampled = num_sampled
                cls.gp_test_environments.append(cls._build_gaussian_process_test_data(cls.gp_test_environment_input))

    @staticmethod
    def _build_gaussian_process_test_data(test_environment):
        """Build up a Gaussian Process randomly by repeatedly drawing from and then adding to the prior.

        :param test_environment: parameters describing how to construct a GP prior
        :type test_environment: GaussianProcessTestEnvironmentInput
        :return: gaussian process environments that can be used to run tests
        :rtype: GaussianProcessTestEnvironment

        """
        covariance = gp_utils.fill_random_covariance_hyperparameters(
            test_environment.hyperparameter_interval,
            test_environment.num_hyperparameters,
            covariance_type=test_environment.covariance_class,
        )

        domain_bounds = gp_utils.fill_random_domain_bounds(
            test_environment.lower_bound_interval,
            test_environment.upper_bound_interval,
            test_environment.dim,
        )
        domain = test_environment.spatial_domain_class(ClosedInterval.build_closed_intervals_from_list(domain_bounds))
        points_sampled = domain.generate_uniform_random_points_in_domain(test_environment.num_sampled)

        gaussian_process = gp_utils.build_random_gaussian_process(
            points_sampled,
            covariance,
            noise_variance=test_environment.noise_variance,
            gaussian_process_type=test_environment.gaussian_process_class,
        )
        return GaussianProcessTestEnvironment(domain, gaussian_process)
