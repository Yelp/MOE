# -*- coding: utf-8 -*-
"""Thin domain-related data containers that can be passed to cpp_wrappers.* functions/classes requiring domain data.

C++ domain objects currently do not expose their members to Python. So the classes in this file track the data necessary
for C++ calls to construct the matching C++ domain object.

"""
import copy

import moe.build.GPP as C_GP
from moe.optimal_learning.python.constant import TENSOR_PRODUCT_DOMAIN_TYPE, SIMPLEX_INTERSECT_TENSOR_PRODUCT_DOMAIN_TYPE
from moe.optimal_learning.python.interfaces.domain_interface import DomainInterface


class TensorProductDomain(DomainInterface):

    r"""Domain type for a tensor product domain.

    A d-dimensional tensor product domain is ``D = [x_0_{min}, x_0_{max}] X [x_1_{min}, x_1_{max}] X ... X [x_d_{min}, x_d_{max}]``
    At the moment, this is just a dummy container for the domain boundaries, since the C++ object currently does not expose its
    internals to Python.

    """

    domain_type = TENSOR_PRODUCT_DOMAIN_TYPE

    def __init__(self, domain_bounds):
        """Construct a TensorProductDomain that can be used with cpp_wrappers.* functions/classes.

        :param domain_bounds: the boundaries of a dim-dimensional tensor-product domain
        :type domain_bounds: iterable of dim ClosedInterval

        """
        self._domain_bounds = copy.deepcopy(domain_bounds)
        self._domain_type = C_GP.DomainTypes.tensor_product

    @property
    def dim(self):
        """Return the number of spatial dimensions."""
        return len(self._domain_bounds)

    def get_json_serializable_info(self, minimal=False):
        """Create and return a domain_info dictionary of this domain object.

        :param minimal: True for all domain contents; False for ``domain_type`` and ``dim`` only
        :type minimal: bool
        :return: dict representation of this domain
        :rtype: dict

        """
        response = {
            'domain_type': self.domain_type,
            'dim': self.dim,
        }
        if not minimal:
            response['domain_bounds'] = self._domain_bounds

        return response

    # HACK: Until we can build C++ domain objects from Python, we need to be able to hand C++ enough data to reconstruct domains.
    @property
    def domain_bounds(self):
        """Return the [min, max] bounds for each spatial dimension."""
        return self._domain_bounds

    def check_point_inside(self, point):
        r"""Check if a point is inside the domain/on its boundary or outside.

        We do not currently expose a C++ endpoint for this call; see :mod:`moe.optimal_learning.python.interfaces.domain_interface` for interface specification.

        """
        raise NotImplementedError("C++ wrapper currently does not support domain member functions.")

    def get_bounding_box(self):
        """Return a list of ClosedIntervals representing a bounding box for this domain."""
        return copy.copy(self._domain_bounds)

    def get_constraint_list(self):
        """Return a list of lambda functions expressing the domain bounds as linear constraints. Used by COBYLA."""
        raise NotImplementedError("Constraints are not yet implemented for C++.")

    def generate_random_point_in_domain(self, random_source=None):
        """Generate ``point`` uniformly at random such that ``self.check_point_inside(point)`` is True.

        We do not currently expose a C++ endpoint for this call; see :mod:`moe.optimal_learning.python.interfaces.domain_interface` for interface specification.

        """
        raise NotImplementedError("C++ wrapper currently does not support domain member functions.")

    def generate_uniform_random_points_in_domain(self, num_points, random_source=None):
        r"""Generate ``num_points`` uniformly distributed points from the domain.

        We do not currently expose a C++ endpoint for this call; see :mod:`moe.optimal_learning.python.interfaces.domain_interface` for interface specification.

        """
        raise NotImplementedError("C++ wrapper currently does not support domain member functions.")

    def compute_update_restricted_to_domain(self, max_relative_change, current_point, update_vector):
        r"""Compute a new update so that CheckPointInside(``current_point`` + ``new_update``) is true.

        We do not currently expose a C++ endpoint for this call; see :mod:`moe.optimal_learning.python.interfaces.domain_interface` for interface specification.

        """
        raise NotImplementedError("C++ wrapper currently does not support domain member functions.")


class SimplexIntersectTensorProductDomain(DomainInterface):

    r"""Domain class for the intersection of the unit simplex with an arbitrary tensor product domain.

    At the moment, this is just a dummy container for the domain boundaries, since the C++ object currently does not expose its
    internals to Python.

    This object has a TensorProductDomain object as a data member and uses its functions when possible.
    See TensorProductDomain for what that means.

    The unit d-simplex is defined as the set of x_i such that:

    1. ``x_i >= 0 \forall i  (i ranging over dimension)``
    2. ``\sum_i x_i <= 1``

    (Implying that ``x_i <= 1 \forall i``)

    ASSUMPTION: most of the volume of the tensor product region lies inside the simplex region.

    """

    domain_type = SIMPLEX_INTERSECT_TENSOR_PRODUCT_DOMAIN_TYPE

    def __init__(self, domain_bounds):
        """Construct a SimplexIntersectTensorProductDomain that can be used with cpp_wrappers.* functions/classes.

        :param domain_bounds: the boundaries of a dim-dimensional tensor-product domain.
        :type domain_bounds: iterable of dim ClosedInterval

        """
        self._tensor_product_domain = TensorProductDomain(domain_bounds)
        self._domain_type = C_GP.DomainTypes.simplex

    @property
    def dim(self):
        """Return the number of spatial dimensions."""
        return self._tensor_product_domain.dim()

    # HACK: Until we can build C++ domain objects from Python, we need to be able to hand C++ enough data to reconstruct domains.
    @property
    def domain_bounds(self):
        """Return the [min, max] bounds for each spatial dimension."""
        return self._tensor_product_domain._domain_bounds

    def check_point_inside(self, point):
        r"""Check if a point is inside the domain/on its boundary or outside.

        We do not currently expose a C++ endpoint for this call; see :mod:`moe.optimal_learning.python.interfaces.domain_interface` for interface specification.

        """
        raise NotImplementedError("C++ wrapper currently does not support domain member functions.")

    def generate_uniform_random_points_in_domain(self, num_points, random_source):
        r"""Generate AT MOST ``num_points`` uniformly distributed points from the domain.

        We do not currently expose a C++ endpoint for this call; see :mod:`moe.optimal_learning.python.interfaces.domain_interface` for interface specification.

        """
        raise NotImplementedError("C++ wrapper currently does not support domain member functions.")

    def compute_update_restricted_to_domain(self, max_relative_change, current_point, update_vector):
        r"""Compute a new update so that CheckPointInside(``current_point`` + ``new_update``) is true.

        We do not currently expose a C++ endpoint for this call; see :mod:`moe.optimal_learning.python.interfaces.domain_interface` for interface specification.

        """
        raise NotImplementedError("C++ wrapper currently does not support domain member functions.")
