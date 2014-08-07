/*!
  \file gpp_domain.hpp
  \rst
  This file contains "DomainType" classes that specify different kinds of domains (e.g., TensorProduct, Simplex).  These are
  currently used to describe domain limits for optimizers (defined in gpp_optimization, used by gpp_math, gpp_model_selection).

  Currently, we only support domains with planar (linear) boundaries.

  Each domain provides functions to describe the set of boundary planes, check whether a point is inside/outside, generate
  random points inside, and limit updates (from optimizers) so that a path stays inside the domain.

  See gpp_geometry.hpp for how to specify a plane.
\endrst*/

#ifndef MOE_OPTIMAL_LEARNING_CPP_GPP_DOMAIN_HPP_
#define MOE_OPTIMAL_LEARNING_CPP_GPP_DOMAIN_HPP_

#include <cmath>

#include <algorithm>
#include <limits>
#include <vector>

#include "gpp_common.hpp"
#include "gpp_exception.hpp"
#include "gpp_geometry.hpp"
#include "gpp_random.hpp"

namespace optimal_learning {

/*!\rst
  Enumerating domains for convenience. Useful for selecting which tests to run and also
  used by the Python interface to communicate domain type to C++.
\endrst*/
enum class DomainTypes {
  //! TensorProductDomain
  kTensorProduct = 0,
  //! SimplexIntersectTensorProductDomain
  kSimplex = 1,
};

/*!\rst
  A dummy domain; commonly paired with the NullOptimizer. Use when domain is irrelevant.

  It does not track any member data and claims all points are inside.
\endrst*/
class DummyDomain {
 public:
  //! string name of this domain for logging
  constexpr static char const * kName = "dummy_domain";

  /*!\rst
    Always returns true: DummyDomain contains all points.

    \param
      :point[dim]: point to check
    \return
      true if point is inside the domain or on its boundary, false otherwise
  \endrst*/
  bool CheckPointInside(double const * restrict OL_UNUSED(point)) const OL_NONNULL_POINTERS OL_WARN_UNUSED_RESULT {
    return true;
  }
};

/*!\rst
  Domain type for a tensor product domain.

  A d-dimensional tensor product domain is ``D = [x_0_{min}, x_0_{max}] X [x_1_{min}, x_1_{max}] X ... X [x_d_{min}, x_d_{max}]``
\endrst*/
class TensorProductDomain {
  //! attempt to scale down the step-size (or distance to wall) by this factor when a domain-exiting (i.e., invalid) step is requested
  static constexpr double kInvalidStepScaleFactor = 0.5;

 public:
  //! string name of this domain for logging
  constexpr static char const * kName = "tensor_product";

  TensorProductDomain() = delete;  // no default ctor; dim = 0 doesn't reallly make sense as a default

  /*!\rst
    Constructs a TensorProductDomain.

    \param
      :domain[dim]: array of ClosedInterval specifying the boundaries of a dim-dimensional tensor-product domain.
      :dim_in: number of spatial dimensions
  \endrst*/
  TensorProductDomain(ClosedInterval const * restrict domain, int dim_in);

  int dim() const OL_PURE_FUNCTION OL_WARN_UNUSED_RESULT {
    return dim_;
  }

  /*!\rst
    Explicitly set the domain boundaries. Assumes specified domain is non-empty.

    \param
       :domain[dim]: array of ClosedInterval specifying the boundaries of a dim-dimensional tensor-product domain.
  \endrst*/
  void SetDomain(ClosedInterval const * restrict domain) OL_NONNULL_POINTERS;

  /*!\rst
    Maximum number of planes that define the boundary of this domain.
    Used for testing.

    This result is exact.

    \return
      max number of planes defining the boundary of this domain
  \endrst*/
  int GetMaxNumberOfBoundaryPlanes() const OL_PURE_FUNCTION OL_WARN_UNUSED_RESULT {
    return 2*dim_;
  }

  /*!\rst
    Fills an input array with all bounding planes of this domain.
    See struct Plane in gpp_geometry.hpp for how to specify a plane.
    Used for testing.

    Let max_num_bound = GetMaxNumberOfBoundaryPlanes()

    \param
      :planes[max_num_bound]: properly allocated space: max_num_bound Plane objects in dim spatial dimensions
    \output
      :planes[max_num_bound]: array of planes of this domain
  \endrst*/
  void GetBoundaryPlanes(Plane * restrict planes) const OL_NONNULL_POINTERS;

  /*!\rst
    Check if a point is inside the domain/on its boundary or outside.

    \param
      :point[dim]: point to check
    \return
      true if point is inside the domain or on its boundary, false otherwise
  \endrst*/
  bool CheckPointInside(double const * restrict point) const OL_NONNULL_POINTERS OL_WARN_UNUSED_RESULT {
    return CheckPointInHypercube(domain_.data(), point, dim_);
  }

  /*!\rst
    Generates "point" such that CheckPointInside(point) returns true.

    \param
      :uniform_generator[1]: a UniformRandomGenerator object providing the random engine for uniform random numbers
      :random_point[dim]: properly sized array
    \output
      :uniform_generator[1]: UniformRandomGenerator object will have its state changed due to random draws
      :random_point[dim]: point with coordinates inside the domain (left in invalid state if fcn returns false)
    \return
      true if point generation succeeded
  \endrst*/
  bool GeneratePointInDomain(UniformRandomGenerator * uniform_generator,
                             double * restrict random_point) const OL_NONNULL_POINTERS;

  /*!\rst
    Generates num_points points in the domain (i.e., such that CheckPointInside(point) returns true).  The points
    will be uniformly distributed.

    \param
      :num_points: number of random points to generate
      :uniform_generator[1]: a UniformRandomGenerator object providing the random engine for uniform random numbers
      :random_points[dim][num_points]: properly sized array
    \output
      :uniform_generator[1]: UniformRandomGenerator object will have its state changed due to random draws
      :random_points[dim][num_points]: point with coordinates inside the domain
    \return
      number of points generated (always num_points; ok to not use this result)
  \endrst*/
  int GenerateUniformPointsInDomain(int num_points, UniformRandomGenerator * uniform_generator,
                                    double * restrict random_points) const OL_NONNULL_POINTERS;

  /*!\rst
    Changes update_vector so that:

      ``point_new = point + update_vector``

    has coordinates such that ``CheckPointInside(point_new)`` returns true.

    ``update_vector`` is UNMODIFIED if point_new is already inside the domain.

    .. Note:: we modify update_vector (instead of returning ``point_new``) so that further update
      limiting/testing may be performed.

    \param
      :max_relative_change: max change allowed per update (as a relative fraction of current distance to boundary)
      :current_point[dim]: starting point
      :update_vector[dim]: proposed update
    \output
      :update_vector[dim]: modified update so that the final point remains inside the domain
  \endrst*/
  void LimitUpdate(double max_relative_change, double const * restrict current_point,
                   double * restrict update_vector) const OL_NONNULL_POINTERS;

 private:
  //! the number of spatial dimensions of this domain
  int dim_;
  //! the list of ClosedInterval that define the boundaries of this tensor product domain
  std::vector<ClosedInterval> domain_;
};

/*!\rst
  Domain class for the intersection of the unit simplex with an arbitrary tensor product domain.  To that end,
  this object has a TensorProductDomain object as a data member and uses its functions when possible.

  See TensorProductDomain for what that means.
  The unit d-simplex is defined as the set of ``x_i`` such that:

  1. ``x_i >= 0 \forall i  (i ranging over dimension)``
  2. ``\sum_i x_i <= 1``

  (Implying that ``x_i <= 1 \forall i``)

  ASSUMPTION: most of the volume of the tensor product region lies inside the simplex region.
\endrst*/
class SimplexIntersectTensorProductDomain {
  //! GenerateUniformPointsInDomain() is happy if ratio*requested_points number of points is generated
  static constexpr double kPointGenerationRatio = 0.9;
  //! in GenerateUniformPointsInDomain(), if the ratio of valid points is > this, we retry requesting 1/ratio points. otherwise we retry with 5x the points (i.e., flooring ratio at 5)
  static constexpr double kValidPointRatioFloor = 0.2;
  //! 1/kValidPointRatioFloor; the most we'll increase the number of requested points on a retry
  static constexpr int kMaxPointRatioGrowth = 5;
  //! attempt to scale down the step-size (or distance to wall) by this factor when a domain-exiting (i.e., invalid) step is requested
  static constexpr double kInvalidStepScaleFactor = 0.5;
  //! small tweak to relative_change (to prevent max_relative_change == 1.0 exactly; see LimitUpdate comments)
  static constexpr double kRelativeChangeEpsilonTweak = 4*std::numeric_limits<double>::epsilon();

 public:
  //! string name of this domain for logging
  constexpr static char const * kName = "simplex_tensor_product";

  SimplexIntersectTensorProductDomain() = delete;  // no default ctor; dim = 0 doesn't reallly make sense as a default

  /*!\rst
    Constructs a SimplexIntersectTensorProductDomain.  The bounds of the tensor product region are specified through
    the "domain" input, just as with TensorProductDomain.

    \param
      :domain[dim]: array of ClosedInterval specifying the boundaries of a dim-dimensional tensor-product domain.
      :dim_in: number of spatial dimensions
  \endrst*/
  SimplexIntersectTensorProductDomain(ClosedInterval const * restrict domain, int dim_in);

  int dim() const OL_PURE_FUNCTION OL_WARN_UNUSED_RESULT {
    return dim_;
  }

  /*!\rst
    Maximum number of planes that define the boundary of this domain.
    Used for testing.

    This result is NOT exact.

    \return
      max number of planes defining the boundary of this domain
  \endrst*/
  int GetMaxNumberOfBoundaryPlanes() const OL_PURE_FUNCTION OL_WARN_UNUSED_RESULT;

  /*!\rst
    Fills an input array with all bounding planes of this domain.
    See struct Plane in gpp_geometry.hpp for how to specify a plane.
    Used for testing.

    Let max_num_bound = GetMaxNumberOfBoundaryPlanes()

    \param
      :planes[max_num_bound]: properly allocated space: max_num_bound Plane objects in dim spatial dimensions
    \output
      :planes[max_num_bound]: array of planes of this domain
  \endrst*/
  void GetBoundaryPlanes(Plane * restrict planes) const OL_NONNULL_POINTERS;

  /*!\rst
    Check if a point is inside the domain/on its domain or outside

    \param
      :point[dim]: point to check
    \return
      true if point is inside the domain or on its boundary, false otherwise
  \endrst*/
  bool CheckPointInside(double const * restrict point) const OL_NONNULL_POINTERS OL_WARN_UNUSED_RESULT {
    return tensor_product_domain_.CheckPointInside(point) && CheckPointInUnitSimplex(point, dim_);
  }

  /*!\rst
    Generates "point" such that CheckPointInside(point) returns true.

    Uses rejection sampling so point generation may fail.

    \param
      :uniform_generator[1]: a UniformRandomGenerator object providing the random engine for uniform random numbers
      :random_point[dim]: properly sized array
    \output
      :uniform_generator[1]: UniformRandomGenerator object will have its state changed due to random draws
      :random_point[dim]: point with coordinates inside the domain (left in invalid state if fcn returns false)
    \return
      true if point generation succeeded
  \endrst*/
  bool GeneratePointInDomain(UniformRandomGenerator * uniform_generator,
                             double * restrict random_point) const OL_NONNULL_POINTERS OL_WARN_UNUSED_RESULT;

  /*!\rst
    Generates AT MOST num_points points in the domain (i.e., such that CheckPointInside(point) returns true).  The points
    will be uniformly distributed.

    Uses rejection sampling so we are not guaranteed to generate num_points samples.

    \param
      :num_points: number of random points to generate
      :uniform_generator[1]: a UniformRandomGenerator object providing the random engine for uniform random numbers
      :random_points[dim][num_points]: properly sized array
    \output
      :uniform_generator[1]: UniformRandomGenerator object will have its state changed due to random draws
      :random_points[dim][num_points]: point with coordinates inside the domain
    \return
      number of points actually generated
  \endrst*/
  int GenerateUniformPointsInDomain(int num_points, UniformRandomGenerator * uniform_generator,
                                    double * restrict random_points) const OL_NONNULL_POINTERS OL_WARN_UNUSED_RESULT;

  /*!\rst
    Changes update_vector so that:

      ``point_new = point + update_vector``

    has coordinates such that ``CheckPointInside(point_new)`` returns true.

    ``update_vector`` is UNMODIFIED if point_new is already inside the domain.

    .. Note:: we modify update_vector (instead of returning ``point_new``) so that further update
      limiting/testing may be performed.

    \param
      :max_relative_change: max change allowed per update (as a relative fraction of current distance to boundary)
      :current_point[dim]: starting point
      :update_vector[dim]: proposed update
    \output
      :update_vector[dim]: modified update so that the final point remains inside the domain
  \endrst*/
  void LimitUpdate(double max_relative_change, double const * restrict current_point,
                   double * restrict update_vector) const OL_NONNULL_POINTERS;

 private:
  //! the number of spatial dimensions of this domain
  int dim_;
  //! the tensor product domain to intersect with
  TensorProductDomain tensor_product_domain_;
  //! the plane defining the simplex
  Plane simplex_plane_;
};

/*!\rst
  A generic domain type for simultaneously manipulating ``num_repeats`` points in a "regular" domain (the kernel).

  .. Note:: Comments in this class are copied to RepeatedDomain in optimal_learning/python/repated_domain.py.

  .. Note:: the kernel domain is *not* copied. Instead, the kernel functions are called
    ``num_repeats`` times in a loop. In some cases, data reordering is also necessary
    to preserve the output properties (e.g., uniform distribution).

  For some use cases (e.g., q,p-EI optimization with ``q > 1``), we need to simultaneously
  manipulate several points within the same domain. To support this use case, we have
  the ``RepeatedDomain``, a light-weight wrapper around any ``DomainType`` object
  that kernalizes that object's functionality.

  In general, kernel domain operations need be performed ``num_repeats`` times, once
  for each point. This class hides the looping logic so that use cases like various
  Optimizer implementations (gpp_optimization.hpp) do not need to be explicitly aware
  of whether they are optimizing 1 point or 50 points. Instead, an optimizable
  Evaluator/State pair provides GetProblemSize() and appropriately sized gradient information.
  Coupled with ``RepeatedDomain``, Optimizers can remain oblivious.

  In simpler terms, say we want to solve 5,0-EI in a parameter-space of dimension 3.
  So we would have 5 points moving around in a 3D space. The 3D space, whatever it is,
  is the kernel domain. We "repeat" the kernel 5 times; in practice this mostly amounts to
  simple loops around kernel functions and sometimes data reordering is also needed.

  .. Note:: this operation is more complex than just working in a higher dimensional space.
    3 points in a 2D simplex is not the same as 1 point in a 6D simplex; e.g.,
    ``[(0.5, 0.5), (0.5, 0.5), (0.5, 0.5)]`` is valid in the first scenario but not in the second.

  Where the member domain takes ``kernel_input``, this class's members take an array with
  of ``num_repeats`` data with the same size as ``kernel_input``, ordered sequentially. So
  if we have ``kernel_input[dim][num_points]``, we now have
  ``repeated_input[dim][num_points][num_repeats]``. The same is true for outputs.

  For example, ``CheckPointInside()`` calls the kernel domain's ``CheckPointInside()``
  function ``num_repeats`` times, returning True only if all ``num_repeats`` input
  points are inside the kernel domain.
\endrst*/
template <typename DomainType_>
class RepeatedDomain {
 public:
  using DomainType = DomainType_;

  RepeatedDomain() = delete;  // no default ctor; it makes no sense to specify no domain to repeat

  /*!\rst
    Construct a RepeatedDomain object, which kernalizes and ``repeats`` an input ``DomainType`` object.

    .. Note:: this class maintains a *pointer* to the input domain. Do not let the domain object go
      out of scope before this object goes out of scope.

    \param
      :domain: the domain to repeat
      :num_repeats: number of times to repeat the input domain
  \endrst*/
  RepeatedDomain(const DomainType& domain, int num_repeats_in) : num_repeats_(num_repeats_in), domain_(&domain) {
    if (num_repeats_ <= 0) {
      OL_THROW_EXCEPTION(LowerBoundException<int>, "num_repeats must be positive.", num_repeats_, 1);
    }
  }

  int dim() const OL_PURE_FUNCTION OL_WARN_UNUSED_RESULT {
    return domain_->dim();
  }

  int num_repeats() const OL_PURE_FUNCTION OL_WARN_UNUSED_RESULT {
    return num_repeats_;
  }

  /*!\rst
    Check if a point is inside the domain/on its domain or outside

    \param
      :point[dim][num_repeats]: point to check
    \return
      true if point is inside the domain or on its boundary, false otherwise
  \endrst*/
  bool CheckPointInside(double const * restrict point) const OL_NONNULL_POINTERS OL_WARN_UNUSED_RESULT {
    for (int i = 0; i < num_repeats_; ++i) {
      if (domain_->CheckPointInside(point) == false) {
        return false;
      }
      point += dim();
    }
    return true;
  }

  /*!\rst
    Generates "point" such that CheckPointInside(point) returns true.

    May use rejection sampling so point generation may fail.

    \param
      :uniform_generator[1]: a UniformRandomGenerator object providing the random engine for uniform random numbers
      :random_point[dim][num_repeats]: properly sized array
    \output
      :uniform_generator[1]: UniformRandomGenerator object will have its state changed due to random draws
      :random_point[dim][num_repeats]: point with coordinates inside the domain (left in invalid state if fcn returns false)
    \return
      true if point generation succeeded
  \endrst*/
  bool GeneratePointInDomain(UniformRandomGenerator * uniform_generator, double * restrict random_point) const OL_NONNULL_POINTERS OL_WARN_UNUSED_RESULT {
    for (int i = 0; i < num_repeats_; ++i) {
      if (unlikely(domain_->GeneratePointInDomain(uniform_generator, random_point) == false)) {
        return false;
      }
      random_point += dim();
    }
    return true;
  }

  /*!\rst
    Generates AT MOST num_points points in the domain (i.e., such that CheckPointInside(point) returns true).  The points
    will be uniformly distributed.

    May use rejection sampling so we are not guaranteed to generate num_points samples.

    \param
      :num_points: number of random points to generate
      :uniform_generator[1]: a UniformRandomGenerator object providing the random engine for uniform random numbers
      :random_points[dim][num_repeats][num_points]: properly sized array
    \output
      :uniform_generator[1]: UniformRandomGenerator object will have its state changed due to random draws
      :random_points[dim][num_repeats][num_points]: point with coordinates inside the domain
    \return
      number of points actually generated
  \endrst*/
  int GenerateUniformPointsInDomain(int num_points, UniformRandomGenerator * uniform_generator, double * restrict random_points) const OL_NONNULL_POINTERS OL_WARN_UNUSED_RESULT {
    int _dim = dim();
    std::vector<double> temp(_dim*num_points);
    int num_points_actual = num_points;
    int num_points_temp;

    // Generate num_repeats sets of points from some sampling (e.g., LHC)
    // Then we "transpose" the output ordering: the i-th point in RepeatedDomain is constructed
    // from the i-th points of LHC_1 ... LHC_{num_repeats}
    for (int i = 0; i < num_repeats_; ++i) {
      // Only generate as many points as we can use (if a previous iteration came up short, generate fewer points)
      num_points_temp = domain_->GenerateUniformPointsInDomain(num_points_actual, uniform_generator, temp.data());
      // Since GenerateUniformPointsInDomain() may not always return num_points
      // points, we need to make sure we only use the valid results
      num_points_actual = std::min(num_points_actual, num_points_temp);

      // "Transpose" the data ordering
      for (int j = 0; j < num_points_actual; ++j) {
        for (int k = 0; k < _dim; ++k) {
          random_points[j*num_repeats_*_dim + i*_dim + k] = temp[j*_dim + k];
        }
      }
    }
    // We can only use the smallest num_points that came out of our draws
    return num_points_actual;
  }

  /*!\rst
    Changes update_vector so that:

      ``point_new = point + update_vector``

    has coordinates such that ``CheckPointInside(point_new)`` returns true.

    ``update_vector`` is UNMODIFIED if point_new is already inside the domain.

    .. Note:: we modify update_vector (instead of returning ``point_new``) so that further update
      limiting/testing may be performed.

    \param
      :max_relative_change: max change allowed per update (as a relative fraction of current distance to boundary)
      :current_point[dim][num_repeats]: starting point
      :update_vector[dim][num_repeats]: proposed update
    \output
      :update_vector[dim][num_repeats]: modified update so that the final point remains inside the domain
  \endrst*/
  void LimitUpdate(double max_relative_change, double const * restrict current_point, double * restrict update_vector) const OL_NONNULL_POINTERS {
    for (int i = 0; i < num_repeats_; ++i) {
      domain_->LimitUpdate(max_relative_change, current_point + i*dim(), update_vector + i*dim());
    }
  }

 private:
  //! number of times to repeat the input domain
  int num_repeats_;
  //! pointer to the domain to repeat
  const DomainType * restrict domain_;
};

}  // end namespace optimal_learning

#endif  // MOE_OPTIMAL_LEARNING_CPP_GPP_DOMAIN_HPP_
