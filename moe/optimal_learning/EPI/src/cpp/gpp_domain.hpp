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

#ifndef OPTIMAL_LEARNING_EPI_SRC_CPP_GPP_DOMAIN_HPP_
#define OPTIMAL_LEARNING_EPI_SRC_CPP_GPP_DOMAIN_HPP_

#include <cmath>

#include <algorithm>
#include <limits>
#include <vector>

#include "gpp_common.hpp"
#include "gpp_exception.hpp"
#include "gpp_geometry.hpp"
#include "gpp_logging.hpp"
#include "gpp_random.hpp"

namespace optimal_learning {

/*!\rst
  Enumerating domains for convenience. Useful for selecting which tests to run and also
  used by the Python interface to communicate domain type to C++.
\endrst*/
enum class DomainTypes {
  kTensorProduct = 0,  // TensorProductDomain
  kSimplex = 1,  // SimplexIntersectTensorProductDomain
};

/*!\rst
  Domain type for a tensor product domain.

  A d-dimensional tensor product domain is ``D = [x_0_{min}, x_0_{max}] X [x_1_{min}, x_1_{max}] X ... X [x_d_{min}, x_d_{max}]``
\endrst*/
class TensorProductDomain {
  static constexpr double kInvalidStepScaleFactor = 0.5;  // attempt to scale down the step-size (or distance to wall) by this factor when a domain-exiting (i.e., invalid) step is requested

 public:
  TensorProductDomain() = delete;  // no default ctor; dim = 0 doesn't reallly make sense as a default

  /*!\rst
    Constructs a TensorProductDomain.

    INPUTS:
    domain[dim]: array of ClosedInterval specifying the boundaries of a dim-dimensional tensor-product domain.
    dim_in: number of spatial dimensions
  \endrst*/
  TensorProductDomain(ClosedInterval const * restrict domain, int dim_in) : dim_(dim_in), domain_(domain, domain + dim_) {
    bool is_empty = std::any_of(domain_.begin(), domain_.end(), [](ClosedInterval interval) {
      return interval.IsEmpty();
    });
    if (is_empty) {
#ifdef OL_ERROR_PRINT
      OL_ERROR_PRINTF("WARNING: Tensor product region is EMPTY!\n");
      PrintDomainBounds(domain_.data(), dim_);
#endif
      OL_THROW_EXCEPTION(RuntimeException, "Tensor product region is EMPTY.");
    }
  }

  int dim() const OL_PURE_FUNCTION OL_WARN_UNUSED_RESULT {
    return dim_;
  }

  /*!\rst
    Explicitly set the domain boundaries. Assumes specified domain is non-empty.

    INPUTS:
    domain[dim]: array of ClosedInterval specifying the boundaries of a dim-dimensional tensor-product domain.
  \endrst*/
  void SetDomain(ClosedInterval const * restrict domain) OL_NONNULL_POINTERS {
    std::copy(domain, domain + dim_, domain_.begin());
  }

  /*!\rst
    Maximum number of planes that define the boundary of this domain.
    Used for testing.

    This result is exact.

    RETURNS:
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
    INPUTS:
    planes[max_num_bound]: properly allocated space: max_num_bound Plane objects in dim spatial dimensions
    OUTPUTS:
    planes[max_num_bound]: array of planes of this domain
  \endrst*/
  void GetBoundaryPlanes(Plane * restrict planes) const OL_NONNULL_POINTERS {
    int num_planes = GetMaxNumberOfBoundaryPlanes();
    std::fill(planes, planes + num_planes, Plane(dim_));
    for (int i = 0; i < dim_; ++i) {
      // "left" boundary
      planes[2*i + 0].unit_normal[i] = -1.0;
      planes[2*i + 0].offset = domain_[i].min;
      // "right" boundary
      planes[2*i + 1].unit_normal[i] = 1.0;
      planes[2*i + 1].offset = -domain_[i].max;
    }
  }

  /*!\rst
    Check if a point is inside the domain/on its boundary or outside.

    INPUTS:
    point[dim_]: point to check
    RETURNS:
    true if point is inside the domain or on its boundary, false otherwise
  \endrst*/
  bool CheckPointInside(double const * restrict point) const OL_NONNULL_POINTERS OL_WARN_UNUSED_RESULT {
    return CheckPointInHypercube(domain_.data(), point, dim_);
  }

  /*!\rst
    Generates "point" such that CheckPointInside(point) returns true.

    INPUTS:
    uniform_generator[1]: a UniformRandomGenerator object providing the random engine for uniform random numbers
    random_point[dim_]: properly sized array
    OUTPUTS:
    uniform_generator[1]: UniformRandomGenerator object will have its state changed due to random draws
    random_point[dim_]: point with coordinates inside the domain (left in invalid state if fcn returns false)
    RETURNS:
    true if point generation succeeded
  \endrst*/
  bool GeneratePointInDomain(UniformRandomGenerator * uniform_generator, double * restrict random_point) const OL_NONNULL_POINTERS {
    ComputeRandomPointInDomain(domain_.data(), dim_, uniform_generator, random_point);
    return true;
  }

  /*!\rst
    Generates num_points points in the domain (i.e., such that CheckPointInside(point) returns true).  The points
    will be uniformly distributed.

    INPUTS:
    num_points: number of random points to generate
    uniform_generator[1]: a UniformRandomGenerator object providing the random engine for uniform random numbers
    random_points[dim_]: properly sized array
    OUTPUTS:
    uniform_generator[1]: UniformRandomGenerator object will have its state changed due to random draws
    random_points[dim_][num_points]: point with coordinates inside the domain
    RETURNS:
    number of points generated (always num_points; ok to not use this result)
  \endrst*/
  int GenerateUniformPointsInDomain(int num_points, UniformRandomGenerator * uniform_generator, double * restrict random_points) const OL_NONNULL_POINTERS {
    ComputeLatinHypercubePointsInDomain(domain_.data(), dim_, num_points, uniform_generator, random_points);
    return num_points;
  }

  /*!\rst
    Changes update_vector so that:
      ``point_new = point + update_vector``
    has coordinates such that CheckPointInside(point_new) returns true.

    update_vector is UNMODIFIED if point_new is already inside the domain.

    Note: we modify update_vector (instead of returning point_new) so that further update
    limiting/testing may be performed.

    INPUTS:
    max_relative_change: max change allowed per update (as a relative fraction of current distance to boundary)
    current_point[dim_]: starting point
    update_vector[dim_]: proposed update
    OUTPUTS:
    update_vector[dim_]: modified update so that the final point remains inside the domain
  \endrst*/
  void LimitUpdate(double max_relative_change, double const * restrict current_point, double * restrict update_vector) const OL_NONNULL_POINTERS {
    for (int j = 0; j < dim_; ++j) {
      double desired_step = update_vector[j];

      double distance_to_boundary = std::fmin(current_point[j] - domain_[j].min, domain_[j].max - current_point[j]);

      if (unlikely(std::fabs(desired_step) > max_relative_change*distance_to_boundary)) {
        desired_step = std::copysign(max_relative_change*distance_to_boundary, desired_step);
      }

      double potential_next_coordinate = current_point[j] + desired_step;

      // this if-block fixes desired_step so that the resulting point will be inside the domain
      if (unlikely(potential_next_coordinate < domain_[j].min ||
                   potential_next_coordinate > domain_[j].max)) {
        // requesting a step that would lead us outside the domain
        // take whichever step is larger (but still lies in domain): half of the requested step OR half of the distance to wall
        if (potential_next_coordinate < domain_[j].min) {
          distance_to_boundary = domain_[j].min - current_point[j];
          if (current_point[j] + desired_step*kInvalidStepScaleFactor < domain_[j].min) {
            // desired_step/2 is not safe; step kInvalidStepScaleFactor * distance to wall instead
            desired_step = distance_to_boundary*kInvalidStepScaleFactor;
          } else {
            // desired_step*kInvalidStepScaleFactor is safe
            desired_step *= kInvalidStepScaleFactor;
          }
        } else {
          distance_to_boundary = domain_[j].max - current_point[j];
          if (current_point[j] + desired_step*kInvalidStepScaleFactor > domain_[j].max) {
            // desired_step/2 is not safe; step kInvalidStepScaleFactor * distance to wall instead
            desired_step = distance_to_boundary*kInvalidStepScaleFactor;
          } else {
            // desired_step*kInvalidStepScaleFactor is safe
            desired_step *= kInvalidStepScaleFactor;
          }
        }
      }

      update_vector[j] = desired_step;
    }
  }

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
  1) ``x_i >= 0 \forall i  (i ranging over dimension)``
  2) ``\sum_i x_i <= 1``
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
  SimplexIntersectTensorProductDomain() = delete;  // no default ctor; dim = 0 doesn't reallly make sense as a default

  /*!\rst
    Constructs a SimplexIntersectTensorProductDomain.  The bounds of the tensor product region are specified through
    the "domain" input, just as with TensorProductDomain.

    INPUTS:
    domain[dim]: array of ClosedInterval specifying the boundaries of a dim-dimensional tensor-product domain.
    dim_in: number of spatial dimensions
  \endrst*/
  SimplexIntersectTensorProductDomain(ClosedInterval const * restrict domain, int dim_in) : dim_(dim_in), tensor_product_domain_(domain, dim_), simplex_plane_(dim_) {
    // Equation for the unit simplex plane is: -1/sqrt(dim) + \sum_i 1/sqrt(dim)*x_i = 0
    std::fill(simplex_plane_.unit_normal.begin(), simplex_plane_.unit_normal.end(), 1.0/std::sqrt(static_cast<double>(dim_)));
    // a_0 is the same value but opposite sign as any entry of the unit_normal (see plane equation above).
    simplex_plane_.offset = -1.0 * simplex_plane_.unit_normal[0];

    // restrict tensor product domain if needed: it should never exceed the unit hypercube in any direction since
    // the unit hypercube is the unit simplex's bounding box
    std::vector<ClosedInterval> domain_local(domain, domain + dim_);
    double corner_sum = 0.0;
    bool is_empty = false;
    for (int i = 0; i < dim_; ++i) {
      domain_local[i].min = std::fmax(domain_local[i].min, 0.0);
      domain_local[i].max = std::fmin(domain_local[i].max, 1.0);

      is_empty |= domain_local[i].IsEmpty();

      // Ensure that the tensor product region and the simplex have non-empty intersection:
      // check that the coordinate-wise sum of the corner with smallest coordinates
      // ("lower left" corner) sums to < 1
      corner_sum += domain_local[i].min;
    }

    if (corner_sum >= 1.0 || is_empty) {
#ifdef OL_ERROR_PRINT
      OL_ERROR_PRINTF("WARNING: Intersected region is EMPTY\n");
      PrintDomainBounds(domain_local.data(), dim_);
#endif
      OL_THROW_EXCEPTION(BoundsException<double>, "Simplex/Tensor product intersection is EMPTY; 'lower left' corner coordinate sum out of bounds or bounding boxes do not intersect.", corner_sum, 0.0, 1.0);
    }

    tensor_product_domain_.SetDomain(domain_local.data());
  }

  int dim() const OL_PURE_FUNCTION OL_WARN_UNUSED_RESULT {
    return dim_;
  }

  /*!\rst
    Maximum number of planes that define the boundary of this domain.
    Used for testing.

    This result is NOT exact.

    RETURNS:
    max number of planes defining the boundary of this domain
  \endrst*/
  int GetMaxNumberOfBoundaryPlanes() const OL_PURE_FUNCTION OL_WARN_UNUSED_RESULT {
    // The intersection of tensor product region and simplex can have at most
    // 2*dim + 1 faces. This maximum is achieved when:
    // ___
    // |  \ <-- simplex clips the corner of the tensor product region but doesn't
    // |___|    entirely remove any edges
    // It can also have fewer faces:
    //  ____
    // |    |
    // |\   |
    // |__\_| <-- simplex cuts out 2 faces entirely
    return tensor_product_domain_.GetMaxNumberOfBoundaryPlanes() + 1;
  }

  /*!\rst
    Fills an input array with all bounding planes of this domain.
    See struct Plane in gpp_geometry.hpp for how to specify a plane.
    Used for testing.

    Let max_num_bound = GetMaxNumberOfBoundaryPlanes()
    INPUTS:
    planes[max_num_bound]: properly allocated space: max_num_bound Plane objects in dim spatial dimensions
    OUTPUTS:
    planes[max_num_bound]: array of planes of this domain
  \endrst*/
  void GetBoundaryPlanes(Plane * restrict planes) const OL_NONNULL_POINTERS {
    int num_planes = GetMaxNumberOfBoundaryPlanes();
    // first, the planes from the tensor-product part of the domain
    tensor_product_domain_.GetBoundaryPlanes(planes);

    // set the simplex's "diagonal" plane last
    planes[num_planes - 1] = simplex_plane_;
  }

  /*!\rst
    Check if a point is inside the domain/on its domain or outside

    INPUTS:
    point[dim_]: point to check
    RETURNS:
    true if point is inside the domain or on its boundary, false otherwise
  \endrst*/
  bool CheckPointInside(double const * restrict point) const OL_NONNULL_POINTERS OL_WARN_UNUSED_RESULT {
    return tensor_product_domain_.CheckPointInside(point) && CheckPointInUnitSimplex(point, dim_);
  }

  /*!\rst
    Generates "point" such that CheckPointInside(point) returns true.

    INPUTS:
    uniform_generator[1]: a UniformRandomGenerator object providing the random engine for uniform random numbers
    random_point[dim_]: properly sized array
    OUTPUTS:
    uniform_generator[1]: UniformRandomGenerator object will have its state changed due to random draws
    random_point[dim_]: point with coordinates inside the domain (left in invalid state if fcn returns false)
    RETURNS:
    true if point generation succeeded
  \endrst*/
  bool GeneratePointInDomain(UniformRandomGenerator * uniform_generator, double * restrict random_point) const OL_NONNULL_POINTERS OL_WARN_UNUSED_RESULT {
    bool point_found = false;
    // Attempt to generate a point via rejection sampling 10000 times, then give up.
    // The intersection between tensor product region and simplex can be *very* small so that rejection
    // sampling has a low rate of success. We do not want to wait "forever."
    for (int i = 0; i < 10000 && point_found == false; ++i) {
      tensor_product_domain_.GeneratePointInDomain(uniform_generator, random_point);
      point_found = CheckPointInUnitSimplex(random_point, dim_);  // stop if the point is inside the simplex too
    }
    return point_found;
  }

  /*!\rst
    Generates AT MOST num_points points in the domain (i.e., such that CheckPointInside(point) returns true).  The points
    will be uniformly distributed.

    Uses rejection sampling so we are not guaranteed to generate num_points samples.

    INPUTS:
    num_points: number of random points to generate
    uniform_generator[1]: a UniformRandomGenerator object providing the random engine for uniform random numbers
    random_points[dim_]: properly sized array
    OUTPUTS:
    uniform_generator[1]: UniformRandomGenerator object will have its state changed due to random draws
    random_points[dim_][num_points]: point with coordinates inside the domain
    RETURNS:
    number of points actually generated
  \endrst*/
  int GenerateUniformPointsInDomain(int num_points, UniformRandomGenerator * uniform_generator, double * restrict random_points) const OL_NONNULL_POINTERS OL_WARN_UNUSED_RESULT {
    // ASSUME: most of the tensor product domain lies inside the simplex domain
    // TODO(eliu): if the opposite is true (most of the simplex lies inside the tensor prod), then we need to reverse this
    //             sampling: draw a uniform sample from the simplex first, then reject based on the tensor product region
    //             (ticket 55780)
    int num_points_local = std::max(10, num_points);
    std::vector<double> random_points_local(num_points_local*dim_);

    const int max_num_attempts = 10;
    int num_points_generated = 0;
    for (int j = 0; j < max_num_attempts; ++j) {
      random_points_local.resize(num_points_local*dim_);
      num_points_generated = 0;
      // generate points in the tensor product domain
      num_points_local = tensor_product_domain_.GenerateUniformPointsInDomain(num_points_local, uniform_generator, random_points_local.data());

      double * current_random_point = random_points;
      double const * current_random_point_local = random_points_local.data();
      // now reject points that are not also in the simplex
      for (int i = 0; i < num_points_local; ++i) {
        if (CheckPointInUnitSimplex(current_random_point_local, dim_) == true) {
          std::copy(current_random_point_local, current_random_point_local + dim_, current_random_point);
          current_random_point += dim_;
          ++num_points_generated;
          if (unlikely(num_points_generated >= num_points)) {
            break;
          }
        }
        current_random_point_local += dim_;
      }

      // we are happy if kPointGenerationRatio of the requested points are generated
      if (static_cast<double>(num_points_generated) >= kPointGenerationRatio*static_cast<double>(num_points)) {
        break;
      } else {
        // how close did we come to meeting our goal?
        double generation_ratio = static_cast<double>(num_points_generated) / static_cast<double>(num_points);
        // we want to try again with enough buffer room to hopefully guarantee we meet the objective next time
        if (generation_ratio < kValidPointRatioFloor) {
          num_points_local *= kMaxPointRatioGrowth;  // limit the growth rate on retries; also avoids div by 0
        } else {
          num_points_local = static_cast<int>(std::ceil(static_cast<double>(num_points_local) / generation_ratio));
        }
      }
    }

    if (unlikely(num_points_generated > num_points)) {
      OL_ERROR_PRINTF("ERROR: %s generated too many points!\n", OL_CURRENT_FUNCTION_NAME);
    }

    return num_points_generated;
  }

  /*!\rst
    Changes update_vector so that:
      ``point_new = point + update_vector``
    has coordinates such that CheckPointInside(point_new) returns true.

    update_vector is UNMODIFIED if point_new is already inside the domain.

    Note: we modify update_vector (instead of returning point_new) so that further update
    limiting/testing may be performed.

    INPUTS:
    max_relative_change: max change allowed per update (as a relative fraction of current distance to boundary)
    current_point[dim_]: starting point
    update_vector[dim_]: proposed update
    OUTPUTS:
    update_vector[dim_]: modified update so that the final point remains inside the domain
  \endrst*/
  void LimitUpdate(double max_relative_change, double const * restrict current_point, double * restrict update_vector) const OL_NONNULL_POINTERS {
    // if the first geometry check (over the hypercube domain) sees max_relative_change = 1.0, it will snap you "directly"
    // to the boundary (if you would have exited the boundary).  Quotes b/c you may be slightly off the boundary (in or outside)
    // Inside is OK, but outside causes problems: we won't be outside by more than 1e-20 or so.  BUT when the simplex domain check
    // fires, it will attempt a similar correction (to move us 1.0e-20); problematically, this number will register as "0.0"
    // relatively if we have other coordinates larger than 1.0e-4 (machine precision issue).  So we stay outside the domain, which
    // is invalid (even though being outside by so little is not a problem and the solution is valid).

    // Easiest solution is just to tweak max_relative_change to be slightly less than 1.0 so we "never" end up *precisely*
    // on the boundary.
    if (max_relative_change == 1.0) {
      max_relative_change -= kRelativeChangeEpsilonTweak;
    }

    // we first limit the update within the tensor product domain, then check if we are also violating the simplex domain.
    // another approach would be to loop over all faces (2*dim + 1), computing DistanceToPlaneAlongVector in each;
    // this would preserve the gradient direction.
    tensor_product_domain_.LimitUpdate(max_relative_change, current_point, update_vector);

    double norm = VectorNorm(update_vector, dim_);
    std::vector<double> unit_dir(dim_);
    std::vector<double> temp_point(dim_);
    if (unlikely(norm == 0.0)) {
      norm = std::numeric_limits<double>::min();  // prevent divide by 0
    }
    for (int j = 0; j < dim_; ++j) {
      unit_dir[j] = update_vector[j] / norm;
      temp_point[j] = current_point[j] + update_vector[j];
    }

    // make sure the proposed next point, after the hypercube domain check, is inside the simplex domain.
    // if not, reduce the distance moved.
    if (unlikely(CheckPointInUnitSimplex(temp_point.data(), dim_) == false)) {
      // we already guaranteed the update is inside the tensor product region, which is a *subset* of the simplex's
      // bounding box.  thus we do not need to re-check [0,1] X [0,1] X ... X [0, 1].

      // the udpate MUST be outside *only* the diagonal face
      double min_distance = simplex_plane_.DistanceToPlaneAlongVector(current_point, unit_dir.data());
      if (unlikely(min_distance < 0.0)) {
        min_distance = 0.0;  // stop numerical precision issues
      }

      // adjust step size to only take us *half* the distance to the nearest boundary
      double step_under_relaxation = kInvalidStepScaleFactor * min_distance;

      // take the step
      for (int j = 0; j < dim_; ++j) {
        update_vector[j] = step_under_relaxation * unit_dir[j];
      }
    }
    // if we're already inside the simplex, then nothing to do; we have not modified update_vector
  }

 private:
  //! the number of spatial dimensions of this domain
  int dim_;
  //! the tensor product domain to intersect with
  TensorProductDomain tensor_product_domain_;
  //! the plane defining the simplex
  Plane simplex_plane_;
};

}  // end namespace optimal_learning

#endif  // OPTIMAL_LEARNING_EPI_SRC_CPP_GPP_DOMAIN_HPP_
