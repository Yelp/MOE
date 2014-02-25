// gpp_geometry.hpp
/*
  This file contains utilities for some simple problems in n-dimensional computational geometry.  For example,
  (orthogonal) distance from point to plane, point projection, hypercube/simplex intersection, etc.

  Unless indicated otherwise, we will specify a plane in dim-space by dim + 1 numbers.  The equation of plane is:
  a_0 + \sum_i n_i * x_i = 0, i = 1..dim
  Hence we can describe any plane as: [n_1, n_2, ..., n_{dim}, a_0]
  Here, n_vec = [n_1, ..., n_{dim}] is the (outward) normal vector.
  By convention, ||n_vec||_2 = 1 (UNIT normal).

  Recall that a plane is fully specified by a point r_0 and a normal vector n_vec. Then a point r is in the plane
  if and only if (r-r_0) \cdot n_vec = 0. Since r_0 is constant, we can precompute and store r_0 \cdot n_vec = -a_0.
*/

#ifndef OPTIMAL_LEARNING_EPI_SRC_CPP_GPP_GEOMETRY_HPP_
#define OPTIMAL_LEARNING_EPI_SRC_CPP_GPP_GEOMETRY_HPP_

#include <limits>
#include <vector>

#include "gpp_common.hpp"
#include "gpp_linear_algebra.hpp"

namespace optimal_learning {

/*
  Container to represent the mathematical notion of a closed interval, commonly written [a,b].
  The closed interval [a,b] is the set of all numbers x \in R such that a <= x <= b.
  Note that "closed" here indicates the interval *includes* both endpoints.
  An interval with a > b is considered empty.

  WARNING: *undefined behavior* if either endpoint is NaN or if the interval is 
           [+inf, +inf] or [-inf, -inf].
           None of these conditions make any sense mathematically either.

  This struct is "trivial" and "standard layout" and thus "POD" (in the C++11 sense).
  http://en.cppreference.com/w/cpp/types/is_pod
  http://stackoverflow.com/questions/4178175/what-are-aggregates-and-pods-and-how-why-are-they-special/7189821#7189821

  This struct is not an aggregate; list (aka brace) initialization and a 2-argument constructor are both available:
  ClosedInterval tmp(1.0, 2.0);  // this ctor makes it non-aggregate
  ClosedInterval tmp{1.0, 2.0};  // and brace-style (aka initializer list) inits also work
*/
struct ClosedInterval {
  /*
    Explicitly defaulted default constructor.
    Defining a custom ctor (below) disables the default ctor, so we explicitly default it.
    This is needed to maintain POD-ness.

    Note: this ctor cannot be declared constexpr because the implicit default ctor is not
    constexpr. It does not make sense in the same way that "constexpr double d;" is undefined.
  */
  ClosedInterval() = default;

  /*
    Constructs a ClosedInterval object with specified min, max.

    The presence of this ctor makes this object a non-aggregate, so brace-initialization
    follow list initialization rules (not aggregate initialization):
    http://en.cppreference.com/w/cpp/language/list_initialization

    INPUTS:
    min_in: left bound of the interval
    max_in: right bound of the interval
  */
  constexpr ClosedInterval(double min_in, double max_in) noexcept : min(min_in), max(max_in) {  // NOLINT(build/include_what_you_use) misinterpreting these as calls to std::min, max
  }

  /*
    Check if a value is inside this ClosedInterval.

    INPUTS:
    value: the value to check
    RETURNS:
    true if min <= value <= max
  */
  constexpr bool IsInside(double value) const noexcept OL_PURE_FUNCTION OL_WARN_UNUSED_RESULT {
    return (value >= min) & (value <= max);
  }

  /*
    Compute the length of this ClosedInterval; result can be negative (i.e., an empty interval).

    RETURNS:
    length of the interval
  */
  constexpr double Length() const noexcept OL_PURE_FUNCTION OL_WARN_UNUSED_RESULT {
    return max - min;
  }

  /*
    Checks whether the interval is \emptyset (empty, max < min).
    Equivalent to Length() >= 0.0.

    RETURNS:
    true if the interval is non-empty: max >= min
  */
  constexpr bool IsEmpty() const noexcept OL_PURE_FUNCTION OL_WARN_UNUSED_RESULT {
    return max < min;
  }

  double min, max;
};

/*
  Simple auxilliary function that checks if a point is within the given hypercube.

  INPUTS:
  domain[dim]: array of ClosedInterval specifying the boundaries of a dim-dimensional tensor-product domain.
  point[dim]: the point to check
  dim: the number of spatial dimensions
  OUTPUTS:
  true if the point is inside the specified tensor-product domain
*/
inline OL_WARN_UNUSED_RESULT OL_NONNULL_POINTERS bool CheckPointInHypercube(ClosedInterval const * restrict domain, double const * restrict point, int dim) noexcept {
  for (int i = 0; i < dim; ++i) {
    if (domain[i].IsInside(point[i]) == false) {
      return false;
    }
  }
  return true;
}

/*
  Checks if a point is inside/on the unit d-simplex.  A point x_i lies inside the unit d-simplex if:
  1) x_i >= 0 \forall i  (i ranging over dimension)
  2) \sum_i x_i <= 1
  (Implying that x_i <= 1 \forall i)

  INPUTS:
  point[dim]: point to check
  dim: number of dimensions
  OUTPUTS:
  true if the point lies inside/on the unit d-simplex
*/
inline OL_NONNULL_POINTERS OL_WARN_UNUSED_RESULT bool CheckPointInUnitSimplex(double const * restrict point, int dim) noexcept {
  static constexpr double wall_tolerance = 4*std::numeric_limits<double>::epsilon();  // being this far outside the simplex still counts as inside
  double sum = 0.0;
  for (int i = 0; i < dim; ++i) {
    if (point[i] < 0.0) {
      return false;
    }
    sum += point[i];
  }
  return (sum - wall_tolerance) <= 1.0;  // can be slightly beyond 1 to account for floating point issues
}

/*
  Signed, shortest distance from point to plane: + means the point is on the same half-space as the plane's normal vector

  INPUTS:
  point[dim]: point to compute distance from
  plane[dim+1]: plane to compute distance to; data ordered as specified in file docs
  dim: number of spatial dimensions
  RETURNS:
  signed, shortest distance from point to plane where + means the point and normal are in the same half-space
*/
inline OL_NONNULL_POINTERS OL_WARN_UNUSED_RESULT double OrthogonalDistanceToPlane(double const * restrict point, double const * restrict plane, int dim) noexcept {
  // formula: let p_1 = "point", p_0 be any point in the plane, and n be the normal vector
  // distance = |(p_1 - p_0) \cdot n|/||n||_2 = |p_1 \cdot n + a_0|/||n||_2 (b/c \sum_i n_i * p_0_i = -a_0)
  const double distance = DotProduct(point, plane, dim);
  return distance + plane[dim];  // plane[dim] is a_0 AND we assume ||n||_2 = 1.
}

/*
  Projects a point onto a plane.

  INPUTS:
  plane[dim+1]: plane to compute distance to; data ordered as specified in file docs
  dim: number of spatial dimensions
  point[dim]: point to project onto plane
  OUTPUTS:
  point[dim]: point projected onto plane
*/
inline OL_NONNULL_POINTERS void OrthogonalProjectionOntoPlane(double const * restrict plane, int dim, double * restrict point) noexcept {
  // formula: let d be the orthogonal, signed distance from point to plane (where + means the point lies in the half-space
  //          pointed to by the normal vector)
  // then we compute: projected_point = point - d*unit_normal, where unit_normal is the unit normal vector of the plane

  // It is also possible to parameterize the operation and do constrained optimization (in the space of the plane) and find
  // the point on the plane that is nearest to the specified point.  This is generally better-conditioned but we are not
  // presently concerned.
  const double distance = OrthogonalDistanceToPlane(point, plane, dim);
  for (int i = 0; i < dim; ++i) {
    point[i] -= distance*plane[i];
  }
}

/*
  "plane" is specified as [n_1, n_2, ..., n_{dim}, a_0], where the hyperplane has the equation: a_0 + \sum_i n_i * x_i = 0
  Hence n_vec = [n_1, ..., n_{dim}] is the (outward) normal vector.

  By convention, ||n_vec||_2 = 1.

  WARNING: This fails UNGRACEFULLY if vector \cdot normal = 0.0 and point \cdot normal + a_0 = 0
  i.e., the vector is parallel to the plane and the starting point lies on the plane.

  INPUTS:
  point[dim]: point to compute distance from
  plane[dim+1]: plane to compute distance to; data ordered as specified in header docs
  vector[dim]: vector to compute distance along
  dim: number of spatial dimensions
  RETURNS:
  signed distance along the given vector; + means the intersection is in the same direction as the vector
*/
inline OL_NONNULL_POINTERS OL_WARN_UNUSED_RESULT double DistanceToPlaneAlongVector(double const * restrict point, double const * restrict plane, double const * restrict vector, int dim) noexcept {
  // Let p_1 be the intersection of the ray (point, vector) and the plane.  Let p_0 be any point in the plane.
  // Then (p_1 - p_0) \cdot n = 0.  Also, p_0 \cdot n = -a_0 by the definition of the plane.
  // Let v be "vector" and x_0 be "point"; these two define our ray.
  // We want to find the distance d for which: p_1 = d*v + x_0
  // Substitute: (d*v + x_0 - p_0) \cdot n = 0 = d*(v \cdot n) + (x_0 - p_0) \cdot n
  // d = ((p_0 - x_0) \cdot n)/(v \cdot n)
  //   = (-a_0 - x_0 \cdot n)/(v \cdot n)
  const double numerator = -plane[dim] - DotProduct(point, plane, dim);  // (p_0 - l_0) \cdot n = -a_0 - x_0 \cdot n
  const double denominator = DotProduct(vector, plane, dim);  // v \cdot n
  if (unlikely(denominator == 0.0)) {
    if (numerator == 0.0) {
      return 0.0;  // vector and point lie in the plane
    } else {
      return std::numeric_limits<double>::infinity();  // way of "signalling" no intersection
    }
    // maybe this return should be ::max() instead to prevent bad-ness from infinity?
  } else {
    return numerator / denominator;
  }
}

}  // end namespace optimal_learning

#endif  // OPTIMAL_LEARNING_EPI_SRC_CPP_GPP_GEOMETRY_HPP_
