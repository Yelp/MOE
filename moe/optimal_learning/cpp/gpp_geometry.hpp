/*!
  \file gpp_geometry.hpp
  \rst
  This file contains classes and utilities to help with (planar) geometry operations.

    * Functions:

        * CheckPointInHypercube: checks whether a point is in the domain \ms [x_{1,min}, x_{1,max}] X ... X [x_{dim,min}, x_{dim,max}]\me
        * CheckPointInUnitSimplex: checks whether a point is in the unit simplex

    * Structs:

        * ClosedInterval: represents the closed interval \ms [a, b]\me with utilities for length, in/out test etc.
        * Plane: represents a dim-dimensional plane as its "outward" unit normal and the signed distance from the origin.
          Contains functions for distance, projection, etc.
\endrst*/

#ifndef MOE_OPTIMAL_LEARNING_CPP_GPP_GEOMETRY_HPP_
#define MOE_OPTIMAL_LEARNING_CPP_GPP_GEOMETRY_HPP_

#include <limits>
#include <vector>

#include "gpp_common.hpp"
#include "gpp_linear_algebra.hpp"

namespace optimal_learning {

/*!\rst
  Container to represent the mathematical notion of a closed interval, commonly written \ms [a,b]\me.
  The closed interval \ms [a,b]\me is the set of all numbers \ms x \in \mathbb{R}\me such that \ms a \leq x \leq b\me.
  Note that "closed" here indicates the interval *includes* both endpoints.
  An interval with \ms a > b\me is considered empty.

  This struct is "trivial" and "standard layout" and thus "POD" (in the C++11 sense).

  * http://en.cppreference.com/w/cpp/types/is_pod
  * http://stackoverflow.com/questions/4178175/what-are-aggregates-and-pods-and-how-why-are-they-special/7189821#7189821

  This struct is not an aggregate; list (aka brace) initialization and a 2-argument constructor are both available::

    ClosedInterval tmp(1.0, 2.0);  // this ctor makes it non-aggregate
    ClosedInterval tmp{1.0, 2.0};  // and brace-style (aka initializer list) inits also work
\endrst*/
struct ClosedInterval {
  /*!\rst
    Explicitly defaulted default constructor.
    Defining a custom ctor (below) disables the default ctor, so we explicitly default it.
    This is needed to maintain POD-ness.
  \endrst*/
  ClosedInterval() = default;

  /*!\rst
    Constructs a ClosedInterval object with specified ``min``, ``max``.

    The presence of this ctor makes this object a non-aggregate, so brace-initialization
    follow list initialization rules (not aggregate initialization):

    * http://en.cppreference.com/w/cpp/language/list_initialization

    \param
      :min_in: left bound of the interval
      :max_in: right bound of the interval
  \endrst*/
  ClosedInterval(double min_in, double max_in) : min(min_in), max(max_in) {  // NOLINT(build/include_what_you_use) misinterpreting these as calls to std::min, max
  }

  /*!\rst
    Check if a value is inside this ClosedInterval.

    \param
      :value: the value to check

    \return
      true if ``min`` \ms\leq\me ``value`` \ms\leq\me ``max``
  \endrst*/
  bool IsInside(double value) const OL_PURE_FUNCTION OL_WARN_UNUSED_RESULT {
    return (value >= min) & (value <= max);
  }

  /*!\rst
    Compute the length of this ClosedInterval; result can be negative (i.e., an empty interval).

    \return
      length of the interval
  \endrst*/
  double Length() const OL_PURE_FUNCTION OL_WARN_UNUSED_RESULT {
    return max - min;
  }

  /*!\rst
    Checks whether the interval is \ms\emptyset\me (i.e., ``max`` < ``min``).

    Equivalent to Length() \ms\geq\me 0.0.

    \return
      true if the interval is non-empty: ``max`` \ms\geq\me ``min``
  \endrst*/
  bool IsEmpty() const OL_PURE_FUNCTION OL_WARN_UNUSED_RESULT {
    return max < min;
  }

  //! left and right boundaries, ``[min, max]``, of this ClosedInterval
  double min, max;
};

/*!\rst
  We specify a plane in dim-space using the Hesse (or Hessian) Normal Form:

  * http://mathworld.wolfram.com/HessianNormalForm.html

  The equation of plane requires dim + 1 real numbers:

  \ms a_0 + \sum_{i=0}^{dim} n_i x_i = 0\me

  Hence we can describe any plane as a vector, \ms [n_0, n_2, ..., n_{dim-1}]\me, and a real number, \ms a_0\me.

  Let \ms n_{vec} = [n_0, ..., n_{dim-1}]\me be the (outward) *unit* normal vector. By convention, \ms \|n_{vec}\|_2 = 1\me.
  \ms a_0\me is the *signed* distance to the origin. This is the distance from the plane to the origin in the direction of
  \ms n_{vec}\me. Put another way, \ms a_0\me is positive if the origin is in the same half-space "pointed to" by
  \ms n_{vec}\me and negative otherwise.

  Note: \ms a_0\me is measured in units of \ms \|n_{vec}\|\me, so if it is *not* an unit vector, that is analogous to scaling \ms a_0\nme.

  As an example, let's consider 4 planes with dim = 2:

  * \ms a_0\me = -1, and \ms n_{vec}\me = { 1.0, 0.0}: the plane x =  1 with rightward pointing normal.
  * \ms a_0\me = -1, and \ms n_{vec}\me = {-1.0, 0.0}: the plane x = -1 with leftward  pointing normal.
  * \ms a_0\me =  1, and \ms n_{vec}\me = { 1.0, 0.0}: the plane x = -1 with rightward pointing normal.
  * \ms a_0\me =  1, and \ms n_{vec}\me = {-1.0, 0.0}: the plane x =  1 with leftward  pointing normal.

  Be careful with the signs.

  Another common way of specifying a plane is via a point \ms x_0\me and an unit normal, \ms n_{vec}\me. A point x is in the plane
  if and only if \ms (x-x_0) \cdot n_{vec} = 0\me. Since \ms x_0\me is constant, we can precompute and store
  \ms ms x_0 \cdot n_{vec} = -a_0\me, yielding: \ms x \cdot n_{vec} - x_0 \cdot n_{vec} = x \cdot n_{vec} + a_0\me, which is our
  original equation of a plane.
\endrst*/
struct Plane {
 public:
  Plane() = delete;  // no default ctor; dim = 0 doesn't really make sense as a default

  /*!\rst
    Creates a zero-initialized plane object with enough space for dim-dimensions.

    .. NOTE::
         This plane is invalid (``unit_normal`` := zero is not a unit vector) and needs to have its members initialized.
         That said, no member functions will fail even without complete initialization.

    \param
      :dim: the number of spatial dimensions
  \endrst*/
  explicit Plane(int dim_in) : offset(0.0), unit_normal(dim_in) {
  }

  /*!\rst
    Creates a plane in dim-dimensions with the specified unit normal (\ms n_i\me) and offset (\ms a_0\me):

    \ms a_0 + \sum_{i=0}^{dim} n_i * x_i = 0\me

    .. NOTE::
         Failure to specify a unit normal will result in surprising behavior. \ms a_0\me is really in units of
         \ms\|n_{vec}\|\me, so if \ms\|n_{vec}\| = 3.5\me, then the actual distance to the origin is \ms 3.5 a_0\me.

    \param
      :dim: the number of spatial dimensions
      :unit_normal[dim]: the unit normal vector. VectorNorm(unit_normal, dim) must be 1.0
      :offset: a_0, the signed distance to the origin (see class docs)
  \endrst*/
  Plane(double const * restrict unit_normal_in, double offset_in, int dim_in) OL_NONNULL_POINTERS : offset(offset_in), unit_normal(unit_normal_in, unit_normal_in + dim_in) {
  }

  /*!\rst
    Creates a plane in dim-dimensions that contains "point," with the specified unit normal.

    \param
      :dim: the number of spatial dimensions
      :unit_normal[dim]: the unit normal vector. VectorNorm(unit_normal, dim) must be 1.0
      :point[dim]: a point contained in the plane
  \endrst*/
  Plane(double const * restrict unit_normal_in, double const * restrict point, int dim_in) OL_NONNULL_POINTERS : unit_normal(unit_normal_in, unit_normal_in + dim_in) {
    // As noted in the class docs, a point x is in the plane if and only if (x-x_0) \cdot n_vec = 0.
    // Since x_0 is constant (this is "point"), we can precompute and store x_0 \cdot n_vec = -a_0,
    offset = -DotProduct(point, unit_normal.data(), dim_in);
  }

  /*!\rst
    \return
      the number of spatial dimensions (that this plane lives in)
  \endrst*/
  int dim() const OL_PURE_FUNCTION OL_WARN_UNUSED_RESULT {
    return unit_normal.size();
  }

  /*!\rst
    Computes the signed, shortest distance from this plane to point: positive means the point is in the half-space
    determined by the direction of ``unit_normal``.

    .. Note:: if point is the origin, this yields precisely \ms a_0\me (``offset``).

    \param
      :point[dim]: point to compute distance to
    \return
      signed, shortest distance from this plane to point, where positive means the point and normal are in the same half-space
  \endrst*/
  double OrthogonalDistanceToPoint(double const * restrict point) const OL_NONNULL_POINTERS OL_WARN_UNUSED_RESULT {
    // formula: let p_1 = "point", p_0 be any point in the plane, and n be the normal vector
    // distance = (p_1 - p_0) \cdot n_vec = p_1 \cdot n_vec + a_0 (\sum_i n_i * p_0_i = -a_0; see end of class docs)
    const double distance = DotProduct(point, unit_normal.data(), dim());
    return distance + offset;  // we assume ||n||_2 = 1.
  }

  /*!\rst
    Projects a point (orthogonally) onto a plane; i.e., finds the point on the plane that is closest to the input point.

    \param
      :point[dim]: point to project onto plane
    \output
      :point[dim]: point projected onto plane
  \endrst*/
  void OrthogonalProjectionOntoPlane(double * restrict point) const OL_NONNULL_POINTERS {
    // formula: let d be the orthogonal, signed distance from point to plane (where + means the point lies in the half-space
    //          pointed to by the normal vector)
    // then we compute: projected_point = point - d*unit_normal, where unit_normal is the unit normal vector of the plane

    // It is also possible to parameterize the operation and do constrained optimization (in the space of the plane) and find
    // the point on the plane that is nearest to the specified point.  This is generally better-conditioned but we are not
    // presently concerned.
    const double distance = OrthogonalDistanceToPoint(point);
    for (int i = 0; i < dim(); ++i) {
      point[i] -= distance*unit_normal[i];
    }
  }

  /*!\rst
    Computes the signed distance from the specified ``point`` to this plane along the specified ``vector``. This result
    is computed in units of \ms\|vector\|_2\me. That is, a distance of 3.14 means if we compute:

    ``new_point = 3.14*vector + point``,

    then new_point will be on this plane.

    A negative distance means the plane is "behind" the ray.

    \param
      :point[dim]: point to compute distance from
      :vector[dim]: vector to compute distance along
    \return
      Signed distance along the given vector; positive means the intersection is in the same direction as the vector.
      This result is in units of \ms\|vector\|_2\me; normalize vector if you want an actual distance.
  \endrst*/
  double DistanceToPlaneAlongVector(double const * restrict point, double const * restrict vector) const OL_NONNULL_POINTERS OL_WARN_UNUSED_RESULT {
    // Let p_1 be the intersection of the ray (point, vector) and the plane.  Let p_0 be any point in the plane.
    // Then (p_1 - p_0) \cdot n = 0.  Also, p_0 \cdot n = -a_0 by the definition of the plane.
    // Let v be "vector" and x_0 be "point"; these two define our ray.
    // We want to find the distance d for which: p_1 = d*v + x_0
    // Substitute: (d*v + x_0 - p_0) \cdot n = 0 = d*(v \cdot n) + (x_0 - p_0) \cdot n
    // d = ((p_0 - x_0) \cdot n)/(v \cdot n)
    //   = (-a_0 - x_0 \cdot n)/(v \cdot n)
    const double numerator = -offset - DotProduct(point, unit_normal.data(), dim());  // (p_0 - l_0) \cdot n = -a_0 - x_0 \cdot n
    const double denominator = DotProduct(vector, unit_normal.data(), dim());  // v \cdot n
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

  //! the scalar a_0: the distance from the plane to the origin
  double offset;
  //! the vector n_i: the "outward" unit normal vector
  std::vector<double> unit_normal;
};

/*!\rst
  Simple auxilliary function that checks if a point is within the given hypercube.

  \param
    :domain[dim]: array of ClosedInterval specifying the boundaries of a dim-dimensional tensor-product domain.
    :point[dim]: the point to check
    :dim: the number of spatial dimensions
  \return
    true if the point is inside the specified tensor-product domain
\endrst*/
inline OL_WARN_UNUSED_RESULT OL_NONNULL_POINTERS bool CheckPointInHypercube(ClosedInterval const * restrict domain, double const * restrict point, int dim) {
  for (int i = 0; i < dim; ++i) {
    if (domain[i].IsInside(point[i]) == false) {
      return false;
    }
  }
  return true;
}

/*!\rst
  Checks if a point is inside/on the unit d-simplex.  A point \ms x_i\me lies inside the unit d-simplex if:

    1. \ms x_i \geq 0 \ \forall i\me  (i ranging over dimension)
    2. \ms \sum_i x_i \leq 1\me

  Implying that \ms x_i \leq 1 \ \forall i\me.

  \param
    :point[dim]: point to check
    :dim: number of dimensions
  \return
    true if the point lies inside/on the unit d-simplex
\endrst*/
inline OL_NONNULL_POINTERS OL_WARN_UNUSED_RESULT bool CheckPointInUnitSimplex(double const * restrict point, int dim) {
  // Being this far outside the simplex still counts as inside
  static const double kWallTolerance = 4.0*std::numeric_limits<double>::epsilon();
  double sum = 0.0;
  for (int i = 0; i < dim; ++i) {
    if (point[i] < 0.0) {
      return false;
    }
    sum += point[i];
  }
  // sum can be slightly beyond 1.0 to account for floating point issues
  return (sum - kWallTolerance) <= 1.0;
}

}  // end namespace optimal_learning

#endif  // MOE_OPTIMAL_LEARNING_CPP_GPP_GEOMETRY_HPP_
