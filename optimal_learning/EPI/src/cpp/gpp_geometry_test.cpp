// gpp_geometry_test.cpp
/*
  This file contains functions for testing the functions and classes in gpp_geometry.hpp.
*/

#include "gpp_geometry_test.hpp"

#include <algorithm>
#include <limits>
#include <type_traits>
#include <vector>

#include <boost/random/uniform_int.hpp>  // NOLINT(build/include_order)

#include "gpp_common.hpp"
#include "gpp_geometry.hpp"
#include "gpp_linear_algebra.hpp"
#include "gpp_logging.hpp"
#include "gpp_random.hpp"
#include "gpp_test_utils.hpp"

namespace optimal_learning {

namespace {  // local functions for testing ClosedInterval

OL_WARN_UNUSED_RESULT int CheckClosedIntervalTraits() {
  int total_errors = 0;
  if (!std::is_pod<ClosedInterval>::value) {
    ++total_errors;
  }
  if (!std::is_trivial<ClosedInterval>::value) {
    ++total_errors;
  }
  if (!std::is_standard_layout<ClosedInterval>::value) {
    ++total_errors;
  }

  return total_errors;
}

OL_WARN_UNUSED_RESULT int ClosedIntervalIsInsideTest() {
  int total_errors = 0;
  // max == min
  {
    const double kMiddle = 9.378;
    const double kMin = kMiddle;
    const double kMax = kMiddle;
    ClosedInterval interval = {kMin, kMax};

    if (interval.IsInside(kMiddle) != true) {
      ++total_errors;
    }
    if (interval.IsInside(kMin) != true) {
      ++total_errors;
    }
    if (interval.IsInside(kMax) != true) {
      ++total_errors;
    }
    if (interval.IsInside(kMin - 0.5) != false) {
      ++total_errors;
    }
    if (interval.IsInside(kMax + 0.5) != false) {
      ++total_errors;
    }
  }

  // min < max
  {
    const double kMin = -2.71;
    const double kMax = 3.14;

    ClosedInterval interval = {kMin, kMax};

    if (interval.IsInside(0.0) != true) {
      ++total_errors;
    }
    if (interval.IsInside(kMin) != true) {
      ++total_errors;
    }
    if (interval.IsInside(kMax) != true) {
      ++total_errors;
    }
    if (interval.IsInside(kMin - 0.5) != false) {
      ++total_errors;
    }
    if (interval.IsInside(kMax + 0.5) != false) {
      ++total_errors;
    }
  }

  // max < min
  {
    const double kMin = -2.71;
    const double kMax = -3.14;

    ClosedInterval interval = {kMin, kMax};

    if (interval.IsInside(-3.0) != false) {
      ++total_errors;
    }
    if (interval.IsInside(kMin) != false) {
      ++total_errors;
    }
    if (interval.IsInside(kMax) != false) {
      ++total_errors;
    }
    if (interval.IsInside(kMin - 0.5) != false) {
      ++total_errors;
    }
    if (interval.IsInside(kMax + 0.5) != false) {
      ++total_errors;
    }
  }

  // infinity
  {
    const double kMin = 0.0;
    const double kMax = std::numeric_limits<double>::infinity();

    ClosedInterval interval = {kMin, kMax};

    if (interval.IsInside(1.0) != true) {
      ++total_errors;
    }
    if (interval.IsInside(kMin) != true) {
      ++total_errors;
    }
    if (interval.IsInside(kMax) != true) {
      ++total_errors;
    }
    if (interval.IsInside(kMin - 0.5) != false) {
      ++total_errors;
    }
    if (interval.IsInside(kMax + 0.5) != true) {
      ++total_errors;
    }
  }

  return total_errors;
}

OL_WARN_UNUSED_RESULT int ClosedIntervalLengthAndIsEmptyTest() {
  int total_errors = 0;
  // max == min
  {
    const double kMin = 9.378;
    const double kMax = 9.378;
    ClosedInterval interval = {kMin, kMax};

    if (interval.Length() != 0.0) {
      ++total_errors;
    }
    if (interval.IsEmpty() != false) {
      ++total_errors;
    }
  }

  // min < max
  {
    const double kMin = 2.71;
    const double kMax = 3.14;

    ClosedInterval interval = {kMin, kMax};

    if (interval.Length() != (kMax - kMin)) {
      ++total_errors;
    }
    if (interval.IsEmpty() != false) {
      ++total_errors;
    }
  }

  // max > min
  {
    const double kMin = -2.71;
    const double kMax = -3.14;

    ClosedInterval interval = {kMin, kMax};

    if (interval.Length() != (kMax - kMin)) {
      ++total_errors;
    }
    if (interval.IsEmpty() != true) {
      ++total_errors;
    }
  }

  // inifinity
  {
    const double kMin = -std::numeric_limits<double>::infinity();
    const double kMax = std::numeric_limits<double>::infinity();

    ClosedInterval interval = {kMin, kMax};

    if (interval.Length() != (kMax - kMin)) {
      ++total_errors;
    }
    if (interval.IsEmpty() != false) {
      ++total_errors;
    }
  }

  return total_errors;
}

}  // end unnamed namespace

int ClosedIntervalTests() {
  int total_errors = 0;
  int current_errors = 0;

  current_errors = CheckClosedIntervalTraits();
  if (current_errors != 0) {
    OL_PARTIAL_FAILURE_PRINTF("CheckClosedInterval failed with %d errors\n", current_errors);
  }
  total_errors += current_errors;

  current_errors = ClosedIntervalIsInsideTest();
  if (current_errors != 0) {
    OL_PARTIAL_FAILURE_PRINTF("ClosedIntervalIsInsideTest failed with %d errors\n", current_errors);
  }
  total_errors += current_errors;

  current_errors = ClosedIntervalLengthAndIsEmptyTest();
  if (current_errors != 0) {
    OL_PARTIAL_FAILURE_PRINTF("ClosedIntervalLengthAndIsEmptyTest failed with %d errors\n", current_errors);
  }
  total_errors += current_errors;

  return total_errors;
}

namespace {  // local functions for testing gpp_geometry.hpp's functions

OL_WARN_UNUSED_RESULT int CheckPointInHypercubeTest() {
  const int dim = 4;

  // Domain bounds (min, max pairs)
  const std::vector<ClosedInterval> hypercube = {
    {0.0, 1.0},
    {0.0, 1.0},
    {0.0, 1.0},
    {0.0, 2.0}};

  const std::vector<double> points = {
    0.0, 0.0, 0.0, 0.0,    // origin
    0.5, 0.4, 0.3, 0.8,    // hypercube interior
    0.2, 1.2, 0.1, 0.7,    // outside the y-face
    1.0, 0.5, 0.4, 0.2,    // on the x-face
    1.3, -1.0, 2.0, 2.5};  // exterior

  // Truth for whether each point of kPointsCheckPointInside is inside/outside the domain
  const std::vector<bool> truth = {
    true,
    true,
    false,
    true,
    false};

  const int num_points = points.size() / dim;

  int total_errors = 0;
  for (int i = 0; i < num_points; ++i) {
    if (CheckPointInHypercube(hypercube.data(), points.data() + i*dim, dim) != truth[i]) {
      ++total_errors;
    }
  }
  return total_errors;
}

OL_WARN_UNUSED_RESULT int CheckPointInUnitSimplexTest() {
  const int dim = 4;

  const std::vector<double> points = {
    0.0, 0.0, 0.0, 0.0,       // origin
    0.5, 0.4, 0.3, 0.8,       // point in unit hypercube but not simplex
    0.05, 0.45, 0.25, 0.1,    // simplex interior
    0.2, 1.2, 0.1, 0.7,       // point outside y-face
    0.1, 0.4, 0.0, 0.2,       // point on the z-face
    1.3, -5.0, 2.0, 2.5,      // simplex exterior, but 0 <= sum <= 1
    0.25, 0.25, 0.25, 0.25};  // point on the "diagonal" face

  // Truth for whether each point of kPointsCheckPointInside is inside/outside the domain
  const std::vector<bool> truth = {
    true,
    false,
    true,
    false,
    true,
    false,
    true};

  const int num_points = points.size() / dim;

  int total_errors = 0;
  for (int i = 0; i < num_points; ++i) {
    if (CheckPointInUnitSimplex(points.data() + i*dim, dim) != truth[i]) {
      ++total_errors;
    }
  }
  return total_errors;
}

OL_WARN_UNUSED_RESULT int OrthogonalDistanceToPlaneTest() {
  // idea: pick several random hyperplanes (unit normal + right hand side, \sum_i n_i*x_i = -a_0)
  // pick a point on the hyperplane (easy: x_i = -a_0 * n_i)
  // pick random distances and the point-on-plane by distance*normal; check OrthogonalDistanceToPlane == distance
  const int dim = 5;
  const int num_planes = 10;
  const int num_points = 10;

  UniformRandomGenerator uniform_generator(314);
  boost::uniform_real<double> uniform_double_distribution(-2.5, 2.5);

  std::vector<double> hyperplane(dim + 1);
  std::vector<double> point_on_plane(dim);
  std::vector<double> shifted_point(dim);
  double distance_truth, distance_computed, norm;

  int total_errors = 0;
  for (int i = 0; i < num_planes; ++i) {
    // generate random hyperplane
    for (int k = 0; k < dim+1; ++k) {
      hyperplane[k] = uniform_double_distribution(uniform_generator.engine);
    }
    // normalize the normal vector
    norm = VectorNorm(hyperplane.data(), dim);
    norm = 1.0/norm;
    VectorScale(dim, norm, hyperplane.data());

    // generate random point
    for (int k = 0; k < dim; ++k) {
      point_on_plane[k] = hyperplane[k] * (-hyperplane[dim]);
    }

    // could also pick random point by generating dim-1 random coordinates, then computing the last one
    // from the eqn of the plane.  not sure we need this generality though

    // pick random distances
    for (int j = 0; j < num_points; ++j) {
      distance_truth = uniform_double_distribution(uniform_generator.engine);
      for (int k = 0; k < dim; ++k) {
        shifted_point[k] = point_on_plane[k] + distance_truth * hyperplane[k];
      }

      distance_computed = OrthogonalDistanceToPlane(shifted_point.data(), hyperplane.data(), dim);

      if (!CheckDoubleWithinRelative(distance_computed, distance_truth, 2.0e-14)) {
        ++total_errors;
      }
    }
  }
  return total_errors;
}

OL_WARN_UNUSED_RESULT int OrthogonalProjectionOntoPlaneTest() {
  // idea: pick several random hyperplanes (unit normal + right hand side, \sum_i n_i*x_i = -a_0)
  // pick random points
  // project each point onto plane & verify that the resulting point's OrthogonalDistanceToPlane == 0
  const int dim = 5;
  const int num_planes = 10;
  const int num_points = 10;

  UniformRandomGenerator uniform_generator(314);
  boost::uniform_real<double> uniform_double_distribution(-2.5, 2.5);

  std::vector<double> hyperplane(dim + 1);
  std::vector<double> point_on_plane(dim);
  std::vector<double> random_point(dim);
  std::vector<double> projected_point(dim);
  double distance_truth = 0.0;  // we're always projecting onto the plane
  double distance_computed, norm;

  int total_errors = 0;
  for (int i = 0; i < num_planes; ++i) {
    // generate random hyperplane
    for (int k = 0; k < dim+1; ++k) {
      hyperplane[k] = uniform_double_distribution(uniform_generator.engine);
    }
    // normalize the normal vector
    norm = VectorNorm(hyperplane.data(), dim);
    norm = 1.0/norm;
    VectorScale(dim, norm, hyperplane.data());

    // pick random distances
    for (int j = 0; j < num_points; ++j) {
      for (int k = 0; k < dim; ++k) {
        random_point[k] = uniform_double_distribution(uniform_generator.engine);
        projected_point[k] = random_point[k];
      }

      OrthogonalProjectionOntoPlane(hyperplane.data(), dim, projected_point.data());
      distance_computed = OrthogonalDistanceToPlane(projected_point.data(), hyperplane.data(), dim);

      if (!CheckDoubleWithinRelative(distance_computed, distance_truth, 8*std::numeric_limits<double>::epsilon())) {
        ++total_errors;
      }
    }
  }
  return total_errors;
}

/*
  Outline:
  1) Pick several random hyperplanes (unit normal + right hand side, \sum_i n_i*x_i = -a_0)
  2) Pick random points & random vectors
  3) Project each point onto plane (along vector) & verify that the resulting point's OrthogonalDistanceToPlane == 0
*/
OL_WARN_UNUSED_RESULT int DistanceToPlaneAlongVectorTest() {
  const int dim = 5;
  const int num_planes = 10;
  const int num_points = 10;

  UniformRandomGenerator uniform_generator(314);
  boost::uniform_real<double> uniform_double_distribution(-2.5, 2.5);

  std::vector<double> hyperplane(dim + 1);
  std::vector<double> point_on_plane(dim);
  std::vector<double> random_point(dim);
  std::vector<double> projected_point(dim);
  std::vector<double> random_vector(dim);
  double distance_truth = 0.0;  // we're always projecting onto the plane
  double distance, distance_computed, norm;

  int total_errors = 0;
  for (int i = 0; i < num_planes; ++i) {
    // generate random hyperplane
    for (int k = 0; k < dim+1; ++k) {
      hyperplane[k] = uniform_double_distribution(uniform_generator.engine);
    }
    // normalize the normal vector
    norm = VectorNorm(hyperplane.data(), dim);
    norm = 1.0/norm;
    VectorScale(dim, norm, hyperplane.data());

    // pick random distances
    for (int j = 0; j < num_points; ++j) {
      for (int k = 0; k < dim; ++k) {
        random_point[k] = uniform_double_distribution(uniform_generator.engine);
        random_vector[k] = uniform_double_distribution(uniform_generator.engine);
      }

      distance = DistanceToPlaneAlongVector(random_point.data(), hyperplane.data(), random_vector.data(), dim);
      for (int k = 0; k < dim; ++k) {
        projected_point[k] = random_point[k] + distance * random_vector[k];
      }

      distance_computed = OrthogonalDistanceToPlane(projected_point.data(), hyperplane.data(), dim);

      if (!CheckDoubleWithinRelative(distance_computed, distance_truth, 4.0e-14)) {
        ++total_errors;
      }
    }
  }
  return total_errors;
}

}  // end unnamed namespace

int GeometryToolsTests() {
  int total_errors = 0;
  int current_errors = 0;

  current_errors = CheckPointInHypercubeTest();
  if (current_errors != 0) {
    OL_PARTIAL_FAILURE_PRINTF("CheckPointInHypercube failed with %d errors\n", current_errors);
  }
  total_errors += current_errors;

  current_errors = CheckPointInUnitSimplexTest();
  if (current_errors != 0) {
    OL_PARTIAL_FAILURE_PRINTF("CheckPointInUnitSimplex failed with %d errors\n", current_errors);
  }
  total_errors += current_errors;

  current_errors = OrthogonalDistanceToPlaneTest();
  if (current_errors != 0) {
    OL_PARTIAL_FAILURE_PRINTF("OrthogonalDistanceToPlane failed with %d errors\n", current_errors);
  }
  total_errors += current_errors;

  current_errors = OrthogonalProjectionOntoPlaneTest();
  if (current_errors != 0) {
    OL_PARTIAL_FAILURE_PRINTF("OrthogonalProjectionOntoPlane failed with %d errors\n", current_errors);
  }
  total_errors += current_errors;

  current_errors = DistanceToPlaneAlongVectorTest();
  if (current_errors != 0) {
    OL_PARTIAL_FAILURE_PRINTF("DistanceToPlaneAlongVector failed with %d errors\n", current_errors);
  }
  total_errors += current_errors;

  return total_errors;
}

}  // end namespace optimal_learning
