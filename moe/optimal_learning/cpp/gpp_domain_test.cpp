/*!
  \file gpp_domain_test.cpp
  \rst
  This file contains functions for testing the functions and classes in gpp_domain.hpp.

  It has several templated test functions (on DomainTestFixture) for:

  * CheckPointInDomain: specify a domain, point list, and truth values.  checks that points are/are not inside
  * GeneratePointInDomain: generates some random points, checks that they are inside the domain
  * GenerateUniformPointsInDomain: generates uniformly distributed points, checks that they are inside the domain
    TODO(GH-128): Test whether computed point distribution is actually uniform
  * LimitUpdate: starting from points within the domain, checks that:

    * updates to another point in the domain remain unchanged
    * updates to a point outside the domain are limited such that the new endpoint is in the domain

  These are all wrapper'd in RunDomainTests<>(), which is templated on DomainTestFixture.
  A similar batch of tests exists for RepeatedDomain and is accessed through RunRepeatedDomainTests<>(),
  which is also templated on DomainTestFixture.

  To use these tests, define a DomainTestFixture struct with the following fields (and values for them)::

    struct SomeDomainTestFixture final {
      using DomainType = SomeDomain;  // the domain class we want to test

      // use with CheckPointInsideTest
      int kDimCheckPointInside;  // number of dimensions for CheckPointInside test
      vector<double> kDomainBoundsCheckPointInside;  // arg1 to SomeDomain(arg1, dim) constructor
      vector<double> kPointsCheckPointsInside;  // points at which to check inside-ness
      int kNumPointsCheckPointInside;  // number of points to check
      vector<bool> kTruthCheckPointsInside;  // truth for whether each point is inside/outside

      // use with GeneratePointInDomainTest, GenerateUniformPointsInDomainTest, LimitUpdateTest
      int kDimPointGeneration;  // number of dimensions for random point generation
      // ranges for lower, upper bounds of tensor product domain components for random point generation (if needed)
      boost::uniform_real<double> kUniformDoubleDomainLowerBound;
      boost::uniform_real<double> kUniformDoubleDomainUpperBound;
    };

  Each test function accepts a "const DomainTestFixture&" input from which we access the aforementioned parameters.

  See below for examples.
\endrst*/

#include "gpp_domain_test.hpp"

#include <algorithm>
#include <limits>
#include <vector>

#include <boost/random/uniform_int.hpp>  // NOLINT(build/include_order)

#include "gpp_common.hpp"
#include "gpp_domain.hpp"
#include "gpp_geometry.hpp"
#include "gpp_linear_algebra.hpp"
#include "gpp_logging.hpp"
#include "gpp_random.hpp"
#include "gpp_test_utils.hpp"

namespace optimal_learning {

namespace {

/*!\rst
  Domain that behaves like the parent DomainType except that this overrides GenerateUniformPointsInDomain()
  with mock generator. The mock generator is not random nor uniform but produces a simple "distribution"
  that makes testing RepeatedDomain easier.

  .. WARNING:: this class is NOT thread-safe. It maintains ``mutable`` state (to gain persistence through
      const-marked functions).
\endrst*/
template <typename DomainType>
struct DomainWithMockUniformRandomPoints : public DomainType {
  //! On the ``kStep``-th call to GenerateUniformPointsInDomain, that function generates num_points / 2 points
  //! instead of num_points.
  static constexpr int kStep = 2;

  /*!\rst
    Constructs a TensorProductDomainMockUniformRandomPoints.

    It would be more general to use a variadic template and foward all arguments to the underlying DomainType.

    \param
      :domain[dim]: array of ClosedInterval specifying the boundaries of a dim-dimensional domain.
      :dim_in: number of spatial dimensions
  \endrst*/
  DomainWithMockUniformRandomPoints(ClosedInterval const * restrict domain, int dim_in) : DomainType(domain, dim_in), num_calls(0), counter(0.0) {
  }

  /*!\rst
    Reset internal state to initial conditions.
  \endrst*/
  void reset() {
    num_calls = 0;
    counter = 0.0;
  }

  /*!\rst
    Generates points following a simple increasing pattern. The ``kStep``-th call generates num_points / 2 points; other calls
    generate num_points points.

    These points may NOT be in the domain.

    \param
      :num_points: number of random points to generate
      :uniform_generator[1]: a UniformRandomGenerator object providing the random engine for uniform random numbers
      :random_points[dim_][num_points]: properly sized array
    \output
      :uniform_generator[1]: UniformRandomGenerator object will have its state changed due to random draws
      :random_points[dim_][num_points]: point with coordinates inside the domain
    \return
      number of points actually generated
  \endrst*/
  int GenerateUniformPointsInDomain(int num_points, UniformRandomGenerator * OL_UNUSED(uniform_generator), double * restrict random_points) const {
    if (num_calls == kStep) {
      num_points /= 2;
    }
    int _dim = this->dim();

    for (int i = 0; i < num_points; ++i) {
      for (int j = 0; j < _dim; ++j) {
        random_points[i*_dim + j] = counter;
        counter += 1.0;
      }
    }

    ++num_calls;
    return num_points;
  }

  // Hack: use ``mutable`` to maintain state that is modifiable through const-marked functions. These functions remain
  // const so that their signatures match the superclass, but we need the state to construct an interesting mock.
  //! number of times GenerateUniformPointsInDomain has been called
  mutable int num_calls;
  //! number of coordinates generated
  mutable double counter;
};

/*!\rst
  DomainTestFixture for TensorProductDomain.  See file docs for details on the layout.
\endrst*/
struct TensorProductDomainTestFixture final {
  using DomainType = TensorProductDomain;

  //! Dimension to use with CheckPointInsideTest
  const int kDimCheckPointInside = 4;

  //! Domain bounds (min, max pairs)
  const std::vector<ClosedInterval> kDomainBoundsCheckPointInside = {
    {0.0, 1.0},
    {0.0, 1.0},
    {0.0, 1.0},
    {0.0, 2.0}};

  //! points to test PointsCheckPointInside with; truth values in kTruthCheckPointInside
  const std::vector<double> kPointsCheckPointInside = {
    0.0, 0.0, 0.0, 0.0,    // origin
    0.5, 0.4, 0.3, 0.8,    // hypercube interior
    0.2, 1.2, 0.1, 0.7,    // outside the y-face
    1.0, 0.5, 0.4, 0.2,    // on the x-face
    1.3, -1.0, 2.0, 2.5};  // exterior

  //! Truth for whether each point of kPointsCheckPointInside is inside/outside the domain
  const std::vector<bool> kTruthCheckPointInside = {
    true,
    true,
    false,
    true,
    false};

  //! number of points to check
  const int kNumPointsCheckPointInside = kPointsCheckPointInside.size() / kDimCheckPointInside;

  // use with GeneratePointInDomainTest, GenerateUniformPointsInDomainTest, LimitUpdateTest
  //! dimension of points to use in GeneratePointInDomainTest
  const int kDimPointGeneration = 5;
  //! range to draw interval ower bounds from
  const boost::uniform_real<double> kUniformDoubleDomainLowerBound = decltype(kUniformDoubleDomainLowerBound)(-5.0, -0.01);
  //! range to draw interval upper bounds from
  const boost::uniform_real<double> kUniformDoubleDomainUpperBound = decltype(kUniformDoubleDomainUpperBound)(0.02, 4.0);
};

/*
  DomainTestFixture for SimplexIntersectTensorProductDomain.  See file docs for details on the layout.
*/
struct SimplexIntersectTensorProductDomainTestFixture final {
  using DomainType = SimplexIntersectTensorProductDomain;

  //! Dimension to use with CheckPointInsideTest
  const int kDimCheckPointInside = 4;

  //! Domain bounds (min, max pairs)
  const std::vector<ClosedInterval> kDomainBoundsCheckPointInside = {
    {0.0, 1.0},   // same bounds as simplex bounding box
    {-1.0, 1.0},  // exceeds the simplex b-box left boundary (-1.0 < 0.0)
    {0.0, 0.9},   // inside the simplex b-box
    {0.0, 2.0}};  // exceeds the simplex b-box right boundary (2.0 > 1.0)

  //! points to test PointsCheckPointInside with; truth values in kTruthCheckPointInside
  const std::vector<double> kPointsCheckPointInside = {
    0.0, 0.0, 0.0, 0.0,      // origin
    0.5, 0.4, 0.3, 0.8,      // point in unit hypercube but not simplex
    0.05, 0.45, 0.25, 0.1,   // simplex interior
    0.2, 1.2, 0.1, 0.7,      // point outside y-face
    0.1, 0.4, 0.0, 0.2,      // point on the z-face
    1.3, -5.0, 2.0, 2.5,     // simplex exterior, but 0 <= sum <= 1
    0.25, 0.25, 0.25, 0.25,  // point on the "diagonal" face
    0.11, 0.11, 0.8, 0.1,    // point in the tensor prod but not the simplex
    0.01, 0.01, 1.0, 0.01};  // point in the simplex but not the tensor prod

  //! Truth for whether each point of kPointsCheckPointInside is inside/outside the domain
  const std::vector<bool> kTruthCheckPointInside = {
    true,
    false,
    true,
    false,
    true,
    false,
    true,
    false,
    false};

  //! number of points to check
  const int kNumPointsCheckPointInside = kPointsCheckPointInside.size() / kDimCheckPointInside;

  // use with GeneratePointInDomainTest, GenerateUniformPointsInDomainTest, LimitUpdateTest
  //! dimension of points to use in GeneratePointInDomainTest
  const int kDimPointGeneration = 5;
  //! range to draw interval ower bounds from
  const boost::uniform_real<double> kUniformDoubleDomainLowerBound = decltype(kUniformDoubleDomainLowerBound)(0.05, 0.15);
  //! range to draw interval upper bounds from
  const boost::uniform_real<double> kUniformDoubleDomainUpperBound = decltype(kUniformDoubleDomainUpperBound)(0.2, 0.6);
};

/*!\rst
  Check whether kPointsCheckPointInside points are inside the test domain according to kTruthCheckPointInside
  truth values in the input test fixture.

  \return
    number of test failures
\endrst*/
template <typename DomainTestFixture>
OL_WARN_UNUSED_RESULT int CheckPointInsideTest(const DomainTestFixture& domain_test_case) {
  // references for convenience
  const std::vector<ClosedInterval>& domain_bounds = domain_test_case.kDomainBoundsCheckPointInside;
  const std::vector<double>& points = domain_test_case.kPointsCheckPointInside;
  const std::vector<bool>& truth = domain_test_case.kTruthCheckPointInside;
  const int dim = domain_test_case.kDimCheckPointInside;
  const int num_points = domain_test_case.kNumPointsCheckPointInside;

  typename DomainTestFixture::DomainType domain(domain_bounds.data(), dim);

  int total_errors = 0;
  for (int i = 0; i < num_points; ++i) {
    if (domain.CheckPointInside(points.data() + i*dim) != truth[i]) {
      ++total_errors;
    }
  }
  return total_errors;
}

/*!\rst
  Check whether GeneratePointInDomain generates points that are inside the input domain.

  \return
    number of test failures
\endrst*/
template <typename DomainTestFixture>
OL_WARN_UNUSED_RESULT int GeneratePointInDomainTest(const DomainTestFixture& domain_test_case) {
  const int kDim = domain_test_case.kDimPointGeneration;
  const int num_tests = 50;
  std::vector<ClosedInterval> domain_bounds(kDim);
  std::vector<double> random_point(kDim);
  int total_errors = 0;

  UniformRandomGenerator uniform_generator(314);
  const boost::uniform_real<double>& uniform_double_domain_lower_bound = domain_test_case.kUniformDoubleDomainLowerBound;
  const boost::uniform_real<double>& uniform_double_domain_upper_bound = domain_test_case.kUniformDoubleDomainUpperBound;

  // domain_bounds w/min edge length 0.03 and max edge length 9
  for (int i = 0; i < kDim; ++i) {
    domain_bounds[i].min = uniform_double_domain_lower_bound(uniform_generator.engine);
    domain_bounds[i].max = uniform_double_domain_upper_bound(uniform_generator.engine);
  }

  typename DomainTestFixture::DomainType domain(domain_bounds.data(), kDim);

  for (int i = 0; i < num_tests; ++i) {
    if (!domain.GeneratePointInDomain(&uniform_generator, random_point.data())) {
      ++total_errors;
    }

    if (!domain.CheckPointInside(random_point.data())) {
      ++total_errors;
    }
  }

  return total_errors;
}

/*!\rst
  Check whether GenerateUniformPointsInDomain generates points that are inside the input domain.

  TODO(GH-128): Test whether computed point distribution is actually uniform

  \return
    number of test failures
\endrst*/
template <typename DomainTestFixture>
OL_WARN_UNUSED_RESULT int GenerateUniformPointsInDomainTest(const DomainTestFixture& domain_test_case) {
  const int kDim = domain_test_case.kDimPointGeneration;
  int num_tests = 500;
  std::vector<ClosedInterval> domain_bounds(kDim);
  std::vector<double> random_points(kDim*num_tests);
  int total_errors = 0;

  UniformRandomGenerator uniform_generator(3143);
  const boost::uniform_real<double>& uniform_double_domain_lower_bound = domain_test_case.kUniformDoubleDomainLowerBound;
  const boost::uniform_real<double>& uniform_double_domain_upper_bound = domain_test_case.kUniformDoubleDomainUpperBound;

  // domain_bounds w/min edge length 0.03 and max edge length 9
  for (int i = 0; i < kDim; ++i) {
    domain_bounds[i].min = uniform_double_domain_lower_bound(uniform_generator.engine);
    domain_bounds[i].max = uniform_double_domain_upper_bound(uniform_generator.engine);
  }

  typename DomainTestFixture::DomainType domain(domain_bounds.data(), kDim);
  num_tests = domain.GenerateUniformPointsInDomain(num_tests, &uniform_generator, random_points.data());

  for (int i = 0; i < num_tests; ++i) {
    if (!domain.CheckPointInside(random_points.data() + i*kDim)) {
      ++total_errors;
    }
  }

  return total_errors;
}

/*!\rst
  Check whether LimitUpdate is behaving correctly:

  * max_relative_change is respected
  * no change for updates that are within the domain
  * restricting to nearest boundary when update exits the domain

  \return
    number of test failures
\endrst*/
template <typename DomainTestFixture>
OL_WARN_UNUSED_RESULT int LimitUpdateTest(const DomainTestFixture& domain_test_case) {
  int total_errors = 0;
  const int kDim = domain_test_case.kDimPointGeneration;
  int num_tests = 100;

  // test max_relative_change
  {
    const double max_relative_change = 0.2;
    // simple test case: give a simple domain + constructed update, verify that max_relative_change is applied
    std::vector<ClosedInterval> domain_bounds(kDim);
    std::vector<double> current_point(kDim);
    std::vector<double> update_vector(kDim);

    for (int i = 0; i < kDim; ++i) {
      domain_bounds[i] = {0.0, 10.0 + static_cast<double>(i)};
      current_point[i] = (1.0 + static_cast<double>(i))/100.0;
      update_vector[i] = 5.0 + static_cast<double>(i);
    }
    typename DomainTestFixture::DomainType domain(domain_bounds.data(), kDim);

    domain.LimitUpdate(max_relative_change, current_point.data(), update_vector.data());
    for (int k = 0; k < kDim; ++k) {
      if (!CheckDoubleWithinRelative(update_vector[k], current_point[k] * max_relative_change, std::numeric_limits<double>::epsilon())) {
        ++total_errors;
      }
    }
  }

  UniformRandomGenerator uniform_generator(93143);
  boost::uniform_real<double> uniform_double_normal_vector_noise(-0.1, 0.1);
  const boost::uniform_real<double>& uniform_double_domain_lower_bound = domain_test_case.kUniformDoubleDomainLowerBound;
  const boost::uniform_real<double>& uniform_double_domain_upper_bound = domain_test_case.kUniformDoubleDomainUpperBound;
  std::vector<ClosedInterval> domain_bounds(kDim);
  // domain_bounds w/min edge length 0.03 and max edge length 9
  for (int i = 0; i < kDim; ++i) {
    domain_bounds[i].min = uniform_double_domain_lower_bound(uniform_generator.engine);
    domain_bounds[i].max = uniform_double_domain_upper_bound(uniform_generator.engine);
  }

  std::vector<double> random_points(num_tests*kDim);
  typename DomainTestFixture::DomainType domain(domain_bounds.data(), kDim);
  // generate some random points
  num_tests = domain.GenerateUniformPointsInDomain(num_tests, &uniform_generator, random_points.data());

  const int max_num_boundaries = domain.GetMaxNumberOfBoundaryPlanes();
  std::vector<Plane> boundary_planes(max_num_boundaries, Plane(kDim));
  domain.GetBoundaryPlanes(boundary_planes.data());

  // test LimitUpdate when dealing with exiting boundaries
  // for each point, generate an update vector pointed at each face
  // the update will be the face-normal + noise
  // generate two update step sizes:
  //   1) point inside the domain; check that this LimitUpdate does nothing
  //   2) point outside the domain; check that after LimitUpdate, point is inside domain again
  boost::uniform_int<int> uniform_int_point_index(0, num_tests-1);
  std::vector<double> normal_vector_with_noise(kDim);
  std::vector<double> update_vector_short(kDim);
  std::vector<double> update_vector_long(kDim);
  std::vector<double> update_vector_temp(kDim);
  double const * restrict current_point = random_points.data();
  // huge so that max_relative_change condition will not trigger
  // we already tested that it works
  const double max_relative_change = 10000000.0;
  for (int i = 0; i < num_tests; ++i) {
    // test updates that keep us inside the domain
    // build an update that takes us from one point in the domain to another
    for (int j = 0; j < 10; ++j) {
      int index = uniform_int_point_index(uniform_generator.engine);

      for (int k = 0; k < kDim; ++k) {
        update_vector_short[k] = random_points[index*kDim + k] - current_point[k];
      }

      // update_short should keep us in the domain
      std::copy(update_vector_short.begin(), update_vector_short.end(), update_vector_temp.begin());
      domain.LimitUpdate(max_relative_change, current_point, update_vector_temp.data());

      // no change should have occured so we check for *exact* equality
      for (int k = 0; k < kDim; ++k) {
        if (!CheckDoubleWithin(update_vector_temp[k], update_vector_short[k], 0.0)) {
          ++total_errors;
        }
      }
    }

    // test updates that take us outside the domain
    // build a fake update by traveling too far in the (general) direction of each face
    for (int j = 0; j < max_num_boundaries; ++j) {
      // normal vector is just the first dim entries of each boundary plane
      std::copy(boundary_planes[j].unit_normal.begin(), boundary_planes[j].unit_normal.end(), normal_vector_with_noise.begin());
      // add a little noise (just so we are doing more "interesting" cases than say cartesian unit vectors)
      for (int k = 0; k < kDim; ++k) {
        normal_vector_with_noise[k] += uniform_double_normal_vector_noise(uniform_generator.engine);
      }
      // re-normalize
      double norm = VectorNorm(normal_vector_with_noise.data(), kDim);
      for (int k = 0; k < kDim; ++k) {
        normal_vector_with_noise[k] /= norm;
      }

      double distance_to_boundary = boundary_planes[j].DistanceToPlaneAlongVector(current_point, normal_vector_with_noise.data());

      boost::uniform_real<double> uniform_double_long_dist(1.001*distance_to_boundary, 4.0*distance_to_boundary);
      double long_dist = uniform_double_long_dist(uniform_generator.engine);
      for (int k = 0; k < kDim; ++k) {
        update_vector_long[k] = normal_vector_with_noise[k] * long_dist;
      }

      // update_long should take us out of the domain; verify this first
      for (int k = 0; k < kDim; ++k) {
        update_vector_temp[k] = current_point[k] + update_vector_long[k];
      }
      if (domain.CheckPointInside(update_vector_temp.data())) {
        ++total_errors;
      }

      // LimitUpdate should alter update_vector_long so that we stay in the domain
      std::copy(update_vector_long.begin(), update_vector_long.end(), update_vector_temp.begin());
      domain.LimitUpdate(max_relative_change, current_point, update_vector_temp.data());
      // now check that applying the update keeps us in the domain
      for (int k = 0; k < kDim; ++k) {
        update_vector_temp[k] += current_point[k];
      }

      if (!domain.CheckPointInside(update_vector_temp.data())) {
        ++total_errors;
#ifdef OL_ERROR_PRINT
        PrintMatrix(update_vector_temp.data(), 1, kDim);
#endif
      }
    }

    current_point += kDim;
  }

  return total_errors;
}

/*!\rst
  Wrapper to call all test functions for each DomainTestFixture (which tests a single DomainType)

  \return
    total number of test failures
\endrst*/
template <typename DomainTestFixture>
OL_WARN_UNUSED_RESULT int RunDomainTests() {
  DomainTestFixture domain_test_case;

  int total_errors = 0;
  int current_errors = 0;

  current_errors = CheckPointInsideTest(domain_test_case);
  if (current_errors != 0) {
    OL_PARTIAL_FAILURE_PRINTF("%s: CheckPointInside failed with %d errors\n", OL_CURRENT_FUNCTION_NAME, current_errors);
  }
  total_errors += current_errors;

  current_errors = GeneratePointInDomainTest(domain_test_case);
  if (current_errors != 0) {
    OL_PARTIAL_FAILURE_PRINTF("%s: GeneratePointInDomain failed with %d errors\n", OL_CURRENT_FUNCTION_NAME, current_errors);
  }
  total_errors += current_errors;

  current_errors = GenerateUniformPointsInDomainTest(domain_test_case);
  if (current_errors != 0) {
    OL_PARTIAL_FAILURE_PRINTF("%s: GenerateUniformPointsInDomain failed with %d errors\n", OL_CURRENT_FUNCTION_NAME, current_errors);
  }
  total_errors += current_errors;

  current_errors = LimitUpdateTest(domain_test_case);
  if (current_errors != 0) {
    OL_PARTIAL_FAILURE_PRINTF("%s: LimitUpdate failed with %d errors\n", OL_CURRENT_FUNCTION_NAME, current_errors);
  }
  total_errors += current_errors;

  return total_errors;
}

/*!\rst
  Check whether combinations of test fixture points are inside the RepeatedDomain.

  \return
    number of test failures
\endrst*/
template <typename DomainTestFixture>
OL_WARN_UNUSED_RESULT int CheckPointInsideRepeatedDomainTest(const DomainTestFixture& domain_test_case) {
  // references for convenience
  const std::vector<ClosedInterval>& domain_bounds = domain_test_case.kDomainBoundsCheckPointInside;
  const std::vector<double>& points = domain_test_case.kPointsCheckPointInside;
  const std::vector<bool>& truth = domain_test_case.kTruthCheckPointInside;
  const int dim = domain_test_case.kDimCheckPointInside;
  const int num_points = domain_test_case.kNumPointsCheckPointInside;

  typename DomainTestFixture::DomainType domain(domain_bounds.data(), dim);

  int total_errors = 0;

  // number of points inside & outside domain
  int num_inside = std::count(truth.begin(), truth.end(), true);
  int num_outside = num_points - num_inside;
  // points inside/outside the domain
  std::vector<double> points_inside(num_inside*dim);
  std::vector<double> points_outside(num_outside*dim);
  double * current_point_inside = points_inside.data();
  double * current_point_outside = points_outside.data();
  for (int i = 0; i < num_points; ++i) {
    if (truth[i]) {
      std::copy(points.data() + i*dim, points.data() + (i+1)*dim, current_point_inside);
      current_point_inside += dim;
    } else {
      std::copy(points.data() + i*dim, points.data() + (i+1)*dim, current_point_outside);
      current_point_outside += dim;
    }
  }

  // all points inside
  RepeatedDomain<decltype(domain)> repeated_domain_for_all_inside(domain, num_inside);
  if (repeated_domain_for_all_inside.CheckPointInside(points_inside.data()) != true) {
    ++total_errors;
  }

  // no points inside
  RepeatedDomain<decltype(domain)> repeated_domain_for_none_inside(domain, num_outside);
  if (repeated_domain_for_none_inside.CheckPointInside(points_outside.data()) != false) {
    ++total_errors;
  }

  // some points inside
  RepeatedDomain<decltype(domain)> repeated_domain_for_mix(domain, num_points);
  if (repeated_domain_for_mix.CheckPointInside(points.data()) != false) {
    ++total_errors;
  }

  return total_errors;
}

/*!\rst
  Check whether GeneratePointInDomain generates points that are inside the RepeatedDomain.

  \return
    number of test failures
\endrst*/
template <typename DomainTestFixture>
OL_WARN_UNUSED_RESULT int GeneratePointInDomainRepeatedDomainTest(const DomainTestFixture& domain_test_case) {
  const int kDim = domain_test_case.kDimPointGeneration;
  const int num_tests = 30;
  std::vector<ClosedInterval> domain_bounds(kDim);
  int total_errors = 0;

  UniformRandomGenerator uniform_generator(314);
  const boost::uniform_real<double>& uniform_double_domain_lower_bound = domain_test_case.kUniformDoubleDomainLowerBound;
  const boost::uniform_real<double>& uniform_double_domain_upper_bound = domain_test_case.kUniformDoubleDomainUpperBound;

  // domain_bounds w/min edge length 0.03 and max edge length 9
  for (int i = 0; i < kDim; ++i) {
    domain_bounds[i].min = uniform_double_domain_lower_bound(uniform_generator.engine);
    domain_bounds[i].max = uniform_double_domain_upper_bound(uniform_generator.engine);
  }

  typename DomainTestFixture::DomainType domain(domain_bounds.data(), kDim);
  std::vector<int> num_repeats_list = {1, 4, 10};
  for (auto num_repeats : num_repeats_list) {
    RepeatedDomain<decltype(domain)> repeated_domain(domain, num_repeats);
    std::vector<double> random_point(num_repeats*kDim);

    for (int i = 0; i < num_tests; ++i) {
      if (!repeated_domain.GeneratePointInDomain(&uniform_generator, random_point.data())) {
        ++total_errors;
      }

      if (!repeated_domain.CheckPointInside(random_point.data())) {
        ++total_errors;
      }
    }
  }

  return total_errors;
}

/*!\rst
  Check whether GenerateUniformPointsInDomain generates points that are inside the RepeatedDomain.

  \return
    number of test failures
\endrst*/
template <typename DomainTestFixture>
OL_WARN_UNUSED_RESULT int GenerateUniformPointsInDomainRepeatedDomainTest(const DomainTestFixture& domain_test_case) {
  const int kDim = domain_test_case.kDimPointGeneration;
  std::vector<ClosedInterval> domain_bounds(kDim);
  int total_errors = 0;

  UniformRandomGenerator uniform_generator(3143);
  const boost::uniform_real<double>& uniform_double_domain_lower_bound = domain_test_case.kUniformDoubleDomainLowerBound;
  const boost::uniform_real<double>& uniform_double_domain_upper_bound = domain_test_case.kUniformDoubleDomainUpperBound;

  // domain_bounds w/min edge length 0.03 and max edge length 9
  for (int i = 0; i < kDim; ++i) {
    domain_bounds[i].min = uniform_double_domain_lower_bound(uniform_generator.engine);
    domain_bounds[i].max = uniform_double_domain_upper_bound(uniform_generator.engine);
  }

  // use a dummy domain and verify that the ordering of outputs is correct
  {
    using CoreDomainType = typename DomainTestFixture::DomainType;
    DomainWithMockUniformRandomPoints<CoreDomainType> mock_domain(domain_bounds.data(), kDim);
    const int num_tests = 4;
    std::vector<int> num_repeats_list = {1, 4, 10};
    for (auto num_repeats : num_repeats_list) {
      // DomainWithMockUniformRandomPoints has internal state so we need to reset it before use for consistent results.
      mock_domain.reset();
      RepeatedDomain<decltype(mock_domain)> repeated_domain(mock_domain, num_repeats);
      std::vector<double> random_points(num_tests*num_repeats*kDim);
      int num_tests_actual = repeated_domain.GenerateUniformPointsInDomain(num_tests, &uniform_generator, random_points.data());

      // check that we generate the expected number of 'successful' points
      if (num_repeats > decltype(mock_domain)::kStep && num_tests_actual != num_tests / 2) {
        ++total_errors;
      } else if (num_repeats < decltype(mock_domain)::kStep && num_tests_actual != num_tests) {
        ++total_errors;
      }

      // Now we know the precise order of coordinate values that
      // DomainWithMockUniformRandomPoints.GenerateUniformPointsInDomain() will output.
      // Given that, verify that the RepeatedDomain.GenerateUniformPointsInDomain() reorders the data correctly.
      double counter = 0.0;
      for (int i = 0; i < num_repeats; ++i) {
        for (int j = 0; j < num_tests_actual; ++j) {
          for (int d = 0; d < kDim; ++d) {
            if (!CheckDoubleWithin(random_points[j*num_repeats*kDim + i*kDim + d], counter, 0.0)) {
              ++total_errors;
            }
            counter += 1.0;
          }
        }
        // compensate for the generated values that were skipped due to num_points decreasing
        if (i < decltype(mock_domain)::kStep) {
          counter += static_cast<double>(num_tests - num_tests_actual)*kDim;
        }
      }
    }
  }

  // test with a genuine GenerateUniformPointsInDomain implementation
  {
    const int num_tests = 30;
    typename DomainTestFixture::DomainType domain(domain_bounds.data(), kDim);
    std::vector<int> num_repeats_list = {1, 4, 10};
    for (auto num_repeats : num_repeats_list) {
      RepeatedDomain<decltype(domain)> repeated_domain(domain, num_repeats);
      std::vector<double> random_points(num_tests*num_repeats*kDim);
      int num_tests_actual = repeated_domain.GenerateUniformPointsInDomain(num_tests, &uniform_generator, random_points.data());

      for (int i = 0; i < num_tests_actual; ++i) {
        if (!repeated_domain.CheckPointInside(random_points.data() + i*kDim*num_repeats)) {
          ++total_errors;
        }
      }
    }
  }

  return total_errors;
}

/*!\rst
  Check whether LimitUpdate is behaving correctly:

  * updates staying within the domain are unchanged
  * updates staying within/leaving the domain are identical to the kernel domain's output

  \return
    number of test failures
\endrst*/
template <typename DomainTestFixture>
OL_WARN_UNUSED_RESULT int LimitUpdateRepeatedDomainTest(const DomainTestFixture& domain_test_case) {
  int total_errors = 0;
  const int kDim = domain_test_case.kDimPointGeneration;
  int num_tests = 30;
  int num_points = 200;

  UniformRandomGenerator uniform_generator(93143);
  const boost::uniform_real<double>& uniform_double_domain_lower_bound = domain_test_case.kUniformDoubleDomainLowerBound;
  const boost::uniform_real<double>& uniform_double_domain_upper_bound = domain_test_case.kUniformDoubleDomainUpperBound;
  std::vector<ClosedInterval> domain_bounds(kDim);
  // domain_bounds w/min edge length 0.03 and max edge length 9
  for (int i = 0; i < kDim; ++i) {
    domain_bounds[i].min = uniform_double_domain_lower_bound(uniform_generator.engine);
    domain_bounds[i].max = uniform_double_domain_upper_bound(uniform_generator.engine);
  }

  const int num_repeats = 4;
  std::vector<double> random_points(num_repeats*num_points*kDim);
  typename DomainTestFixture::DomainType domain(domain_bounds.data(), kDim);
  RepeatedDomain<decltype(domain)> repeated_domain(domain, num_repeats);
  // generate some random points
  num_points = repeated_domain.GenerateUniformPointsInDomain(num_points, &uniform_generator, random_points.data());

  // generate two update step sizes:
  //   1) point inside the domain; check that this LimitUpdate does nothing
  //   2) point maybe outside the domain; check RepeatedDomain and kernel domain LimitUpdate operations are identical
  boost::uniform_int<int> uniform_int_point_index(0, num_points-1);
  std::vector<double> update_vector(num_repeats*kDim);
  std::vector<double> update_vector_temp(num_repeats*kDim);
  std::vector<double> update_vector_per_repeat(kDim);
  // huge so that max_relative_change condition will not trigger
  const double max_relative_change = 10000000.0;
  for (int j = 0; j < num_tests; ++j) {
    int index_first = uniform_int_point_index(uniform_generator.engine);
    int index_second = (index_first + 1) % num_points;

    // this update is guaranteed to keep us in the domain
    for (int k = 0; k < num_repeats*kDim; ++k) {
      update_vector[k] = random_points[index_second*kDim*num_repeats + k] - random_points[index_first*kDim*num_repeats + k];
    }
    std::copy(update_vector.begin(), update_vector.end(), update_vector_temp.begin());
    repeated_domain.LimitUpdate(max_relative_change, random_points.data() + index_first*kDim*num_repeats, update_vector_temp.data());

    // no change should have occured so we check for *exact* equality
    for (int k = 0; k < num_repeats*kDim; ++k) {
      if (!CheckDoubleWithin(update_vector_temp[k], update_vector[k], 0.0)) {
        ++total_errors;
      }
    }

    // now magnify the update by a large amount to give us a reasonable chance of exiting the domain
    const double scale = 10.0;
    for (auto& entry : update_vector) {
      entry *= scale;
    }
    std::copy(update_vector.begin(), update_vector.end(), update_vector_temp.begin());
    repeated_domain.LimitUpdate(max_relative_change, random_points.data() + index_first*kDim*num_repeats, update_vector_temp.data());

    // the RepeatedDomain version calls the domain kernel in a loop, the results should match *exactly*
    for (int k = 0; k < num_repeats; ++k) {
      std::copy(update_vector.begin() + k*kDim, update_vector.begin() + (k+1)*kDim, update_vector_per_repeat.begin());
      domain.LimitUpdate(max_relative_change, random_points.data() + index_first*kDim*num_repeats + k*kDim, update_vector_per_repeat.data());
      for (int d = 0; d < kDim; ++d) {
        if (!CheckDoubleWithin(update_vector_temp[k*kDim + d], update_vector_per_repeat[d], 0.0)) {
          ++total_errors;
        }
      }
    }
  }  // end loop over num_tests

  return total_errors;
}

/*!\rst
  Wrapper to call all test functions for each DomainTestFixture (which tests a single DomainType)

  \return
    total number of test failures
\endrst*/
template <typename DomainTestFixture>
OL_WARN_UNUSED_RESULT int RunRepeatedDomainTests() {
  DomainTestFixture domain_test_case;

  int total_errors = 0;
  int current_errors = 0;

  current_errors = CheckPointInsideRepeatedDomainTest(domain_test_case);
  if (current_errors != 0) {
    OL_PARTIAL_FAILURE_PRINTF("%s: CheckPointInsideRepeatedDomain failed with %d errors\n", OL_CURRENT_FUNCTION_NAME, current_errors);
  }
  total_errors += current_errors;

  current_errors = GeneratePointInDomainRepeatedDomainTest(domain_test_case);
  if (current_errors != 0) {
    OL_PARTIAL_FAILURE_PRINTF("%s: GeneratePointInDomainRepeatedDomain failed with %d errors\n", OL_CURRENT_FUNCTION_NAME, current_errors);
  }
  total_errors += current_errors;

  current_errors = GenerateUniformPointsInDomainRepeatedDomainTest(domain_test_case);
  if (current_errors != 0) {
    OL_PARTIAL_FAILURE_PRINTF("%s: GenerateUniformPointsInDomainRepeatedDomain failed with %d errors\n", OL_CURRENT_FUNCTION_NAME, current_errors);
  }
  total_errors += current_errors;

  current_errors = LimitUpdateRepeatedDomainTest(domain_test_case);
  if (current_errors != 0) {
    OL_PARTIAL_FAILURE_PRINTF("%s: LimitUpdateRepeatedDomain failed with %d errors\n", OL_CURRENT_FUNCTION_NAME, current_errors);
  }
  total_errors += current_errors;

  return total_errors;
}

/*!\rst
  Checks if valid domain can be constructed from a set of ClosedInterval.

  \param
    :domain_bounds: vector of ClosedInterval specifying the region boundaries
  \return
    true if the resulting domain is valid
\endrst*/
template <typename DomainType>
OL_WARN_UNUSED_RESULT int CheckDomain(const std::vector<ClosedInterval>& domain_bounds) {
  bool domain_valid;
  try {
    DomainType domain(domain_bounds.data(), domain_bounds.size());
    domain_valid = true;
  } catch (const BoundsException<double>& exception) {
    domain_valid = false;
  } catch (const OptimalLearningException& exception) {
    domain_valid = false;
  }
  return domain_valid;
}

/*!\rst
  Checks if valid RepeatedDomain can be constructed from an input Domain and num_repeats.

  \param
    :domain: the domain to repeat
    :num_repeats: number of times to repeat the input domain
  \return
    true if the resulting domain is valid
\endrst*/
template <typename DomainType>
OL_WARN_UNUSED_RESULT int CheckRepeatedDomain(const DomainType& domain, int num_repeats) {
  bool domain_valid;
  try {
    RepeatedDomain<DomainType> repeated_domain(domain, num_repeats);
    domain_valid = true;
  } catch (const LowerBoundException<int>& exception) {
    domain_valid = false;
  }
  return domain_valid;
}

/*!\rst
  Check whether certain domain ctor inputs produce valid/invalid domains as expected.

  \param
    :domain_type: type (enum) of domain to test
  \return
    number of test failures
\endrst*/
template <typename DomainType>
OL_WARN_UNUSED_RESULT int InvalidDomainTests(DomainTypes domain_type) {
  int num_errors = 0;
  // all intervals valid
  std::vector<ClosedInterval> domain_bounds_valid = {
    {0.0, 1.0},   // same bounds as simplex bounding box
    {-1.0, 1.0},  // exceeds the simplex b-box left boundary (-1.0 < 0.0)
    {0.0, 0.9},   // inside the simplex b-box
    {0.0, 2.0}};  // exceeds the simplex b-box right boundary (2.0 > 1.0)

  // all intervals invalid
  std::vector<ClosedInterval> domain_bounds_invalid(domain_bounds_valid);
  std::for_each(domain_bounds_invalid.begin(), domain_bounds_invalid.end(), [](ClosedInterval& interval) { std::swap(interval.min, interval.max); });  // NOLINT(runtime/references)

  // one interval invalid
  std::vector<ClosedInterval> domain_bounds_invalid2(domain_bounds_valid);
  std::swap(domain_bounds_invalid2[2].min, domain_bounds_invalid2[2].max);

  if (CheckDomain<DomainType>(domain_bounds_valid) != true) {
    ++num_errors;
  }
  if (CheckDomain<DomainType>(domain_bounds_invalid) != false) {
    ++num_errors;
  }
  if (CheckDomain<DomainType>(domain_bounds_invalid2) !=false) {
    ++num_errors;
  }

  if (domain_type == DomainTypes::kSimplex) {
    // tensor product region does not intersect [0,1]X[0,1]X... bounding box of unit simplex
    std::vector<ClosedInterval> domain_bounds_invalid3 = {
      {-2.0, -1.0},  // entirely outside [0, 1]
      {-1.0, 1.0},
      {0.0, 0.9},
      {0.0, 2.0}};

    // tensor product region's lower left corner coordinates sum to >= 1.0 (i.e., outside the simplex plane)
    std::vector<ClosedInterval> domain_bounds_invalid4 = {
      {0.3, 1.0},
      {0.4, 1.0},
      {0.1, 0.9},
      {0.2, 2.0}};

    double sum = 0.0;
    for (const auto& interval : domain_bounds_invalid4) {
      sum += interval.min;
    }
    if (sum < 1.0) {
      ++num_errors;
    }

    if (CheckDomain<DomainType>(domain_bounds_invalid3) !=false) {
      ++num_errors;
    }
    if (CheckDomain<DomainType>(domain_bounds_invalid4) !=false) {
      ++num_errors;
    }
  }

  return num_errors;
}

/*!\rst
  Check whether certain RepeatedDomain ctor inputs produce valid/invalid domains as expected.

  \param
    :domain_type: type (enum) of domain to test
  \return
    number of test failures
\endrst*/
OL_WARN_UNUSED_RESULT int InvalidRepeatedDomainTests() {
  int num_errors = 0;
  // all intervals valid
  std::vector<ClosedInterval> domain_bounds_valid = {
    {0.0, 1.0},
    {-1.0, 1.0}};

  TensorProductDomain domain(domain_bounds_valid.data(), domain_bounds_valid.size());

  const int num_tests = 4;
  std::vector<int> num_repeats_list = {-13, 0, 1, 78};
  std::vector<bool> valid_domain_truth = {false, false, true, true};
  for (int i = 0; i < num_tests; ++i) {
    if (CheckRepeatedDomain(domain, num_repeats_list[i]) != valid_domain_truth[i]) {
      ++num_errors;
    }
  }

  return num_errors;
}

/*!\rst
  Wrapper around test functions for the RepeatedDomain class.

  .. Note:: This originally lived inline in DomainTests(), but that caused GPP.so to print from OL_ERROR_PRINTF
    macros even when they were explicitly disabled (i.e., in the HACK at the top of this file).

  \return
    number of RepeatedDomain test failures
\endrst*/
int RepeatedDomainTests() {
  int total_errors = 0;
  int current_errors = 0;

  current_errors = RunRepeatedDomainTests<TensorProductDomainTestFixture>();
  if (current_errors != 0) {
    OL_PARTIAL_FAILURE_PRINTF("RepeatedDomain<TensorProductDomain> failed with %d errors\n", current_errors);
  }
  total_errors += current_errors;

  current_errors = RunRepeatedDomainTests<SimplexIntersectTensorProductDomainTestFixture>();
  if (current_errors != 0) {
    OL_PARTIAL_FAILURE_PRINTF("RepeatedDomain<SimplexIntersectTensorProductDomain> failed with %d errors\n", current_errors);
  }
  total_errors += current_errors;

  current_errors = InvalidRepeatedDomainTests();
  if (current_errors != 0) {
    OL_PARTIAL_FAILURE_PRINTF("RepeatedDomain accepts invalid num_repeats with %d errors\n", current_errors);
  }
  total_errors += current_errors;

  return total_errors;
}

}  // end unnamed namespace

int DomainTests() {
  int total_errors = 0;
  int current_errors = 0;

  current_errors = RunDomainTests<TensorProductDomainTestFixture>();
  if (current_errors != 0) {
    OL_PARTIAL_FAILURE_PRINTF("TensorProductDomain failed with %d errors\n", current_errors);
  }
  total_errors += current_errors;

  current_errors = RunDomainTests<SimplexIntersectTensorProductDomainTestFixture>();
  if (current_errors != 0) {
    OL_PARTIAL_FAILURE_PRINTF("SimplexIntersectTensorProductDomain failed with %d errors\n", current_errors);
  }
  total_errors += current_errors;

  current_errors = InvalidDomainTests<TensorProductDomain>(DomainTypes::kTensorProduct);
  if (current_errors != 0) {
    OL_PARTIAL_FAILURE_PRINTF("TensorProductDomain accepts invalid domains with %d errors\n", current_errors);
  }
  total_errors += current_errors;

  current_errors = InvalidDomainTests<SimplexIntersectTensorProductDomain>(DomainTypes::kSimplex);
  if (current_errors != 0) {
    OL_PARTIAL_FAILURE_PRINTF("SimplexIntersectTensorProductDomain accepts invalid domains with %d errors\n", current_errors);
  }
  total_errors += current_errors;

  // run RepeatedDomain tests
  total_errors += RepeatedDomainTests();

  return total_errors;
}

}  // end namespace optimal_learning
