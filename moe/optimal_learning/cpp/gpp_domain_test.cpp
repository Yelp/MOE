// gpp_domain_test.cpp
/*
  This file contains functions for testing the functions and classes in gpp_domain.hpp.

  It has several templated test functions (on DomainTestFixture) for:
  <> CheckPointInDomain: specify a domain, point list, and truth values.  checks that points are/are not inside
  <> GeneratePointInDomain: generates some random points, checks that they are inside the domain
  <> GenerateUniformPointsInDomain: generates uniformly distributed points, checks that they are inside the domain
     TODO(eliu): how do you check that the points are actually uniformly distributed???
  <> LimitUpdate: starting from points within the domain, checks that:
     <> updates to another point in the domain remain unchanged
     <> updates to a point outside the domain are limited such that the new endpoint is in the domain

  These are all wrapper'd in RunDomainTests<>(), which is templated on DomainTestFixture.

  To use these tests, define a DomainTestFixture struct with the following fields (and values for them):
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
*/

#include "gpp_domain_test.hpp"

#include <algorithm>
#include <limits>
#include <vector>

#include <boost/random/uniform_int.hpp>  // NOLINT(build/include_order)

#include "gpp_common.hpp"
// HACK: temporarily disable printing (this will go away when we switch to GoogleTest)
#define OL_TEMP_ERROR_PRINT OL_ERROR_PRINT
#define OL_TEMP_WARNING_PRINT OL_WARNING_PRINT
#undef OL_ERROR_PRINT
#undef OL_WARNING_PRINT
#include "gpp_domain.hpp"
#define OL_ERROR_PRINT OL_TEMP_ERROR_PRINT
#define OL_WARNING_PRINT OL_TEMP_WARNING_PRINT
#include "gpp_geometry.hpp"
#include "gpp_logging.hpp"
#include "gpp_random.hpp"
#include "gpp_test_utils.hpp"

namespace optimal_learning {

namespace {

/*
  DomainTestFixture for TensorProductDomain.  See file docs for details on the layout.
*/
struct TensorProductDomainTestFixture final {
  using DomainType = TensorProductDomain;

  // use with CheckPointInsideTest
  const int kDimCheckPointInside = 4;

  // Domain bounds (min, max pairs)
  const std::vector<ClosedInterval> kDomainBoundsCheckPointInside = {
    {0.0, 1.0},
    {0.0, 1.0},
    {0.0, 1.0},
    {0.0, 2.0}};

  const std::vector<double> kPointsCheckPointInside = {
    0.0, 0.0, 0.0, 0.0,    // origin
    0.5, 0.4, 0.3, 0.8,    // hypercube interior
    0.2, 1.2, 0.1, 0.7,    // outside the y-face
    1.0, 0.5, 0.4, 0.2,    // on the x-face
    1.3, -1.0, 2.0, 2.5};  // exterior

  // Truth for whether each point of kPointsCheckPointInside is inside/outside the domain
  const std::vector<bool> kTruthCheckPointInside = {
    true,
    true,
    false,
    true,
    false};

  const int kNumPointsCheckPointInside = kPointsCheckPointInside.size() / kDimCheckPointInside;

  // use with GeneratePointInDomainTest, GenerateUniformPointsInDomainTest, LimitUpdateTest
  const int kDimPointGeneration = 5;
  const boost::uniform_real<double> kUniformDoubleDomainLowerBound = decltype(kUniformDoubleDomainLowerBound)(-5.0, -0.01);
  const boost::uniform_real<double> kUniformDoubleDomainUpperBound = decltype(kUniformDoubleDomainUpperBound)(0.02, 4.0);
};

/*
  DomainTestFixture for SimplexIntersectTensorProductDomain.  See file docs for details on the layout.
*/
struct SimplexIntersectTensorProductDomainTestFixture final {
  using DomainType = SimplexIntersectTensorProductDomain;

  // use with CheckPointInsideTest
  const int kDimCheckPointInside = 4;

  // Domain bounds (min, max pairs)
  const std::vector<ClosedInterval> kDomainBoundsCheckPointInside = {
    {0.0, 1.0},   // same bounds as simplex bounding box
    {-1.0, 1.0},  // exceeds the simplex b-box left boundary (-1.0 < 0.0)
    {0.0, 0.9},   // inside the simplex b-box
    {0.0, 2.0}};  // exceeds the simplex b-box right boundary (2.0 > 1.0)

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

  // Truth for whether each point of kPointsCheckPointInside is inside/outside the domain
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

  const int kNumPointsCheckPointInside = kPointsCheckPointInside.size() / kDimCheckPointInside;

  // use with GeneratePointInDomainTest, GenerateUniformPointsInDomainTest, LimitUpdateTest
  const int kDimPointGeneration = 5;
  const boost::uniform_real<double> kUniformDoubleDomainLowerBound = decltype(kUniformDoubleDomainLowerBound)(0.05, 0.15);
  const boost::uniform_real<double> kUniformDoubleDomainUpperBound = decltype(kUniformDoubleDomainUpperBound)(0.2, 0.6);
};

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

// TODO(eliu): HOW do I test whether a point distribution is uniform???
// this test only checks that all points are in the domain
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

/*
  Wrapper to call all test functions for each DomainTestFixture (which tests a single DomainType)
*/
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

/*
  Checks if valid domain can be constructed from a set of ClosedInterval.

  INPUTS:
  domain_bounds: vector of ClosedInterval specifying the region boundaries
  OUTPUTS:
  true if the resulting domain is valid
*/
template <typename DomainType>
OL_WARN_UNUSED_RESULT int CheckDomain(const std::vector<ClosedInterval>& domain_bounds) {
  bool domain_valid;
  try {
    DomainType domain(domain_bounds.data(), domain_bounds.size());
    domain_valid = true;
  } catch (const RuntimeException& exception) {
    domain_valid = false;
  } catch (const BoundsException<double>& exception) {
    domain_valid = false;
  }
  return domain_valid;
}

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

}  // end unnamed namespace

/*
  Calls RunDomainTests<>() for each DomainTestFixture/DomainType pair.
*/
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

  return total_errors;
}

}  // end namespace optimal_learning
