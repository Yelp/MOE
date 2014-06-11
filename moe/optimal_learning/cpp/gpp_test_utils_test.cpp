/*!
  \file gpp_test_utils_test.cpp
  \rst
  This file contains functions for testing the functions and classes in gpp_test_utils.hpp. These
  tests are generally pretty simple since these functions implicitly work; e.g., if
  CheckDoubleWithinRelative() were wrong, other tests wouldn't make any sense.

  TODO(GH-122): implement the rest of the unit tests for gpp_test_utils
\endrst*/

#include "gpp_test_utils_test.hpp"

#include <algorithm>
#include <limits>
#include <vector>

#include <boost/random/uniform_real.hpp>  // NOLINT(build/include_order)

#include "gpp_common.hpp"
#include "gpp_covariance.hpp"
#include "gpp_geometry.hpp"
#include "gpp_logging.hpp"
#include "gpp_math.hpp"
#include "gpp_random.hpp"
#include "gpp_test_utils.hpp"

namespace optimal_learning {

namespace {

/*!\rst
  Assumes CheckDoubleWithin() is already verified.
\endrst*/
OL_WARN_UNUSED_RESULT int FillRandomCovarianceHyperparametersTest() {
  int total_errors = 0;
  const int kDim = 5;

  UniformRandomGenerator uniform_generator(314);
  UniformRandomGenerator uniform_generator_original(uniform_generator);

  boost::uniform_real<double> uniform_double(0.32, 4.7);

  SquareExponential covariance(kDim, 1.0, 1.0);
  int num_hyperparameters = covariance.GetNumberOfHyperparameters();

  std::vector<double> hyperparameters(num_hyperparameters);
  std::vector<double> hyperparameters_covariance(num_hyperparameters);

  FillRandomCovarianceHyperparameters(uniform_double, &uniform_generator, &hyperparameters, &covariance);

  // Verify that hyperparameters output and hyperparameters of covariance are the same
  covariance.GetHyperparameters(hyperparameters_covariance.data());
  if (!CheckIntEquals(hyperparameters.size(), hyperparameters_covariance.size())) {
    ++total_errors;
  }
  if (!std::equal(hyperparameters.begin(), hyperparameters.end(), hyperparameters_covariance.begin())) {
    ++total_errors;
  }

  // Verify that hyperparameters fall within the [min, max] range of the uniform_real interval
  ClosedInterval interval = {uniform_double.a(), uniform_double.b()};
  for (const auto hyperparameter : hyperparameters) {
    if (!interval.IsInside(hyperparameter)) {
      ++total_errors;
    }
  }

  // Verify that uniform_generator changed
  if (uniform_generator == uniform_generator_original) {
    ++total_errors;
  }

  // Verify that the uniform_generator was called hyperparameters.size() times
  uniform_generator_original.engine.discard(num_hyperparameters);
  if (uniform_generator != uniform_generator_original) {
    ++total_errors;
  }

  return total_errors;
}

/*!\rst
  Assumes CheckDoubleWithin() is already verified.
\endrst*/
OL_WARN_UNUSED_RESULT int FillRandomDomainBoundsTest() {
  int total_errors = 0;
  const int kDim = 5;

  UniformRandomGenerator uniform_generator(314);
  UniformRandomGenerator uniform_generator_original(uniform_generator);
  boost::uniform_real<double> uniform_double_lower_bound(-2.0, 0.5);
  boost::uniform_real<double> uniform_double_upper_bound(2.0, 3.5);

  std::vector<ClosedInterval> domain_bounds(kDim);

  FillRandomDomainBounds(uniform_double_lower_bound, uniform_double_upper_bound, &uniform_generator, &domain_bounds);

  // Verify that uniform_generator changed
  if (uniform_generator == uniform_generator_original) {
    ++total_errors;
  }

  // Verify that the uniform_generator was called 2*kDim times
  uniform_generator_original.engine.discard(2*kDim);
  if (uniform_generator != uniform_generator_original) {
    ++total_errors;
  }

  // Verify that the ClosedIntervals' min, max values fall within the specified
  // bounds of the uniform_real input
  ClosedInterval lower_interval = {uniform_double_lower_bound.a(), uniform_double_lower_bound.b()};
  ClosedInterval upper_interval = {uniform_double_upper_bound.a(), uniform_double_upper_bound.b()};
  for (const auto domain_interval : domain_bounds) {
    if (!lower_interval.IsInside(domain_interval.min)) {
      ++total_errors;
    }
    if (!upper_interval.IsInside(domain_interval.max)) {
      ++total_errors;
    }
  }

  return total_errors;
}

/*!\rst
  Assumes FillRandomCovarianceHyperparameters(), FillRandomDomainBounds(), and CheckDoubleWithin()
  are already verified.
\endrst*/
OL_WARN_UNUSED_RESULT int FillRandomGaussianProcess() {
  int total_errors = 0;
  const int kDim = 5;
  const int kNumSampled = 2;
  const int kNumToDraw = 3;
  const int kNumPoints = kNumSampled + kNumToDraw;

  UniformRandomGenerator uniform_generator(314);
  SquareExponential covariance(kDim, 0.4, 0.9);

  std::vector<double> points_sampled(kNumPoints*kDim, std::numeric_limits<double>::quiet_NaN());
  std::vector<double> points_sampled_value(kNumPoints, std::numeric_limits<double>::quiet_NaN());
  std::vector<double> noise_variance(kNumPoints, std::numeric_limits<double>::quiet_NaN());

  boost::uniform_real<double> uniform_double_lower_bound(-2.0, 0.5);
  boost::uniform_real<double> uniform_double_upper_bound(2.0, 3.5);
  std::vector<ClosedInterval> domain_bounds(kDim);
  FillRandomDomainBounds(uniform_double_lower_bound, uniform_double_upper_bound, &uniform_generator, &domain_bounds);
  TensorProductDomain domain(domain_bounds.data(), kDim);
  domain.GenerateUniformPointsInDomain(kNumSampled, &uniform_generator, points_sampled.data());

  boost::uniform_real<double> uniform_double_value(-0.7, 0.5);
  std::generate(points_sampled_value.begin(), points_sampled_value.begin() + kNumSampled, [&]() {
    return uniform_double_value(uniform_generator.engine);
  });

  boost::uniform_real<double> uniform_double_noise(0.01, 0.1);
  std::generate(noise_variance.begin(), noise_variance.begin() + kNumSampled, [&]() {
    return uniform_double_noise(uniform_generator.engine);
  });

  GaussianProcess gaussian_process(covariance, points_sampled.data(), points_sampled_value.data(), noise_variance.data(), kDim, 0);

  // add in kNumSampled points
  FillRandomGaussianProcess(points_sampled.data(), noise_variance.data(), kDim, kNumSampled, points_sampled_value.data(), &gaussian_process);

  // Verify that the GP's num_sampled value is correct
  if (!CheckIntEquals(kNumSampled, gaussian_process.num_sampled())) {
    ++total_errors;
  }

  // Verify that the points_sampled, points_sampled_value, and noise_variance match
  // between the direct output and the GaussianProcess
  if (!std::equal(gaussian_process.points_sampled().begin(), gaussian_process.points_sampled().end(), points_sampled.begin())) {
    ++total_errors;
  }

  if (!std::equal(gaussian_process.points_sampled_value().begin(), gaussian_process.points_sampled_value().end(), points_sampled_value.begin())) {
    ++total_errors;
  }

  if (!std::equal(gaussian_process.noise_variance().begin(), gaussian_process.noise_variance().end(), noise_variance.begin())) {
    ++total_errors;
  }

  // now add in kNumtoDraw points and re-verify
  domain.GenerateUniformPointsInDomain(kNumToDraw, &uniform_generator, points_sampled.data() + kDim*kNumSampled);

  std::generate(points_sampled_value.begin() + kNumSampled, points_sampled_value.begin() + kNumPoints, [&]() {
    return uniform_double_value(uniform_generator.engine);
  });

  std::generate(noise_variance.begin() + kNumSampled, noise_variance.begin() + kNumPoints, [&]() {
    return uniform_double_noise(uniform_generator.engine);
  });

  // add in kNumToDraw points
  FillRandomGaussianProcess(points_sampled.data() + kDim*kNumSampled, noise_variance.data() + kNumSampled, kDim, kNumToDraw, points_sampled_value.data() + kNumSampled, &gaussian_process);

  // Verify that the GP's num_sampled value is correct
  if (!CheckIntEquals(kNumPoints, gaussian_process.num_sampled())) {
    ++total_errors;
  }

  // Verify that the points_sampled, points_sampled_value, and noise_variance match
  // between the direct output and the GaussianProcess
  if (!std::equal(points_sampled.begin(), points_sampled.end(), gaussian_process.points_sampled().begin())) {
    ++total_errors;
  }

  if (!std::equal(points_sampled_value.begin(), points_sampled_value.end(), gaussian_process.points_sampled_value().begin())) {
    ++total_errors;
  }

  if (!std::equal(noise_variance.begin(), noise_variance.end(), gaussian_process.noise_variance().begin())) {
    ++total_errors;
  }

  return total_errors;
}

}  // end unnamed namespace

int TestUtilsTests() {
  int total_errors = 0;
  int current_errors = 0;

  current_errors = FillRandomCovarianceHyperparametersTest();
  if (current_errors != 0) {
    OL_PARTIAL_FAILURE_PRINTF("FillRandomCovarianceHyperparameters failed with %d errors\n", current_errors);
  } else {
    OL_PARTIAL_SUCCESS_PRINTF("FillRandomCovarianceHyperparameters passed\n");
  }
  total_errors += current_errors;

  current_errors = FillRandomDomainBoundsTest();
  if (current_errors != 0) {
    OL_PARTIAL_FAILURE_PRINTF("FillRandomDomainBounds failed with %d errors\n", current_errors);
  } else {
    OL_PARTIAL_SUCCESS_PRINTF("FillRandomDomainBounds passed\n");
  }
  total_errors += current_errors;

  current_errors = FillRandomGaussianProcess();
  if (current_errors != 0) {
    OL_PARTIAL_FAILURE_PRINTF("FillRandomGaussianProcess failed with %d errors\n", current_errors);
  } else {
    OL_PARTIAL_SUCCESS_PRINTF("FillRandomGaussianProcess passed\n");
  }
  total_errors += current_errors;

  return total_errors;
}

}  // end namespace optimal_learning
