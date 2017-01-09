/*!
  \file gpp_random_test.cpp
  \rst
  This file contains functions for testing the functions and classes in gpp_random.hpp.  There are also a number of simple
  supporting routines.  See header for comments on the general layout of these tests.
\endrst*/

#include "gpp_random_test.hpp"

#include <algorithm>
#include <limits>
#include <unordered_set>
#include <vector>

#include <boost/random/uniform_int.hpp>  // NOLINT(build/include_order)
#include <boost/random/uniform_real.hpp>  // NOLINT(build/include_order)

#include "gpp_common.hpp"
#include "gpp_exception.hpp"
#include "gpp_geometry.hpp"
#include "gpp_logging.hpp"
#include "gpp_random.hpp"
#include "gpp_test_utils.hpp"

namespace optimal_learning {

namespace {

/*!\rst
  Randomly generates points in a domain and ensures that the results are valid.  ComputeRandomPointInDomain guarantees
  nothing further about the distribution of its outputs.

  \return
    number of randomly generated points that were not inside the domain
\endrst*/
OL_WARN_UNUSED_RESULT int RandomPointInDomainTest() {
  static const int kDim = 5;
  const int num_tests = 50;
  ClosedInterval domain[kDim];
  double random_point[kDim];
  int total_errors = 0;

  UniformRandomGenerator uniform_generator(314);
  boost::uniform_real<double> uniform_double_domain_lower_bound(-5.0, -0.01);
  boost::uniform_real<double> uniform_double_domain_upper_bound(0.02, 4.0);

  // domain w/min edge length 0.03 and max edge length 9
  for (int i = 0; i < kDim; ++i) {
    domain[i].min = uniform_double_domain_lower_bound(uniform_generator.engine);
    domain[i].max = uniform_double_domain_upper_bound(uniform_generator.engine);
  }

  for (int i = 0; i < num_tests; ++i) {
    ComputeRandomPointInDomain(domain, kDim, &uniform_generator, random_point);

    if (!CheckPointInHypercube(domain, random_point, kDim)) {
      ++total_errors;
    }
  }

  if (total_errors != 0) {
    OL_PARTIAL_FAILURE_PRINTF("random_point_in_domain generated points outside the domain\n");
  }

  return total_errors;
}

/*!\rst
  Just your basic bubble sort.

  Need the ability to sort matrices ``A_{ij}`` in blocks of ``A_{i*}``, doing comparisons only on
  ``j``-th entries.  This is (as far as I know) awkward with STL vectors/sort.
\endrst*/
OL_NONNULL_POINTERS void bubble_sort(int dim_to_sort, int dim, int num_points, double * restrict points) noexcept {
  int newi;

  for (int i = num_points - 1; i > 0; --i) {
    newi = 0;
    for (int j = 1; j <= i; ++j) {
      if (points[(j-1)*dim + dim_to_sort] > points[j*dim + dim_to_sort]) {
        double * restrict point_one = points + (j-1)*dim;
        double * restrict point_two = points + j*dim;
        std::swap_ranges(point_one, point_one + dim, point_two);
        newi = j;
      }
    }
    // after a given pass (loop over j), all elements after the most recent swap are
    // already sorted.  skip over them.
    i = newi;
  }
}

/*!\rst
  Check that the latin hypercube point generation routine generates points in that are:

  1. in the domain
  2. properly distributed

  Latin hypercube sampling with N points divides a d-dimensional domain into N subranges in
  each ordinate direction.  For example, in the 2D domain [0,1]x[0,1], with N=8, the square
  is divided up like a chess board.

  Then lathin hypercube sampling guarantees that there can only be ONE point per row and per column.  In
  the chess analogy, this sampling places N rooks so that no 2 attack each other.  Since placing each point
  eliminates 1 row and 1 column, this is always possible (let the rooks be pidgeons).

  So the test is as follows:

  1. use latin hypercube sampling to sample N points
  2. for each spatial dimension d
  3. sort the points along their d-th coordinate
  4. check that each subrange only contains 1 point

  \return
    number of LHC points that were improperly distributed
\endrst*/
OL_WARN_UNUSED_RESULT int HypercubePointInDomainTest() {
  static const int kDim = 5;
  const int num_tests = 50;
  static const int kNumberOfSamples = 30;
  ClosedInterval domain[kDim];
  double random_points[kDim*kNumberOfSamples];
  double subcube_edge_length, min_val, max_val;
  int errors_this_iteration;
  int total_errors = 0;

  UniformRandomGenerator uniform_generator(314);
  boost::uniform_real<double> uniform_double_domain_lower_bound(-5.0, -0.01);
  boost::uniform_real<double> uniform_double_domain_upper_bound(0.02, 4.0);

  // domain w/min edge length 0.03 and max edge length 9
  for (int i = 0; i < kDim; ++i) {
    domain[i].min = uniform_double_domain_lower_bound(uniform_generator.engine);
    domain[i].max = uniform_double_domain_upper_bound(uniform_generator.engine);
  }

  for (int i = 0; i < num_tests; ++i) {
    ComputeLatinHypercubePointsInDomain(domain, kDim, kNumberOfSamples, &uniform_generator, random_points);

    for (int j = 0; j < kNumberOfSamples; ++j) {
      if (!CheckPointInHypercube(domain, random_points + j*kDim, kDim)) {
        ++total_errors;
      }
    }

    for (int k = 0; k < kDim; ++k) {
      subcube_edge_length = (domain[k].Length())/static_cast<double>(kNumberOfSamples);
      bubble_sort(k, kDim, kNumberOfSamples, random_points);

      // i-th point (sorted) must fall somewhere in the i-th slice of the hypercube
      for (int j = 0; j < kNumberOfSamples; ++j) {
        min_val = domain[k].min + subcube_edge_length*j;
        max_val = min_val + subcube_edge_length;

        errors_this_iteration = 0;
        if (random_points[j*kDim + k] > max_val) {
          ++errors_this_iteration;
        }
        if (random_points[j*kDim + k] < min_val) {
          ++errors_this_iteration;
        }
        total_errors += errors_this_iteration;
      }
    }
  }

  if (total_errors != 0) {
    OL_PARTIAL_FAILURE_PRINTF("hypercube_point_in_domain generated invalid point distributions\n");
  }

  return total_errors;
}

/*!\rst
  Test random point generation in a unit simplex.

  \return
    number of randomly generated points that were not in the unit simplex
\endrst*/
OL_WARN_UNUSED_RESULT int RandomPointInUnitSimplexTest() {
  static const int kDim = 5;
  const int num_tests = 50;
  static const int kNumberOfSamples = 30;
  double random_points[kDim*kNumberOfSamples];
  int total_errors = 0;

  UniformRandomGenerator uniform_generator(314);

  for (int i = 0; i < num_tests; ++i) {
    ComputeUniformPointsInUnitSimplex(kDim, kNumberOfSamples, &uniform_generator, random_points);

    for (int j = 0; j < kNumberOfSamples; ++j) {
      if (!CheckPointInUnitSimplex(random_points + j*kDim, kDim)) {
        ++total_errors;
      }
    }
  }

  if (total_errors != 0) {
    OL_PARTIAL_FAILURE_PRINTF("RandomPointInUnitSimplex generated invalid point distributions\n");
  }

  return total_errors;
}

}  // end unnamed namespace

int RunRandomPointGeneratorTests() {
  int current_errors;
  int total_errors = 0;

  current_errors = RandomPointInDomainTest();
  total_errors += current_errors;
  if (current_errors != 0) {
    OL_PARTIAL_FAILURE_PRINTF("Random Point in Domain errors = %d\n", current_errors);
  }

  current_errors = HypercubePointInDomainTest();
  total_errors += current_errors;
  if (current_errors != 0) {
    OL_PARTIAL_FAILURE_PRINTF("Latin Hypercube Points in Domain errors = %d\n", current_errors);
  }

  current_errors = RandomPointInUnitSimplexTest();
  total_errors += current_errors;
  if (current_errors != 0) {
    OL_PARTIAL_FAILURE_PRINTF("Random Point in Unit Simplex errors = %d\n", current_errors);
  }

  return total_errors;
}

namespace {

/*!\rst
  Checks that all elements of a vector are unique.

  Modifies the vector (by sorting).

  \input
    :input_vector: vector to be checked
  \return
    true if all elements are distinct
\endrst*/
template <typename T>
OL_WARN_UNUSED_RESULT bool CheckAllElementsUnique(const std::vector<T>& input_vector) {
  return std::unordered_set<T>(input_vector.begin(), input_vector.end()).size() == input_vector.size();
}

/*!\rst
  Fill a vector with all unique random elements.  Random elements are meant
  to potentially be fake process IDs for non-system processes.  I'm assuming
  those probably lie in ``(100, 2^32)`` or so.

  \input
    :thread_ids[1]: allocated vector of thread ids, already set to desired size
  \output
    :thread_ids[1]: overwrites all thread_ids entries with unique values
\endrst*/
void GenerateUniqueRandomVector(std::vector<int> * thread_ids) {
  UniformRandomGenerator uniform_generator(314, 0);  // single instance, so thread_id = 0
  boost::uniform_int<int> uniform_int_distribution(100, std::numeric_limits<int>::max());

  // generate seeds
  for (auto& entry : (*thread_ids)) {
    entry = uniform_int_distribution(uniform_generator.engine);
  }

  while (false == CheckAllElementsUnique(*thread_ids)) {
    // could be smarter about this and only regen non-unique elements
    for (auto& entry : (*thread_ids)) {
      entry = uniform_int_distribution(uniform_generator.engine);
    }
  }

  // re-randomize ordering
  std::shuffle((*thread_ids).begin(), (*thread_ids).end(), uniform_generator.engine);
}

/*!\rst
  Test the features of random number generator containers.

  1. Check that explicitly setting the seed works and sets "last_seed" properly
  2. Verify that the reset functionality properly resets to the last seed
  3. verify that with different input thread ids, container will generate will
     generate a unique seed per thread

  \return
    number of test failures
\endrst*/
template <typename RNGContainer>
OL_WARN_UNUSED_RESULT int RandomNumberGeneratorContainerTestCore() {
  int total_errors = 0;
  int current_errors = 0;

  // set seed manually; verify that last seed is set appropriately
  {
    current_errors = 0;
    const typename RNGContainer::EngineType::result_type seed1 = 31415;
    const typename RNGContainer::EngineType::result_type seed2 = 27182;
    RNGContainer test_rng(seed1);
    if (!CheckIntEquals(test_rng.last_seed(), seed1)) {
      ++current_errors;
    }

    test_rng.SetExplicitSeed(seed2);
    if (!CheckIntEquals(test_rng.last_seed(), seed2)) {
      ++current_errors;
    }
    total_errors += current_errors;
  }

  // verify last seed reset: set seed and save off PRNG state.  Generate a few randoms
  // and check that state changed; then reset to last seed and verify against the original state
  {
    current_errors = 0;
    RNGContainer rng;

    typename RNGContainer::EngineType original_engine(rng.GetEngine());  // copy ctor
    rng.GetEngine().discard(13);
    if (rng.GetEngine() == original_engine) {
      ++current_errors;  // engine state should have changed
    }

    rng.ResetToMostRecentSeed();
    if (rng.GetEngine() != original_engine) {
      ++current_errors;  // engine state should have been reset
    }

    total_errors += current_errors;
  }

  // multi-threaded seeding check
  // build several RNGContainer objects w/different thread ids (ensure ids are all different)
  // verify that objects' last_seed() values are all different
  {
    current_errors = 0;
    const int max_num_threads = 11;

    std::vector<RNGContainer> normal_rng_vec(max_num_threads);
    std::vector<typename RNGContainer::EngineType::result_type> seed_values(max_num_threads);
    std::vector<int> thread_ids(max_num_threads);
    GenerateUniqueRandomVector(&thread_ids);

    for (int i = 0; i < max_num_threads; ++i) {
      normal_rng_vec[i].SetRandomizedSeed(38970, thread_ids[i]);
      seed_values[i] = normal_rng_vec[i].last_seed();
    }

    bool all_seeds_different = CheckAllElementsUnique(seed_values);

    if (all_seeds_different == false) {
      OL_PARTIAL_FAILURE_PRINTF("seed values are not all different!\n");
      for (const auto& value : seed_values) {
        OL_ERROR_PRINTF("%d ", value);
      }
      OL_ERROR_PRINTF("\n");
    }

    if (!all_seeds_different) {
      ++current_errors;
    }
    total_errors += current_errors;
  }

  return total_errors;
}

/*!\rst
  Checks that NormalRNGSimulator is behaving correctly:

  * Tests index increments as expected
  * Tests ResetToMostRecentSeed reset index to 0
  * Tests exception handling when number of queries of random numbers exceeds
  * size of the random table

  \return
    number of test failures: 0 if NormalRNGSimulator behaving correctly
\endrst*/
int NormalRNGSimulatorTest() {
  int total_errors = 0;
  int random_table_size = 500;
  std::vector<double> random_table(random_table_size);
  for (int i = 0; i < random_table_size; ++i) {
    random_table[i] = static_cast<double>(i);
  }
  NormalRNGSimulator rng_simulator(random_table);

  for (int n = 0; n < 40; ++n) {
    int current_idx = rng_simulator.index();
    rng_simulator();
    int next_idx = rng_simulator.index();
    total_errors = ((next_idx - current_idx) == 1) ? total_errors : (total_errors+1);
  }

  rng_simulator.ResetToMostRecentSeed();
  total_errors = (rng_simulator.index() == 0) ? total_errors : (total_errors+1);

  for (int n = 0; n < random_table_size; ++n) {
    rng_simulator();
  }

  ++total_errors;

  try {
    rng_simulator();
  } catch (const InvalidValueException<int>& exception) {
    if ((exception.value() == random_table_size) && (exception.truth() == random_table_size)) {
      --total_errors;
    }
  }

  return total_errors;
}

}  // end unnamed namespace

/*!\rst
  .. Note:: only NormalRNG is meant to be used multi-threaded, so UniformRandomGenerator
      is not tested for generating unique seeds in a multi-threaded environment
\endrst*/
int RandomNumberGeneratorContainerTest() {
  int total_errors = 0;
  int current_errors = 0;

  current_errors = RandomNumberGeneratorContainerTestCore<UniformRandomGenerator>();
  if (current_errors != 0) {
    OL_PARTIAL_FAILURE_PRINTF("UniformRandomGenerator failed with %d errors\n", current_errors);
  } else {
    OL_PARTIAL_SUCCESS_PRINTF("UniformRandomGenerator passed all tests\n");
  }
  total_errors += current_errors;

  current_errors = RandomNumberGeneratorContainerTestCore<NormalRNG>();
  if (current_errors != 0) {
    OL_PARTIAL_FAILURE_PRINTF("NormalRNG failed with %d errors\n", current_errors);
  } else {
    OL_PARTIAL_SUCCESS_PRINTF("NormalRNG passed all tests\n");
  }
  total_errors += current_errors;

  current_errors = NormalRNGSimulatorTest();
  if (current_errors != 0) {
    OL_PARTIAL_FAILURE_PRINTF("NormalRNGSimulator failed with %d errors\n", current_errors);
  } else {
    OL_PARTIAL_SUCCESS_PRINTF("NormalRNGSimulator passed all tests\n");
  }
  total_errors += current_errors;

  return total_errors;
}

}  // end namespace optimal_learning
