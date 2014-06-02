/*!
  \file gpp_random_test.hpp
  \rst
  Tests for gpp_random.hpp: PRNG container classes and point sampling.

  The tests for domain sampling test that the sampled point sets have the properties promised by the algorithms that
  computed them (e.g., RandomPointInDomain only promises point in the domain whereas LatinHypercube has a check-able
  guarantee of more evenly distributed samples).

  The PRNG container tests verify that the seeding functions properly generate unique seeds in multithreaded environments,
  "with high probability."
\endrst*/

#ifndef MOE_OPTIMAL_LEARNING_CPP_GPP_RANDOM_TEST_HPP_
#define MOE_OPTIMAL_LEARNING_CPP_GPP_RANDOM_TEST_HPP_

#include "gpp_common.hpp"

namespace optimal_learning {

/*!\rst
  Tests that the various point generator functions available in gpp_random.hpp are working; e.g.,

  * ComputeRandomPointInDomain()
  * ComputeLatinHypercubePointsInDomain()
  * ComputeUniformPointsInUnitSimplex()

  \return
    number of test failures: 0 if all point generator functions are working properly
\endrst*/
int RunRandomPointGeneratorTests() OL_WARN_UNUSED_RESULT;

/*!\rst
  Checks that PRNG container is behaving correctly:

  * Tests manual seed setting
  * Tests last_seed and reset
  * Tests that in multithreaded environemnts, each thread gets a different seed

  \return
    number of test failures: 0 if PRNG containers are behaving correctly
\endrst*/
int RandomNumberGeneratorContainerTest() OL_WARN_UNUSED_RESULT;

}  // end namespace optimal_learning

#endif  // MOE_OPTIMAL_LEARNING_CPP_GPP_RANDOM_TEST_HPP_
