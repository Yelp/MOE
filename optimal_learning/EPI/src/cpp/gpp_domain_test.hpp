// gpp_domain_test.hpp
/*
  Tests for gpp_domain.hpp: classes to represent different domains and utilities for operating on them.
*/

#ifndef OPTIMAL_LEARNING_EPI_SRC_CPP_GPP_DOMAIN_TEST_HPP_
#define OPTIMAL_LEARNING_EPI_SRC_CPP_GPP_DOMAIN_TEST_HPP_

#include "gpp_common.hpp"

namespace optimal_learning {

/*
  Tests the functionality of the various domain classes in gpp_domain.hpp;
  e.g., update limiting, in/out tests, random point generation, etc.

  RETURNS:
  number of test failures: 0 if all domain functions are working properly
*/
OL_WARN_UNUSED_RESULT int DomainTests();

}  // end namespace optimal_learning

#endif  // OPTIMAL_LEARNING_EPI_SRC_CPP_GPP_DOMAIN_TEST_HPP_
