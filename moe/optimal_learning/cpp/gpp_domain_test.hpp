/*!
  \file gpp_domain_test.hpp
  \rst
  Tests for gpp_domain.hpp: classes to represent different domains and utilities for operating on them.
\endrst*/

#ifndef MOE_OPTIMAL_LEARNING_CPP_GPP_DOMAIN_TEST_HPP_
#define MOE_OPTIMAL_LEARNING_CPP_GPP_DOMAIN_TEST_HPP_

#include "gpp_common.hpp"

namespace optimal_learning {

/*!\rst
  Tests the functionality of the various domain classes in gpp_domain.hpp;
  e.g., update limiting, in/out tests, random point generation, etc.

  \return
    number of test failures: 0 if all domain functions are working properly
\endrst*/
OL_WARN_UNUSED_RESULT int DomainTests();

}  // end namespace optimal_learning

#endif  // MOE_OPTIMAL_LEARNING_CPP_GPP_DOMAIN_TEST_HPP_
