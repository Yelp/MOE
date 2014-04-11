// gpp_heuristic_expected_improvement_optimization_test.hpp
/*
  Functions for testing gpp_heuristic_expected_improvement_optimization.cpp's functionality.
  These tests are a combination of unit and integration tests for heuristic optimization methods for
  expected improvement (e.g., Constant Liar, Kriging Believer).

  These heuristic methods are fairly simple compared to their optimal counterparts in gpp_math, so
  the tests generally validate output consistency and any relevant intermediate assumptions.
*/

#ifndef OPTIMAL_LEARNING_EPI_SRC_CPP_GPP_HEURISTIC_EXPECTED_IMPROVEMENT_OPTIMIZATION_TEST_HPP_
#define OPTIMAL_LEARNING_EPI_SRC_CPP_GPP_HEURISTIC_EXPECTED_IMPROVEMENT_OPTIMIZATION_TEST_HPP_

#include "gpp_common.hpp"

namespace optimal_learning {

/*
  Checks that the subclasses of ObjectiveEstimationPolicyInterface declared in
  gpp_heuristic_expected_improvement_optimization.hpp are working correctly. Right now, these are:
  1) Constant Liar
  2) Kriging Believer
  We set up contrived environments where the outputs of these policies is known exactly.

  RETURNS:
  number of test failures: 0 if estimation policies are working properly
*/
OL_WARN_UNUSED_RESULT int EstimationPolicyTest();

/*
  Checks that ComputeHeuristicSetOfPointsToSample() works on a tensor product domain using both
  ConstantLiarEstimationPolicy and KrigingBelieverEstimationPolicy estimation policies.
  This test assumes the the code tested in:
  1) ExpectedImprovementOptimizationTest(DomainTypes::kTensorProduct, ExpectedImprovementEvaluationMode::kAnalytic)
  2) EstimationPolicyTest()
  is working.

  This test checks the generation of multiple, simultaneous experimental points to sample using
  various objective function estimation heuristics; i.e., no monte-carlo needed.

  RETURNS:
  number of test failures: 0 if heuristic EI optimization is working properly
*/
OL_WARN_UNUSED_RESULT int HeuristicExpectedImprovementOptimizationTest();

}  // end namespace optimal_learning

#endif  // OPTIMAL_LEARNING_EPI_SRC_CPP_GPP_HEURISTIC_EXPECTED_IMPROVEMENT_OPTIMIZATION_TEST_HPP_
