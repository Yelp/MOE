/*!
  \file gpp_python_model_selection.cpp
  \rst
  This file contains a wrapper that calls all ``C++`` unit tests. The wrapper prints error messages indicating which
  test(s) failed.
\endrst*/
// This include violates the Google Style Guide by placing an "other" system header ahead of C and C++ system headers.  However,
// it needs to be at the top, otherwise compilation fails on some systems with some versions of python: OS X, python 2.7.3.
// Putting this include first prevents pyport from doing something illegal in C++; reference: http://bugs.python.org/issue10910
#include "Python.h"  // NOLINT(build/include)

#include "gpp_python_test.hpp"

#include <boost/python/def.hpp>  // NOLINT(build/include_order)

#include "gpp_common.hpp"
#include "gpp_covariance_test.hpp"
#include "gpp_domain.hpp"
#include "gpp_domain_test.hpp"
#include "gpp_expected_improvement_gpu_test.hpp"
#include "gpp_geometry_test.hpp"
#include "gpp_heuristic_expected_improvement_optimization_test.hpp"
#include "gpp_linear_algebra_test.hpp"
#include "gpp_math_test.hpp"
#include "gpp_model_selection.hpp"
#include "gpp_model_selection_test.hpp"
#include "gpp_optimization_test.hpp"
#include "gpp_random_test.hpp"
#include "gpp_test_utils_test.hpp"

namespace optimal_learning {

namespace {

int RunCppTestsWrapper() {
  int total_errors = 0;
  int error = 0;

  error = TestUtilsTests();
  if (error != 0) {
    OL_FAILURE_PRINTF("test utils\n");
  } else {
    OL_SUCCESS_PRINTF("test utils\n");
  }
  total_errors += error;

  error = RunLinearAlgebraTests();
  if (error != 0) {
    OL_FAILURE_PRINTF("linear algebra tests failed\n");
  } else {
    OL_SUCCESS_PRINTF("linear algebra tests\n");
  }
  total_errors += error;

  error = RunCovarianceTests();
  if (error != 0) {
    OL_FAILURE_PRINTF("covariance ping tests failed\n");
  } else {
    OL_SUCCESS_PRINTF("covariance ping tests\n");
  }
  total_errors += error;

  error = RunGPTests();
  if (error != 0) {
    OL_FAILURE_PRINTF("GP (mean, var, EI) tests failed\n");
  } else {
    OL_SUCCESS_PRINTF("GP (mean, var, EI) tests\n");
  }
  total_errors += error;

  error = RunEIConsistencyTests();
  if (error != 0) {
    OL_FAILURE_PRINTF("analytic, MC EI do not match for 1 potential sample case\n");
  } else {
    OL_SUCCESS_PRINTF("analytic, MC EI match for 1 potential sample case\n");
  }
  total_errors += error;

  error = RunGPUTests();
  if (error != 0) {
    OL_FAILURE_PRINTF("GPU tests failed\n");
  } else {
    OL_SUCCESS_PRINTF("GPU tests passed\n");
  }
  total_errors += error;

  error = RunLogLikelihoodPingTests();
  if (error != 0) {
    OL_FAILURE_PRINTF("LogLikelihood ping tests failed\n");
  } else {
    OL_SUCCESS_PRINTF("LogLikelihood ping tests\n");
  }
  total_errors += error;

  error = RunRandomPointGeneratorTests();
  if (error != 0) {
    OL_FAILURE_PRINTF("various random sampling\n");
  } else {
    OL_SUCCESS_PRINTF("various random sampling\n");
  }
  total_errors += error;

  error = RandomNumberGeneratorContainerTest();
  if (error != 0) {
    OL_FAILURE_PRINTF("random number generator containers\n");
  } else {
    OL_SUCCESS_PRINTF("random number generator containers\n");
  }
  total_errors += error;

  error = DomainTests();
  if (error != 0) {
    OL_FAILURE_PRINTF("domain classes\n");
  } else {
    OL_SUCCESS_PRINTF("domain classes\n");
  }
  total_errors += error;

  error = ClosedIntervalTests();
  if (error != 0) {
    OL_FAILURE_PRINTF("ClosedInterval member functions\n");
  } else {
    OL_SUCCESS_PRINTF("ClosedInterval member functions\n");
  }
  total_errors += error;

  error = GeometryToolsTests();
  if (error != 0) {
    OL_FAILURE_PRINTF("geometry tools\n");
  } else {
    OL_SUCCESS_PRINTF("geometry tools\n");
  }
  total_errors += error;

  error += EstimationPolicyTest();
  if (error != 0) {
    OL_FAILURE_PRINTF("Estimation Policies\n");
  } else {
    OL_SUCCESS_PRINTF("Estimation Policies\n");
  }
  total_errors += error;

  error = RunOptimizationTests();
  if (error != 0) {
    OL_FAILURE_PRINTF("basic optimization tests (simple objectives, exception handling)\n");
  } else {
    OL_SUCCESS_PRINTF("basic optimization tests (simple objectives, exception handling)\n");
  }
  total_errors += error;

  error = HyperparameterLikelihoodOptimizationTest(OptimizerTypes::kGradientDescent, LogLikelihoodTypes::kLogMarginalLikelihood);
  if (error != 0) {
    OL_FAILURE_PRINTF("log likelihood hyperparameter optimization\n");
  } else {
    OL_SUCCESS_PRINTF("log likelihood hyperparameter optimization\n");
  }
  total_errors += error;

  error = HyperparameterLikelihoodOptimizationTest(OptimizerTypes::kGradientDescent, LogLikelihoodTypes::kLeaveOneOutLogLikelihood);
  if (error != 0) {
    OL_FAILURE_PRINTF("LOO likelihood hyperparameter optimization\n");
  } else {
    OL_SUCCESS_PRINTF("LOO likelihood hyperparameter optimization\n");
  }
  total_errors += error;

  error = HyperparameterLikelihoodOptimizationTest(OptimizerTypes::kNewton, LogLikelihoodTypes::kLogMarginalLikelihood);
  if (error != 0) {
    OL_FAILURE_PRINTF("log likelihood hyperparameter newton optimization\n");
  } else {
    OL_SUCCESS_PRINTF("log likelihood hyperparameter newton optimization\n");
  }
  total_errors += error;

  error = EvaluateLogLikelihoodAtPointListTest();
  if (error != 0) {
    OL_FAILURE_PRINTF("log likelihood evaluation at point list\n");
  } else {
    OL_SUCCESS_PRINTF("log likelihood evaluation at point list\n");
  }
  total_errors += error;

  error = EvaluateEIAtPointListTest();
  if (error != 0) {
    OL_FAILURE_PRINTF("EI evaluation at point list\n");
  } else {
    OL_SUCCESS_PRINTF("EI evaluation at point list\n");
  }
  total_errors += error;

  error = MultithreadedEIOptimizationTest(ExpectedImprovementEvaluationMode::kAnalytic);
  if (error != 0) {
    OL_FAILURE_PRINTF("analytic EI Optimization single/multithreaded consistency check\n");
  } else {
    OL_SUCCESS_PRINTF("analytic EI single/multithreaded consistency check\n");
  }
  total_errors += error;

  error = MultithreadedEIOptimizationTest(ExpectedImprovementEvaluationMode::kMonteCarlo);
  if (error != 0) {
    OL_FAILURE_PRINTF("EI Optimization single/multithreaded consistency check\n");
  } else {
    OL_SUCCESS_PRINTF("EI single/multithreaded consistency check\n");
  }
  total_errors += error;

  error += HeuristicExpectedImprovementOptimizationTest();
  if (error != 0) {
    OL_FAILURE_PRINTF("Heuristic EI Optimization\n");
  } else {
    OL_SUCCESS_PRINTF("Heuristic EI Optimization\n");
  }
  total_errors += error;

  error = ExpectedImprovementOptimizationTest(DomainTypes::kTensorProduct, ExpectedImprovementEvaluationMode::kAnalytic);
  if (error != 0) {
    OL_FAILURE_PRINTF("analytic EI optimization\n");
  } else {
    OL_SUCCESS_PRINTF("analytic EI optimization\n");
  }
  total_errors += error;

  error = ExpectedImprovementOptimizationTest(DomainTypes::kTensorProduct, ExpectedImprovementEvaluationMode::kMonteCarlo);
  if (error != 0) {
    OL_FAILURE_PRINTF("monte-carlo EI optimization\n");
  } else {
    OL_SUCCESS_PRINTF("monte-carlo EI optimization\n");
  }
  total_errors += error;

  error = ExpectedImprovementOptimizationMultipleSamplesTest();
  if (error != 0) {
    OL_FAILURE_PRINTF("monte-carlo EI optimization for multiple simultaneous experiments\n");
  } else {
    OL_SUCCESS_PRINTF("monte-carlo EI optimization for multiple simultaneous experiments\n");
  }
  total_errors += error;

  error = ExpectedImprovementOptimizationTest(DomainTypes::kSimplex, ExpectedImprovementEvaluationMode::kAnalytic);
  if (error != 0) {
    OL_FAILURE_PRINTF("analytic simplex EI optimization\n");
  } else {
    OL_SUCCESS_PRINTF("analytic simplex EI optimization\n");
  }
  total_errors += error;

  error = ExpectedImprovementOptimizationTest(DomainTypes::kSimplex, ExpectedImprovementEvaluationMode::kMonteCarlo);
  if (error != 0) {
    OL_FAILURE_PRINTF("monte-carlo simplex EI optimization\n");
  } else {
    OL_SUCCESS_PRINTF("monte-carlo simplex EI optimization\n");
  }
  total_errors += error;

  return total_errors;
}

}  // end unnamed namespace

void ExportCppTestFunctions() {
  boost::python::def("run_cpp_tests", RunCppTestsWrapper, R"%%(
    Runs all current C++ unit tests and reports failures.

    :return: number of test failures. expected to be 0.
    :rtype: int >= 0
    )%%");
}

}  // end namespace optimal_learning
