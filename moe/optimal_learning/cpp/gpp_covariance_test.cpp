/*!
  \file gpp_covariance_test.cpp
  \rst
  This file contains two template classes: one supporting computing covariance and its analytic spatial derivatives, and the other
  for covariance and its analytic hyperparameter derivatives.  Then through a matched pair of template functions, we ping
  the analytic derivatives using finite differences for validation.  (The pinging is done through PingDerivatve() in test_utils.hpp.)

  The Run.*() functions invoke the derivative ping funtions on all of the covariance functions declared in gpp_covariance.hpp.
\endrst*/

#include "gpp_covariance_test.hpp"

#include <algorithm>
#include <string>
#include <stdexcept>
#include <vector>

#include "gpp_common.hpp"
#include "gpp_covariance.hpp"
#include "gpp_exception.hpp"
#include "gpp_logging.hpp"
#include "gpp_random.hpp"
#include "gpp_test_utils.hpp"

namespace optimal_learning {

namespace {

/*!\rst
  Supports evaluating a covariance function, Covariance.Covariance() and its gradient, Covariance.GradCovariance()

  Covariance has the form f = cov(x_d, y_d), where x, y are vectors of size dim.
  The underlying function only works with r = x - y.  In order to reflect this behavior, this class takes
  y_d (reference_point) as a constant input in the constructor.  Then EvaluateAndStoreAnalyticGradient and EvaluateFunction both take
  point_delta as an argument.  x (point) is computed as: x = y + point_delta

  This is required for ping testing: error checking needs to know the magnitude of r (aka point_delta).
  Additionally, since r is the important quantity, different (x,y) pairs that yield the same r are completely identical here.

  The output of coariance is a scalar.

  WARNING: this class is NOT THREAD SAFE.
\endrst*/
template <typename CovarianceClass>
class PingCovarianceSpatialDerivatives final : public PingableMatrixInputVectorOutputInterface {
 public:
  PingCovarianceSpatialDerivatives(double const * restrict lengths, double const * restrict reference_point, double alpha, int dim) OL_NONNULL_POINTERS
      : dim_(dim),
        gradients_already_computed_(false),
        point_(dim),
        point_delta_base_(dim),
        reference_point_(reference_point, reference_point + dim),
        grad_covariance_(dim),
        covariance_(dim, alpha, lengths) {
  }

  virtual void GetInputSizes(int * num_rows, int * num_cols) const noexcept override OL_NONNULL_POINTERS {
    *num_rows = dim_;
    *num_cols = 1;
  }

  virtual int GetGradientsSize() const noexcept override OL_WARN_UNUSED_RESULT {
    return dim_*GetOutputSize();
  }

  virtual int GetOutputSize() const noexcept override OL_WARN_UNUSED_RESULT {
    return 1;
  }

  virtual void EvaluateAndStoreAnalyticGradient(double const * restrict point_delta, double * restrict gradients) noexcept override OL_NONNULL_POINTERS_LIST(2) {
    if (gradients_already_computed_ == true) {
      OL_WARNING_PRINTF("WARNING: grad_covariance data already set.  Overwriting...\n");
    }
    gradients_already_computed_ = true;

    std::copy(point_delta, point_delta + dim_, point_delta_base_.begin());
    ShiftScaleReferencePoint(point_delta);
    covariance_.GradCovariance(point_.data(), reference_point_.data(), grad_covariance_.data());

    if (gradients != nullptr) {
      std::copy(grad_covariance_.begin(), grad_covariance_.end(), gradients);
    }
  }

  int CheckSymmetry() const OL_WARN_UNUSED_RESULT {
    if (gradients_already_computed_ == false) {
      OL_THROW_EXCEPTION(OptimalLearningException, "PingCovarianceSpatialDerivatives::CheckSymmetry() called BEFORE EvaluateAndStoreAnalyticGradient. NO DATA!");
    }
    ShiftScaleReferencePoint(point_delta_base_.data());
    double covariance_val = covariance_.Covariance(point_.data(), reference_point_.data());
    double covariance_val_transpose = covariance_.Covariance(reference_point_.data(), point_.data());

    int total_errors = 0;
    if (!CheckDoubleWithinRelative(covariance_val, covariance_val_transpose, 0.0)) {
      ++total_errors;
    }
    return total_errors;
  }

  virtual double GetAnalyticGradient(int row_index, int OL_UNUSED(column_index), int OL_UNUSED(output_index)) const override OL_WARN_UNUSED_RESULT {
    if (gradients_already_computed_ == false) {
      OL_THROW_EXCEPTION(OptimalLearningException, "PingCovarianceSpatialDerivatives::GetAnalyticGradient() called BEFORE EvaluateAndStoreAnalyticGradient. NO DATA!");
    }

    return grad_covariance_[row_index];
  }

  virtual void EvaluateFunction(double const * restrict point_delta, double * restrict function_values) const noexcept override OL_NONNULL_POINTERS {
    ShiftScaleReferencePoint(point_delta);
    *function_values = covariance_.Covariance(point_.data(), reference_point_.data());
  }

  OL_DISALLOW_DEFAULT_AND_COPY_AND_ASSIGN(PingCovarianceSpatialDerivatives);

 private:
  OL_NONNULL_POINTERS void ShiftScaleReferencePoint(double const * restrict point_delta) const noexcept {
    // point is just temporary storage.  Its state is irrelevant to the const-ness of this class
    for (int i = 0; i < dim_; ++i) {
      point_[i] = point_delta[i] + reference_point_[i]/3.0;
    }
  }

  int dim_;
  bool gradients_already_computed_;

  mutable std::vector<double> point_;
  std::vector<double> point_delta_base_;
  std::vector<double> reference_point_;
  std::vector<double> grad_covariance_;

  CovarianceClass covariance_;
};

/*!\rst
  Supports evaluating a covariance function, covariance.Covariance() and its hyperparameter gradient,
  covariance.HyperparameterGradCovariance()

  Since we're differentiating against hyperparameters, this class saves off the base set of hyperparameters, living in the
  class variable covariance (saved off by EvaluateAndStoreAnalyticGradient).  Then evaluation at different hyperparameters
  (via EvaluateFunction) is supported by building additional local covariance objects.

  The output of coariance is a scalar.
\endrst*/
template <typename CovarianceClass>
class PingGradCovarianceHyperparameters final : public PingableMatrixInputVectorOutputInterface {
 public:
  PingGradCovarianceHyperparameters(double const * restrict point1, double const * restrict point2, int dim) OL_NONNULL_POINTERS
      : dim_(dim),
        num_hyperparameters_(0),
        gradients_already_computed_(false),
        point1_(point1, point1 + dim),
        point2_(point2, point2 + dim),
        grad_hyperparameter_covariance_(dim+1),
        covariance_(dim, 1.0, 1.0) {
    num_hyperparameters_ = covariance_.GetNumberOfHyperparameters();
    grad_hyperparameter_covariance_.resize(num_hyperparameters_);
  }

  virtual void GetInputSizes(int * num_rows, int * num_cols) const noexcept override OL_NONNULL_POINTERS {
    *num_rows = num_hyperparameters_;
    *num_cols = 1;
  }

  virtual int GetGradientsSize() const noexcept override OL_WARN_UNUSED_RESULT {
    return num_hyperparameters_*GetOutputSize();
  }

  virtual int GetOutputSize() const noexcept override OL_WARN_UNUSED_RESULT {
    return 1;
  }

  virtual void EvaluateAndStoreAnalyticGradient(double const * restrict hyperparameters, double * restrict gradients) noexcept override OL_NONNULL_POINTERS_LIST(2) {
    if (gradients_already_computed_ == true) {
      OL_WARNING_PRINTF("WARNING: grad_covariance data already set.  Overwriting...\n");
    }
    gradients_already_computed_ = true;

    covariance_.SetHyperparameters(hyperparameters);
    covariance_.HyperparameterGradCovariance(point1_.data(), point2_.data(), grad_hyperparameter_covariance_.data());

    if (gradients != nullptr) {
      std::copy(grad_hyperparameter_covariance_.begin(), grad_hyperparameter_covariance_.end(), gradients);
    }
  }

  int CheckSymmetry() const OL_WARN_UNUSED_RESULT {
    if (gradients_already_computed_ == false) {
      OL_THROW_EXCEPTION(OptimalLearningException, "PingGradCovarianceHyperparameters::CheckSymmetry() called BEFORE EvaluateAndStoreAnalyticGradient. NO DATA!");
    }
    std::vector<double> gradients_transpose(num_hyperparameters_);
    covariance_.HyperparameterGradCovariance(point2_.data(), point1_.data(), gradients_transpose.data());

    int total_errors = 0;
    for (int i = 0; i < num_hyperparameters_; ++i) {
      if (!CheckDoubleWithinRelative(grad_hyperparameter_covariance_[i], gradients_transpose[i], 0.0)) {
        ++total_errors;
      }
    }
    return total_errors;
  }

  virtual double GetAnalyticGradient(int row_index, int OL_UNUSED(column_index), int OL_UNUSED(output_index)) const override OL_WARN_UNUSED_RESULT {
    if (gradients_already_computed_ == false) {
      OL_THROW_EXCEPTION(OptimalLearningException, "PingGradCovarianceHyperparameters::GetAnalyticGradient() called BEFORE EvaluateAndStoreAnalyticGradient. NO DATA!");
    }

    return grad_hyperparameter_covariance_[row_index];
  }

  virtual void EvaluateFunction(double const * restrict hyperparameters, double * restrict function_values) const noexcept override OL_NONNULL_POINTERS {
    CovarianceClass covariance_local(dim_, hyperparameters[0], hyperparameters + 1);

    *function_values = covariance_local.Covariance(point1_.data(), point2_.data());
  }

  OL_DISALLOW_DEFAULT_AND_COPY_AND_ASSIGN(PingGradCovarianceHyperparameters);

 private:
  int dim_;
  int num_hyperparameters_;
  bool gradients_already_computed_;

  std::vector<double> point1_;
  std::vector<double> point2_;
  std::vector<double> grad_hyperparameter_covariance_;

  CovarianceClass covariance_;
};

template <typename CovarianceClass>
class PingHessianCovarianceHyperparameters final : public PingableMatrixInputVectorOutputInterface {
 public:
  PingHessianCovarianceHyperparameters(double const * restrict point1, double const * restrict point2, int dim) OL_NONNULL_POINTERS
      : dim_(dim),
        num_hyperparameters_(0),
        gradients_already_computed_(false),
        point1_(point1, point1 + dim),
        point2_(point2, point2 + dim),
        hessian_hyperparameter_covariance_(Square(dim+1)),
        covariance_(dim, 1.0, 1.0) {
    num_hyperparameters_ = covariance_.GetNumberOfHyperparameters();
    hessian_hyperparameter_covariance_.resize(Square(num_hyperparameters_));
  }

  virtual void GetInputSizes(int * num_rows, int * num_cols) const noexcept override OL_NONNULL_POINTERS {
    *num_rows = num_hyperparameters_;
    *num_cols = 1;
  }

  virtual int GetGradientsSize() const noexcept override OL_WARN_UNUSED_RESULT {
    return num_hyperparameters_*GetOutputSize();
  }

  virtual int GetOutputSize() const noexcept override OL_WARN_UNUSED_RESULT {
    return num_hyperparameters_;
  }

  virtual void EvaluateAndStoreAnalyticGradient(double const * restrict hyperparameters, double * restrict gradients) noexcept override OL_NONNULL_POINTERS_LIST(2) {
    if (gradients_already_computed_ == true) {
      OL_WARNING_PRINTF("WARNING: grad_covariance data already set.  Overwriting...\n");
    }
    gradients_already_computed_ = true;

    covariance_.SetHyperparameters(hyperparameters);
    covariance_.HyperparameterHessianCovariance(point1_.data(), point2_.data(), hessian_hyperparameter_covariance_.data());

    if (gradients != nullptr) {
      std::copy(hessian_hyperparameter_covariance_.begin(), hessian_hyperparameter_covariance_.end(), gradients);
    }
  }

  int CheckSymmetry() const OL_WARN_UNUSED_RESULT {
    if (gradients_already_computed_ == false) {
      OL_THROW_EXCEPTION(OptimalLearningException, "PingHessianCovarianceHyperparameters::CheckSymmetry() called BEFORE EvaluateAndStoreAnalyticGradient. NO DATA!");
    }
    // check that output is symmetric to ordering of points
    // this is equivalent to checking that the matrix is exactly symmetric
    std::vector<double> gradients_transpose(Square(num_hyperparameters_));
    covariance_.HyperparameterHessianCovariance(point2_.data(), point1_.data(), gradients_transpose.data());

    int total_errors = 0;
    for (int i = 0; i < Square(num_hyperparameters_); ++i) {
      if (!CheckDoubleWithinRelative(hessian_hyperparameter_covariance_[i], gradients_transpose[i], 0.0)) {
        ++total_errors;
      }
    }

    // check that output hessian matrix is symmetric
    for (int j = 0; j < num_hyperparameters_; ++j) {
      for (int i = 0; i < num_hyperparameters_; ++i) {
        if (!CheckDoubleWithinRelative(hessian_hyperparameter_covariance_[j*num_hyperparameters_ +i], hessian_hyperparameter_covariance_[i*num_hyperparameters_ + j], 0.0)) {
          ++total_errors;
        }
      }
    }

    return total_errors;
  }

  virtual double GetAnalyticGradient(int row_index, int OL_UNUSED(column_index), int output_index) const override OL_WARN_UNUSED_RESULT {
    if (gradients_already_computed_ == false) {
      OL_THROW_EXCEPTION(OptimalLearningException, "PingHessianCovarianceHyperparameters::GetAnalyticGradient() called BEFORE EvaluateAndStoreAnalyticGradient. NO DATA!");
    }

    return hessian_hyperparameter_covariance_[row_index*num_hyperparameters_ + output_index];
  }

  virtual void EvaluateFunction(double const * restrict hyperparameters, double * restrict function_values) const noexcept override OL_NONNULL_POINTERS {
    CovarianceClass covariance_local(dim_, hyperparameters[0], hyperparameters + 1);

    covariance_local.HyperparameterGradCovariance(point1_.data(), point2_.data(), function_values);
  }

  OL_DISALLOW_DEFAULT_AND_COPY_AND_ASSIGN(PingHessianCovarianceHyperparameters);

 private:
  int dim_;
  int num_hyperparameters_;
  bool gradients_already_computed_;

  std::vector<double> point1_;
  std::vector<double> point2_;
  std::vector<double> hessian_hyperparameter_covariance_;

  CovarianceClass covariance_;
};

template <typename PingCovarianceClass>
OL_NONNULL_POINTERS OL_WARN_UNUSED_RESULT int PingCovarianceHyperparameterDerivativesTest(char const * class_name, int num_hyperparameters, double epsilon[2], double tolerance_fine, double tolerance_coarse, double input_output_ratio) {
  const int dim = 3;
  int errors_this_iteration = 0;
  int total_errors = 0;

  std::vector<double> hyperparameters(num_hyperparameters);

  int num_being_sampled = 0;
  int num_to_sample = 1;
  int num_sampled = 1;

  MockExpectedImprovementEnvironment EI_environment;
  UniformRandomGenerator uniform_generator(2718);
  boost::uniform_real<double> uniform_double(3.0, 5.0);

  for (int i = 0; i < 10; ++i) {
    EI_environment.Initialize(dim, num_to_sample, num_being_sampled, num_sampled);

    for (int j = 0; j < num_hyperparameters; ++j) {
      hyperparameters[j] = uniform_double(uniform_generator.engine);
    }

    PingCovarianceClass covariance_evaluator(EI_environment.points_to_sample(), EI_environment.points_sampled(), EI_environment.dim);
    covariance_evaluator.EvaluateAndStoreAnalyticGradient(hyperparameters.data(), nullptr);
    errors_this_iteration = covariance_evaluator.CheckSymmetry();
    if (errors_this_iteration != 0) {
      OL_PARTIAL_FAILURE_PRINTF("hyperparameter gradients from %s are NOT symmetric! %d fails\n", class_name, errors_this_iteration);
    }

    errors_this_iteration += PingDerivative(covariance_evaluator, hyperparameters.data(), epsilon, tolerance_fine, tolerance_coarse, input_output_ratio);

    if (errors_this_iteration != 0) {
      OL_PARTIAL_FAILURE_PRINTF("on iteration %d\n", i);
    }
    total_errors += errors_this_iteration;
  }

  return total_errors;
}

template <typename PingCovarianceClass>
OL_NONNULL_POINTERS OL_WARN_UNUSED_RESULT int PingCovarianceHyperparameterGradientsTest(char const * class_name, int num_hyperparameters, double epsilon[2], double tolerance_fine, double tolerance_coarse, double input_output_ratio) {
  const int dim = 3;
  int errors_this_iteration = 0;
  int total_errors = 0;
  std::vector<double> hyperparameters(num_hyperparameters);

  UniformRandomGenerator uniform_generator(31415);
  boost::uniform_real<double> uniform_double(3.0, 5.0);

  {
    // check that at r = x1 - x2 = 0, the gradient wrt alpha is 1.0 and wrt length scales is 0.0
    double point3[dim] = {0.0};
    double point4[dim] = {0.0};
    for (int j = 0; j < num_hyperparameters; ++j) {
      hyperparameters[j] = uniform_double(uniform_generator.engine);
    }
    PingCovarianceClass covariance_evaluator2(point3, point4, dim);
    covariance_evaluator2.EvaluateAndStoreAnalyticGradient(hyperparameters.data(), nullptr);
    errors_this_iteration = PingDerivative(covariance_evaluator2, hyperparameters.data(), epsilon, tolerance_fine, tolerance_coarse, input_output_ratio);

    // wrt alpha
    if (covariance_evaluator2.GetAnalyticGradient(0, 0, 0) != 1.0) {
      errors_this_iteration += 1;
    }

    // wrt length scales
    for (int j = 1; j < covariance_evaluator2.GetGradientsSize(); ++j) {
      if (covariance_evaluator2.GetAnalyticGradient(j, 0, 0) != 0.0) {
        errors_this_iteration += 1;
      }
    }

    if (errors_this_iteration != 0) {
      OL_PARTIAL_FAILURE_PRINTF("on hyperparameter zero test case");
    }
    total_errors += errors_this_iteration;
  }

  total_errors += PingCovarianceHyperparameterDerivativesTest<PingCovarianceClass>(class_name, num_hyperparameters, epsilon, tolerance_fine, tolerance_coarse, input_output_ratio);

  if (total_errors != 0) {
    OL_PARTIAL_FAILURE_PRINTF("%s covariance hyperparameter gradient pings failed with %d errors\n", class_name, total_errors);
  } else {
    OL_PARTIAL_SUCCESS_PRINTF("%s covariance hyperparameter gradient pings passed\n", class_name);
  }

  return total_errors;
}

template <typename PingCovarianceClass>
OL_NONNULL_POINTERS OL_WARN_UNUSED_RESULT int PingCovarianceHyperparameterHessianTest(char const * class_name, int num_hyperparameters, double epsilon[2], double tolerance_fine, double tolerance_coarse, double input_output_ratio) {
  const int dim = 3;
  int errors_this_iteration = 0;
  int total_errors = 0;

  std::vector<double> hyperparameters(num_hyperparameters);

  UniformRandomGenerator uniform_generator(31415);
  boost::uniform_real<double> uniform_double(3.0, 5.0);

  {
    // check that at r = x1 - x2 = 0, the gradient wrt alpha is 1.0 and wrt length scales is 0.0
    double point3[dim] = {0.0};
    double point4[dim] = {0.0};
    for (int j = 0; j < num_hyperparameters; ++j) {
      hyperparameters[j] = uniform_double(uniform_generator.engine);
    }
    PingCovarianceClass covariance_evaluator2(point3, point4, dim);
    covariance_evaluator2.EvaluateAndStoreAnalyticGradient(hyperparameters.data(), nullptr);
    errors_this_iteration = PingDerivative(covariance_evaluator2, hyperparameters.data(), epsilon, tolerance_fine, tolerance_coarse, input_output_ratio);

    // second derivatives are all exactly 0 when r = 0
    for (int j = 0; j < covariance_evaluator2.GetGradientsSize(); ++j) {
      if (covariance_evaluator2.GetAnalyticGradient(0, 0, j) != 0.0) {
        errors_this_iteration += 1;
      }
    }

    if (errors_this_iteration != 0) {
      OL_PARTIAL_FAILURE_PRINTF("on hyperparameter zero test case");
    }
    total_errors += errors_this_iteration;
  }

  total_errors += PingCovarianceHyperparameterDerivativesTest<PingCovarianceClass>(class_name, num_hyperparameters, epsilon, tolerance_fine, tolerance_coarse, input_output_ratio);

  if (total_errors != 0) {
    OL_PARTIAL_FAILURE_PRINTF("%s covariance hyperparameter hessian pings failed with %d errors\n", class_name, total_errors);
  } else {
    OL_PARTIAL_SUCCESS_PRINTF("%s covariance hyperparameter hessian pings passed\n", class_name);
  }

  return total_errors;
}

/*!\rst
  Pings the gradient of covariance functions to check their validity.
  Test cases include a couple of simple hand-checked cases as well as a run
  of 50 randomly generated tests.

  \return
    number of pings that failed
\endrst*/
template <typename PingCovarianceClass>
OL_NONNULL_POINTERS OL_WARN_UNUSED_RESULT int PingCovarianceSpatialDerivativesTest(char const * class_name, double epsilon[2], double tolerance_fine, double tolerance_coarse, double input_output_ratio) {
  const  int dim = 3;
  double point1[dim] = {0.2, -1.7, 0.91};
  double point2[dim] = {-2.1, 0.32, 1.12};
  double point3[dim] = {0.0};
  double point4[dim] = {0.0};
  int errors_this_iteration;
  int total_errors = 0;

  std::vector<double> lengths(dim);
  double alpha = 2.80723;

  UniformRandomGenerator uniform_generator(31415);
  boost::uniform_real<double> uniform_double(0.5, 2.5);

  {
    // hand-checked test-case
    for (int j = 0; j < dim; ++j) {
      lengths[j] = uniform_double(uniform_generator.engine);
    }
    PingCovarianceClass covariance_evaluator1(lengths.data(), point2, alpha, dim);
    covariance_evaluator1.EvaluateAndStoreAnalyticGradient(point1, nullptr);
    errors_this_iteration = PingDerivative(covariance_evaluator1, point1, epsilon, tolerance_fine, tolerance_coarse, input_output_ratio);

    if (errors_this_iteration != 0) {
      OL_PARTIAL_FAILURE_PRINTF("on hand-checked case");
    }
    total_errors += errors_this_iteration;
  }

  {
    // check that at r = x1 - x2 = 0, the gradient is precisely 0
    for (int j = 0; j < dim; ++j) {
      lengths[j] = uniform_double(uniform_generator.engine);
    }
    PingCovarianceClass covariance_evaluator2(lengths.data(), point4, alpha, dim);
    covariance_evaluator2.EvaluateAndStoreAnalyticGradient(point3, nullptr);

    errors_this_iteration = PingDerivative(covariance_evaluator2, point3, epsilon, tolerance_fine, tolerance_coarse, input_output_ratio);
    for (int j = 0; j < dim; ++j) {
      if (covariance_evaluator2.GetAnalyticGradient(j, 0, 0) != 0.0) {
        errors_this_iteration += 1;
      }
    }

    if (errors_this_iteration != 0) {
      OL_PARTIAL_FAILURE_PRINTF("on zero test case");
    }
    total_errors += errors_this_iteration;
  }

  int num_being_sampled = 0;
  int num_to_sample = 1;
  int num_sampled = 1;

  MockExpectedImprovementEnvironment EI_environment;

  for (int i = 0; i < 50; ++i) {
    EI_environment.Initialize(dim, num_to_sample, num_being_sampled, num_sampled);

    for (int j = 0; j < dim; ++j) {
      lengths[j] = uniform_double(uniform_generator.engine);
    }

    PingCovarianceClass covariance_evaluator(lengths.data(), EI_environment.points_sampled(), alpha, EI_environment.dim);
    covariance_evaluator.EvaluateAndStoreAnalyticGradient(EI_environment.points_to_sample(), nullptr);

    errors_this_iteration = covariance_evaluator.CheckSymmetry();
    if (errors_this_iteration != 0) {
      OL_PARTIAL_FAILURE_PRINTF("hyperparameter gradients from %s are NOT symmetric! %d fails\n", class_name, errors_this_iteration);
    }

    errors_this_iteration += PingDerivative(covariance_evaluator, EI_environment.points_to_sample(), epsilon, tolerance_fine, tolerance_coarse, input_output_ratio);

    if (errors_this_iteration != 0) {
      OL_PARTIAL_FAILURE_PRINTF("on iteration %d\n", i);
    }
    total_errors += errors_this_iteration;
  }

  if (total_errors != 0) {
    OL_PARTIAL_FAILURE_PRINTF("%s covariance pings failed with %d errors\n", class_name, total_errors);
  } else {
    OL_PARTIAL_SUCCESS_PRINTF("%s  covariance pings passed\n", class_name);
  }

  return total_errors;
}

/*!\rst
  Test that gradient wrt spatial coordinates of various covariance functions are working.

  Uses separate scopes to prevent accidental misuse of variable names.

  \return
    Number of covariance functions where spatial gradient pings failed
\endrst*/
OL_WARN_UNUSED_RESULT int RunCovarianceSpatialDerivativesTests() {
  int total_errors = 0;
  int current_errors = 0;

  {
    double epsilon_square_exponential[2] = {1.0e-2, 1.0e-3};
    current_errors = PingCovarianceSpatialDerivativesTest<PingCovarianceSpatialDerivatives<SquareExponential> >("Square Exponential", epsilon_square_exponential, 4.0e-3, 1.0e-2, 1.0e-18);
    if (current_errors != 0) {
      OL_PARTIAL_FAILURE_PRINTF("pinging sqexp covariance failed with %d errors\n", current_errors);
    }
    total_errors += current_errors;
  }

  {
    double epsilon_matern_nu_1p5[2] = {1.0e-2, 1.0e-3};
    current_errors = PingCovarianceSpatialDerivativesTest<PingCovarianceSpatialDerivatives<MaternNu1p5> >("Matern nu=1.5", epsilon_matern_nu_1p5, 4.0e-3, 1.0e-2, 1.0e-18);
    if (current_errors != 0) {
      OL_PARTIAL_FAILURE_PRINTF("pinging matern 1.5 covariance failed with %d errors\n", current_errors);
    }
    total_errors += current_errors;
  }

  {
    double epsilon_matern_nu_2p5[2] = {1.0e-2, 1.0e-3};
    current_errors = PingCovarianceSpatialDerivativesTest<PingCovarianceSpatialDerivatives<MaternNu2p5> >("Matern nu=2.5", epsilon_matern_nu_2p5, 4.0e-3, 1.0e-2, 1.0e-18);
    if (current_errors != 0) {
      OL_PARTIAL_FAILURE_PRINTF("pinging matern 2.5 covariance failed with %d errors\n", current_errors);
    }
    total_errors += current_errors;
  }

  return total_errors;
}

/*!\rst
  Test that gradient and hessian wrt hyperparameters of various covariance functions are working.

  Uses separate scopes to prevent accidental misuse of variable names.

  \return
    Number of covariance functions where hyperparameter gradients/hessian pings failed
\endrst*/
OL_WARN_UNUSED_RESULT int RunCovarianceHyperparameterDerivativesTests() noexcept {
  int total_errors = 0;
  int current_errors = 0;

  {
    double epsilon_square_exponential_hyperparameters[2] = {5.0e-2, 1.0e-2};
    current_errors = PingCovarianceHyperparameterGradientsTest<PingGradCovarianceHyperparameters<SquareExponentialSingleLength> >("Square Exponential Single Length", 2, epsilon_square_exponential_hyperparameters, 4.0e-3, 4.0e-3, 5.0e-15);
    if (current_errors != 0) {
      OL_PARTIAL_FAILURE_PRINTF("pinging sqexp covariance single length hyperparameters failed with %d errors\n", current_errors);
    }
    total_errors += current_errors;
  }

  {
    double epsilon_square_exponential_hyperparameters[2] = {9.0e-3, 2.0e-3};
    current_errors = PingCovarianceHyperparameterGradientsTest<PingGradCovarianceHyperparameters<SquareExponential> >("Square Exponential", 4, epsilon_square_exponential_hyperparameters, 4.0e-3, 5.0e-3, 3.0e-14);
    if (current_errors != 0) {
      OL_PARTIAL_FAILURE_PRINTF("pinging sqexp covariance hyperparameters failed with %d errors\n", current_errors);
    }
    total_errors += current_errors;
  }

  {
    double epsilon_matern_nu_1p5_hyperparameters[2] = {3.0e-2, 4.0e-3};
    current_errors = PingCovarianceHyperparameterGradientsTest<PingGradCovarianceHyperparameters<MaternNu1p5> >("Matern nu=1.5", 4, epsilon_matern_nu_1p5_hyperparameters, 4.0e-3, 5.0e-3, 3.0e-14);
    if (current_errors != 0) {
      OL_PARTIAL_FAILURE_PRINTF("pinging matern nu=1.5 covariance hyperparameters failed with %d errors\n", current_errors);
    }
    total_errors += current_errors;
  }

  {
    double epsilon_matern_nu_2p5_hyperparameters[2] = {5.0e-2, 1.0e-2};
    current_errors = PingCovarianceHyperparameterGradientsTest<PingGradCovarianceHyperparameters<MaternNu2p5> >("Matern nu=2.5", 4, epsilon_matern_nu_2p5_hyperparameters, 3.0e-3, 4.0e-3, 3.0e-14);
    if (current_errors != 0) {
      OL_PARTIAL_FAILURE_PRINTF("pinging matern nu=2.5 covariance hyperparameters failed with %d errors\n", current_errors);
    }
    total_errors += current_errors;
  }

  {
    double epsilon_square_exponential_hyperparameters[2] = {9.0e-3, 2.0e-3};
    current_errors = PingCovarianceHyperparameterHessianTest<PingHessianCovarianceHyperparameters<SquareExponential> >("Square Exponential", 4, epsilon_square_exponential_hyperparameters, 4.0e-3, 5.0e-3, 8.0e-15);
    if (current_errors != 0) {
      OL_PARTIAL_FAILURE_PRINTF("pinging sqexp hessian covariance hyperparameters failed with %d errors\n", current_errors);
    }
    total_errors += current_errors;
  }

  {
    double epsilon_square_exponential_hyperparameters[2] = {5.0e-2, 8.0e-3};
    current_errors = PingCovarianceHyperparameterHessianTest<PingHessianCovarianceHyperparameters<SquareExponentialSingleLength> >("Square Exponential Single Length", 4, epsilon_square_exponential_hyperparameters, 4.0e-3, 5.0e-3, 5.0e-15);
    if (current_errors != 0) {
      OL_PARTIAL_FAILURE_PRINTF("pinging sqexp single length hessian covariance hyperparameters failed with %d errors\n", current_errors);
    }
    total_errors += current_errors;
  }

  {
    double epsilon_matern_nu_1p5_hyperparameters[2] = {4.5e-2, 8.0e-3};
    current_errors = PingCovarianceHyperparameterHessianTest<PingHessianCovarianceHyperparameters<MaternNu1p5> >("Matern nu=1.5", 4, epsilon_matern_nu_1p5_hyperparameters, 5.0e-3, 6.0e-3, 5.0e-15);
    if (current_errors != 0) {
      OL_PARTIAL_FAILURE_PRINTF("pinging matern nu=1.5 hessian covariance hyperparameters failed with %d errors\n", current_errors);
    }
    total_errors += current_errors;
  }

  {
    double epsilon_matern_nu_2p5_hyperparameters[2] = {2.5e-2, 4.0e-3};
    current_errors = PingCovarianceHyperparameterHessianTest<PingHessianCovarianceHyperparameters<MaternNu2p5> >("Matern nu=2.5", 4, epsilon_matern_nu_2p5_hyperparameters, 4.0e-3, 5.0e-3, 5.0e-15);
    if (current_errors != 0) {
      OL_PARTIAL_FAILURE_PRINTF("pinging matern nu=2.5 hessian covariance hyperparameters failed with %d errors\n", current_errors);
    }
    total_errors += current_errors;
  }

  return total_errors;
}

}  // end unnamed namespace

int RunCovarianceTests() {
  int total_errors = 0;
  int current_errors = 0;

  current_errors = RunCovarianceSpatialDerivativesTests();
  if (current_errors != 0) {
    OL_PARTIAL_FAILURE_PRINTF("Pinging covariance spatial derivatives failed with %d errors\n", current_errors);
  }
  total_errors += current_errors;

  current_errors = RunCovarianceHyperparameterDerivativesTests();
  if (current_errors != 0) {
    OL_PARTIAL_FAILURE_PRINTF("Pinging covariance hyperparameter derivatives failed with %d errors\n", current_errors);
  }
  total_errors += current_errors;

  return total_errors;
}

}  // end namespace optimal_learning
