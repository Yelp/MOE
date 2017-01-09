/*!
  \file gpp_math_test.cpp
  \rst
  Routines to test the functions in gpp_math.cpp.

  The tests verify GaussianProcess, ExpectedImprovementEvaluator (+OnePotentialSample), and EI optimization from gpp_math.cpp.

  1. Ping testing (verifying analytic gradient computation against finite difference approximations)

     a. Following gpp_covariance_test.cpp, we define classes (PingGPMean + other GP ping, PingExpectedImprovement) for
        evaluating those functions + their spatial gradients.

        Some Pingable classes for GP functions are less general than their gpp_covariance_test or
        gpp_model_selection_test counterparts, since GP derivative functions sometimes return sparse
        or incomplete data (e.g., gradient of mean returned as a vector instead of a diagonal matrix; gradient of variance
        only differentiates wrt a single point at a time); hence we need specialized handlers for testing.
     b. Ping for derivative accuracy (PingGPComponentTest, PingEITest); these unit test the analytic derivatives.

  2. Monte-Carlo EI vs analytic EI validation: the monte-carlo versions are run to "high" accuracy and checked against
     analytic formulae when applicable
  3. Gradient Descent: using polynomials and other simple fucntions with analytically known optima
     to verify that the algorithm(s) underlying EI optimization are performing correctly.
  4. Single-threaded vs multi-threaded EI optimization validation: single and multi-threaded runs are checked to have the same
     output.
  5. End-to-end test of the EI optimization process for the analytic and monte-carlo cases.  These tests use constructed
     data for inputs but otherwise exercise the same code paths used for EI optimization in production.
\endrst*/

// #define OL_VERBOSE_PRINT

#include "gpp_math_test.hpp"

#include <cmath>
#include <cstdio>

#include <algorithm>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <vector>

#include <boost/random/uniform_real.hpp>  // NOLINT(build/include_order)
#include <omp.h>  // NOLINT(build/include_order)

#include "gpp_common.hpp"
#include "gpp_covariance.hpp"
#include "gpp_domain.hpp"
#include "gpp_exception.hpp"
#include "gpp_geometry.hpp"
#include "gpp_linear_algebra.hpp"
#include "gpp_logging.hpp"
#include "gpp_math.hpp"
#include "gpp_mock_optimization_objective_functions.hpp"
#include "gpp_optimization.hpp"
#include "gpp_optimizer_parameters.hpp"
#include "gpp_random.hpp"
#include "gpp_test_utils.hpp"

namespace optimal_learning {

namespace {  // contains classes/routines for ping testing GP/EI quantities and checking EI threaded consistency

/*!\rst
  Supports evaluating the GP mean, ComputeMeanOfPoints() and its gradient, ComputeGradMeanOfPoints.

  The gradient is taken wrt ``points_to_sample[dim][num_to_sample]``, so this is the ``input_matrix``, ``X_{d,i}``.
  The other inputs to GP mean are not differentiated against, so they are taken as input and stored by the constructor.

  Also, ComputeGradMeanOfPoints() stores a compact version of the gradient (by skipping known 0s) that *does not* have size
  GetGradientsSize().  EvaluateAndStoreAnalyticGradient and GetAnalyticGradient account for this indexing scheme appropriately.
\endrst*/
class PingGPPMean final : public PingableMatrixInputVectorOutputInterface {
 public:
  constexpr static char const * const kName = "GP Mean";

  PingGPPMean(double const * restrict lengths, double const * restrict points_sampled, double const * restrict points_sampled_value, double alpha, int dim, int num_to_sample, int num_sampled) OL_NONNULL_POINTERS
      : dim_(dim),
        num_to_sample_(num_to_sample),
        num_sampled_(num_sampled),
        gradients_already_computed_(false),
        noise_variance_(num_sampled_, 0.0),
        points_sampled_(points_sampled, points_sampled + dim_*num_sampled_),
        points_sampled_value_(points_sampled_value, points_sampled_value + num_sampled_),
        grad_mu_(num_to_sample_*dim_),
        sqexp_covariance_(dim_, alpha, lengths),
        gaussian_process_(sqexp_covariance_, points_sampled_.data(), points_sampled_value_.data(), noise_variance_.data(), dim_, num_sampled_) {
  }

  virtual void GetInputSizes(int * num_rows, int * num_cols) const noexcept override OL_NONNULL_POINTERS {
    *num_rows = dim_;
    *num_cols = num_to_sample_;
  }

  virtual int GetGradientsSize() const noexcept override OL_WARN_UNUSED_RESULT {
    return dim_*num_to_sample_*GetOutputSize();
  }

  virtual int GetOutputSize() const noexcept override OL_WARN_UNUSED_RESULT {
    return num_to_sample_;
  }

  virtual void EvaluateAndStoreAnalyticGradient(double const * restrict points_to_sample, double * restrict gradients) noexcept override OL_NONNULL_POINTERS_LIST(2) {
    if (gradients_already_computed_ == true) {
      OL_WARNING_PRINTF("WARNING: grad_mu alrady set.  Overwriting...\n");
    }
    gradients_already_computed_ = true;

    int num_derivatives = num_to_sample_;
    GaussianProcess::StateType points_to_sample_state(gaussian_process_, points_to_sample, num_to_sample_, num_derivatives);
    gaussian_process_.ComputeGradMeanOfPoints(points_to_sample_state, grad_mu_.data());

    if (gradients != nullptr) {
      // Since ComputeGradMeanOfPoints does not store known zeros in the gradient, we need to resconstruct the more general
      // tensor structure (including all zeros). This more general tensor is "block" diagonal.

      std::fill(gradients, gradients + dim_*Square(num_to_sample_), 0.0);

      // Loop over just the block diagonal entries and copy over the computed gradients.
      for (int i = 0; i < num_to_sample_; ++i) {
        for (int d = 0; d < dim_; ++d) {
          gradients[i*dim_*num_to_sample_ + i*dim_ + d] = grad_mu_[i*dim_ + d];
        }
      }
    }
  }

  virtual double GetAnalyticGradient(int row_index, int column_index, int output_index) const OL_WARN_UNUSED_RESULT {
    if (gradients_already_computed_ == false) {
      OL_THROW_EXCEPTION(OptimalLearningException, "PingGPPMean::GetAnalyticGradient() called BEFORE EvaluateAndStoreAnalyticGradient. NO DATA!");
    }

    if (column_index == output_index) {
      return grad_mu_[column_index*dim_ + row_index];
    } else {
      // these entries are analytically known to be 0.0 and thus were not stored
      // in grad_mu_
      return 0.0;
    }
  }

  virtual void EvaluateFunction(double const * restrict points_to_sample, double * restrict function_values) const noexcept override OL_NONNULL_POINTERS {
    int num_derivatives = 0;
    GaussianProcess::StateType points_to_sample_state(gaussian_process_, points_to_sample, num_to_sample_, num_derivatives);
    gaussian_process_.ComputeMeanOfPoints(points_to_sample_state, function_values);
  }

 private:
  //! spatial dimension (e.g., entries per point of ``points_sampled``)
  int dim_;
  //! number of points currently being sampled
  int num_to_sample_;
  //! number of points in ``points_sampled``
  int num_sampled_;
  //! whether gradients been computed and stored--whether this class is ready for use
  bool gradients_already_computed_;

  //! ``\sigma_n^2``, the noise variance
  std::vector<double> noise_variance_;
  //! coordinates of already-sampled points, ``X``
  std::vector<double> points_sampled_;
  //! function values at points_sampled, ``y``
  std::vector<double> points_sampled_value_;
  //! the gradient of the GP mean evaluated at union_of_points, wrt union_of_points[0:num_to_sample]
  std::vector<double> grad_mu_;

  //! covariance class (for computing covariance and its gradients)
  SquareExponential sqexp_covariance_;
  //! gaussian process used for computations
  GaussianProcess gaussian_process_;

  OL_DISALLOW_DEFAULT_AND_COPY_AND_ASSIGN(PingGPPMean);
};

/*!\rst
  Supports evaluating the GP variance, ComputeVarianceOfPoints() and its gradient, ComputeGradVarianceOfPoints.

  The gradient is taken wrt ``points_to_sample[dim][num_to_sample]``, so this is the ``input_matrix``, ``X_{d,i}``.
  The other inputs to GP variance are not differentiated against, so they are taken as input and stored by the constructor.

  The output is a matrix of dimension num_to_sample.  To fit into the PingMatrix...Interface, this is treated as a vector
  of length ``num_to_sample^2``.
\endrst*/
class PingGPPVariance final : public PingableMatrixInputVectorOutputInterface {
 public:
  constexpr static char const * const kName = "GP Variance";

  PingGPPVariance(double const * restrict lengths, double const * restrict points_sampled, double const * restrict OL_UNUSED(points_sampled_value), double alpha, int dim, int num_to_sample, int num_sampled) OL_NONNULL_POINTERS
      : dim_(dim),
        num_to_sample_(num_to_sample),
        num_sampled_(num_sampled),
        gradients_already_computed_(false),
        noise_variance_(num_sampled_, 0.0),
        points_sampled_(points_sampled, points_sampled + dim_*num_sampled_),
        grad_variance_(dim_*Square(num_to_sample_)*num_to_sample_),
        sqexp_covariance_(dim_, alpha, lengths),
        gaussian_process_(sqexp_covariance_, points_sampled_.data(), std::vector<double>(num_sampled_, 0.0).data(), noise_variance_.data(), dim_, num_sampled_) {
  }

  virtual void GetInputSizes(int * num_rows, int * num_cols) const noexcept override OL_NONNULL_POINTERS {
    *num_rows = dim_;
    *num_cols = num_to_sample_;
  }

  virtual int GetGradientsSize() const noexcept override OL_WARN_UNUSED_RESULT {
    return dim_*num_to_sample_*GetOutputSize();
  }

  virtual int GetOutputSize() const noexcept override OL_WARN_UNUSED_RESULT {
    return Square(num_to_sample_);
  }

  virtual void EvaluateAndStoreAnalyticGradient(double const * restrict points_to_sample, double * restrict gradients) noexcept override OL_NONNULL_POINTERS_LIST(2) {
    if (gradients_already_computed_ == true) {
      OL_WARNING_PRINTF("WARNING: grad_variance data already set.  Overwriting...\n");
    }
    gradients_already_computed_ = true;

    int num_derivatives = num_to_sample_;
    GaussianProcess::StateType points_to_sample_state(gaussian_process_, points_to_sample, num_to_sample_, num_derivatives);
    gaussian_process_.ComputeGradVarianceOfPoints(&points_to_sample_state, grad_variance_.data());

    if (gradients != nullptr) {
      OL_THROW_EXCEPTION(OptimalLearningException, "PingGPPVariance::EvaluateAndStoreAnalyticGradient() does not support direct gradient output.");
    }
  }

  virtual double GetAnalyticGradient(int row_index, int column_index, int output_index) const override OL_WARN_UNUSED_RESULT {
    if (gradients_already_computed_ == false) {
      OL_THROW_EXCEPTION(OptimalLearningException, "PingGPPVariance::GetAnalyticGradient() called BEFORE EvaluateAndStoreAnalyticGradient. NO DATA!");
    }

    return grad_variance_[column_index*Square(num_to_sample_)*dim_ + output_index*dim_ + row_index];
  }

  virtual void EvaluateFunction(double const * restrict points_to_sample, double * restrict function_values) const noexcept override OL_NONNULL_POINTERS {
    int num_derivatives = 0;
    GaussianProcess::StateType points_to_sample_state(gaussian_process_, points_to_sample, num_to_sample_, num_derivatives);
    gaussian_process_.ComputeVarianceOfPoints(&points_to_sample_state, function_values);

    // var_of_points outputs only to the lower triangle.  Copy it into the upper triangle to get a symmetric matrix
    for (int i = 0; i < num_to_sample_; ++i) {
      for (int j = 0; j < i; ++j) {
        function_values[i*num_to_sample_ + j] = function_values[j*num_to_sample_ + i];
      }
    }
  }

 private:
  //! spatial dimension (e.g., entries per point of ``points_sampled``)
  int dim_;
  //! number of points currently being sampled
  int num_to_sample_;
  //! number of points in ``points_sampled``
  int num_sampled_;
  //! whether gradients been computed and stored--whether this class is ready for use
  bool gradients_already_computed_;

  //! ``\sigma_n^2``, the noise variance
  std::vector<double> noise_variance_;
  //! coordinates of already-sampled points, ``X``
  std::vector<double> points_sampled_;
  //! the gradient of the GP variance evaluated at union_of_points, wrt union_of_points[0:num_to_sample]
  std::vector<double> grad_variance_;

  //! covariance class (for computing covariance and its gradients)
  SquareExponential sqexp_covariance_;
  //! gaussian process used for computations
  GaussianProcess gaussian_process_;

  OL_DISALLOW_DEFAULT_AND_COPY_AND_ASSIGN(PingGPPVariance);
};

/*!\rst
  Supports evaluating the cholesky factorization of the GP variance, the transpose of the cholesky factorization of: ComputeVarianceOfPoints()
  and its gradient, ComputeGradCholeskyVarianceOfPoints.

  The gradient is taken wrt ``points_to_sample[dim][num_to_sample]``, so this is the ``input_matrix``, ``X_{d,i}``.
  The other inputs to GP variance are not differentiated against, so they are taken as input and stored by the constructor.

  The output is a matrix of dimension num_to_sample.  To fit into the PingMatrix...Interface, this is treated as a vector
  of length ``num_to_sample^2``.
\endrst*/
class PingGPPCholeskyVariance final : public PingableMatrixInputVectorOutputInterface {
 public:
  constexpr static char const * const kName = "GP Cholesky Variance";

  PingGPPCholeskyVariance(double const * restrict lengths, double const * restrict points_sampled, double const * restrict OL_UNUSED(points_sampled_value), double alpha, int dim, int num_to_sample, int num_sampled) OL_NONNULL_POINTERS
      : dim_(dim),
        num_to_sample_(num_to_sample),
        num_sampled_(num_sampled),
        gradients_already_computed_(false),
        noise_variance_(num_sampled_, 0.0),
        points_sampled_(points_sampled, points_sampled + dim_*num_sampled_),
        grad_variance_(dim_*Square(num_to_sample_)*num_to_sample_),
        sqexp_covariance_(dim_, alpha, lengths),
        gaussian_process_(sqexp_covariance_, points_sampled_.data(), std::vector<double>(num_sampled_, 0.0).data(), noise_variance_.data(), dim_, num_sampled_) {
  }

  virtual void GetInputSizes(int * num_rows, int * num_cols) const noexcept override OL_NONNULL_POINTERS {
    *num_rows = dim_;
    *num_cols = num_to_sample_;
  }

  virtual int GetGradientsSize() const noexcept override OL_WARN_UNUSED_RESULT {
    return dim_*num_to_sample_*GetOutputSize();
  }

  virtual int GetOutputSize() const noexcept override OL_WARN_UNUSED_RESULT {
    return Square(num_to_sample_);
  }

  virtual void EvaluateAndStoreAnalyticGradient(double const * restrict points_to_sample, double * restrict gradients) noexcept override OL_NONNULL_POINTERS_LIST(2) {
    if (gradients_already_computed_ == true) {
      OL_WARNING_PRINTF("WARNING: grad_variance data already set.  Overwriting...\n");
    }
    gradients_already_computed_ = true;

    int num_derivatives = num_to_sample_;
    GaussianProcess::StateType points_to_sample_state(gaussian_process_, points_to_sample, num_to_sample_, num_derivatives);
    std::vector<double> variance_of_points(Square(num_to_sample_));
    gaussian_process_.ComputeVarianceOfPoints(&points_to_sample_state, variance_of_points.data());
    int OL_UNUSED(chol_info) = ComputeCholeskyFactorL(num_to_sample_, variance_of_points.data());

    gaussian_process_.ComputeGradCholeskyVarianceOfPoints(&points_to_sample_state, variance_of_points.data(), grad_variance_.data());

    if (gradients != nullptr) {
      OL_THROW_EXCEPTION(OptimalLearningException, "PingGPPCholeskyVariance::EvaluateAndStoreAnalyticGradient() does not support direct gradient output.");
    }
  }

  virtual double GetAnalyticGradient(int row_index, int column_index, int output_index) const override OL_WARN_UNUSED_RESULT {
    if (gradients_already_computed_ == false) {
      OL_THROW_EXCEPTION(OptimalLearningException, "PingGPPCholeskyVariance::GetAnalyticGradient() called BEFORE EvaluateAndStoreAnalyticGradient. NO DATA!");
    }

    return grad_variance_[column_index*Square(num_to_sample_)*dim_ + output_index*dim_ + row_index];
  }

  OL_NONNULL_POINTERS void EvaluateFunction(double const * restrict points_to_sample, double * restrict function_values) const noexcept override {
    int num_derivatives = 0;
    GaussianProcess::StateType points_to_sample_state(gaussian_process_, points_to_sample, num_to_sample_, num_derivatives);
    std::vector<double> chol_temp(Square(num_to_sample_));
    gaussian_process_.ComputeVarianceOfPoints(&points_to_sample_state, chol_temp.data());
    int OL_UNUSED(chol_info) = ComputeCholeskyFactorL(num_to_sample_, chol_temp.data());
    ZeroUpperTriangle(num_to_sample_, chol_temp.data());
    MatrixTranspose(chol_temp.data(), num_to_sample_, num_to_sample_, function_values);
  }

 private:
  //! spatial dimension (e.g., entries per point of ``points_sampled``)
  int dim_;
  //! number of points currently being sampled
  int num_to_sample_;
  //! number of points in ``points_sampled``
  int num_sampled_;
  //! whether gradients been computed and stored--whether this class is ready for use
  bool gradients_already_computed_;

  //! ``\sigma_n^2``, the noise variance
  std::vector<double> noise_variance_;
  //! coordinates of already-sampled points, ``X``
  std::vector<double> points_sampled_;
  //! the gradient of the cholesky factorization of the GP variance evaluated at union_of_points, wrt union_of_points[0:num_to_sample]
  std::vector<double> grad_variance_;

  //! covariance class (for computing covariance and its gradients)
  SquareExponential sqexp_covariance_;
  //! gaussian process used for computations
  GaussianProcess gaussian_process_;

  OL_DISALLOW_DEFAULT_AND_COPY_AND_ASSIGN(PingGPPCholeskyVariance);
};

/*!\rst
  Supports evaluating the expected improvement, ExpectedImprovementEvaluator::ComputeExpectedImprovement() and
  its gradient, ExpectedImprovementEvaluator::ComputeGradExpectedImprovement()

  The gradient is taken wrt ``points_to_sample[dim]``, so this is the ``input_matrix``, ``X_{d,i}``.
  The other inputs to EI are not differentiated against, so they are taken as input and stored by the constructor.

  The output of EI is a scalar.
\endrst*/
class PingExpectedImprovement final : public PingableMatrixInputVectorOutputInterface {
 public:
  constexpr static char const * const kName = "EI with MC integration";

  PingExpectedImprovement(double const * restrict lengths, double const * restrict points_being_sampled, double const * restrict points_sampled, double const * restrict points_sampled_value, double alpha, double best_so_far, int dim, int num_to_sample, int num_being_sampled, int num_sampled, int num_mc_iter) OL_NONNULL_POINTERS
      : dim_(dim),
        num_to_sample_(num_to_sample),
        num_being_sampled_(num_being_sampled),
        num_sampled_(num_sampled),
        gradients_already_computed_(false),
        noise_variance_(num_sampled_, 0.0),
        points_sampled_(points_sampled, points_sampled + dim_*num_sampled_),
        points_sampled_value_(points_sampled_value, points_sampled_value + num_sampled_),
        points_being_sampled_(points_being_sampled, points_being_sampled + num_being_sampled_*dim_),
        grad_EI_(num_to_sample_*dim_),
        sqexp_covariance_(dim_, alpha, lengths),
        gaussian_process_(sqexp_covariance_, points_sampled_.data(), points_sampled_value_.data(), noise_variance_.data(), dim_, num_sampled_), ei_evaluator_(gaussian_process_, num_mc_iter, best_so_far) {
  }

  virtual void GetInputSizes(int * num_rows, int * num_cols) const noexcept override OL_NONNULL_POINTERS {
    *num_rows = dim_;
    *num_cols = num_to_sample_;
  }

  virtual int GetGradientsSize() const noexcept override OL_WARN_UNUSED_RESULT {
    return dim_*GetOutputSize();
  }

  virtual int GetOutputSize() const noexcept override OL_WARN_UNUSED_RESULT {
    return 1;
  }

  virtual void EvaluateAndStoreAnalyticGradient(double const * restrict points_to_sample, double * restrict gradients) noexcept override OL_NONNULL_POINTERS_LIST(2) {
    if (gradients_already_computed_ == true) {
      OL_WARNING_PRINTF("WARNING: grad_EI data already set.  Overwriting...\n");
    }
    gradients_already_computed_ = true;

    NormalRNG normal_rng(3141);
    bool configure_for_gradients = true;
    ExpectedImprovementEvaluator::StateType ei_state(ei_evaluator_, points_to_sample, points_being_sampled_.data(), num_to_sample_, num_being_sampled_, configure_for_gradients, &normal_rng);
    ei_evaluator_.ComputeGradExpectedImprovement(&ei_state, grad_EI_.data());

    if (gradients != nullptr) {
      std::copy(grad_EI_.begin(), grad_EI_.end(), gradients);
    }
  }

  virtual double GetAnalyticGradient(int row_index, int column_index, int OL_UNUSED(output_index)) const override OL_WARN_UNUSED_RESULT {
    if (gradients_already_computed_ == false) {
      OL_THROW_EXCEPTION(OptimalLearningException, "PingExpectedImprovement::GetAnalyticGradient() called BEFORE EvaluateAndStoreAnalyticGradient. NO DATA!");
    }

    return grad_EI_[column_index*dim_ + row_index];
  }

  virtual void EvaluateFunction(double const * restrict points_to_sample, double * restrict function_values) const noexcept override OL_NONNULL_POINTERS {
    NormalRNG normal_rng(3141);
    bool configure_for_gradients = false;
    ExpectedImprovementEvaluator::StateType ei_state(ei_evaluator_, points_to_sample, points_being_sampled_.data(), num_to_sample_, num_being_sampled_, configure_for_gradients, &normal_rng);
    *function_values = ei_evaluator_.ComputeExpectedImprovement(&ei_state);
  }

 private:
  //! spatial dimension (e.g., entries per point of ``points_sampled``)
  int dim_;
  //! number of potential future samples; gradients are evaluated wrt these points (i.e., the "q" in q,p-EI)
  int num_to_sample_;
  //! number of points being sampled concurrently (i.e., the "p" in q,p-EI)
  int num_being_sampled_;
  //! number of points in ``points_sampled``
  int num_sampled_;
  //! whether gradients been computed and stored--whether this class is ready for use
  bool gradients_already_computed_;

  //! ``\sigma_n^2``, the noise variance
  std::vector<double> noise_variance_;
  //! coordinates of already-sampled points, ``X``
  std::vector<double> points_sampled_;
  //! function values at points_sampled, ``y``
  std::vector<double> points_sampled_value_;
  //! points that are being sampled in concurrently experiments
  std::vector<double> points_being_sampled_;
  //! the gradient of EI at union_of_points, wrt union_of_points[0:num_to_sample]
  std::vector<double> grad_EI_;

  //! covariance class (for computing covariance and its gradients)
  SquareExponential sqexp_covariance_;
  //! gaussian process used for computations
  GaussianProcess gaussian_process_;
  //! expected improvement evaluator object that specifies the parameters & GP for EI evaluation
  ExpectedImprovementEvaluator ei_evaluator_;

  OL_DISALLOW_DEFAULT_AND_COPY_AND_ASSIGN(PingExpectedImprovement);
};

/*!\rst
  Supports evaluating an analytic special case of expected improvement via OnePotentialSampleExpectedImprovementEvaluator.

  The gradient is taken wrt ``points_to_sample[dim]``, so this is the ``input_matrix``, ``X_{d,i}`` (with i always indexing 0).
  The other inputs to EI are not differentiated against, so they are taken as input and stored by the constructor.

  The output of EI is a scalar.
\endrst*/
class PingOnePotentialSampleExpectedImprovement final : public PingableMatrixInputVectorOutputInterface {
 public:
  constexpr static char const * const kName = "EI ONE potential sample analytic";

  PingOnePotentialSampleExpectedImprovement(double const * restrict lengths, double const * restrict OL_UNUSED(points_being_sampled), double const * restrict points_sampled, double const * restrict points_sampled_value, double alpha, double best_so_far, int dim, int OL_UNUSED(num_to_sample), int num_being_sampled, int num_sampled, int OL_UNUSED(num_mc_iter)) OL_NONNULL_POINTERS
      : dim_(dim),
        num_sampled_(num_sampled),
        gradients_already_computed_(false),
        noise_variance_(num_sampled_, 0.0),
        points_sampled_(points_sampled, points_sampled + dim_*num_sampled_),
        points_sampled_value_(points_sampled_value, points_sampled_value + num_sampled_),
        grad_EI_(dim_),
        sqexp_covariance_(dim_, alpha, lengths),
        gaussian_process_(sqexp_covariance_, points_sampled_.data(), points_sampled_value_.data(), noise_variance_.data(), dim_, num_sampled_), ei_evaluator_(gaussian_process_, best_so_far) {
    if (num_being_sampled != 0) {
      OL_THROW_EXCEPTION(InvalidValueException<int>, "PingOnePotentialSample: num_being_sampled MUST be 0!", num_being_sampled, 0);
    }
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

  virtual void EvaluateAndStoreAnalyticGradient(double const * restrict points_to_sample, double * restrict gradients) noexcept override OL_NONNULL_POINTERS_LIST(2) {
    if (gradients_already_computed_ == true) {
      OL_WARNING_PRINTF("WARNING: grad_EI data already set.  Overwriting...\n");
    }
    gradients_already_computed_ = true;

    bool configure_for_gradients = true;
    OnePotentialSampleExpectedImprovementEvaluator::StateType ei_state(ei_evaluator_, points_to_sample, configure_for_gradients);
    ei_evaluator_.ComputeGradExpectedImprovement(&ei_state, grad_EI_.data());

    if (gradients != nullptr) {
      std::copy(grad_EI_.begin(), grad_EI_.end(), gradients);
    }
  }

  virtual double GetAnalyticGradient(int row_index, int OL_UNUSED(column_index), int OL_UNUSED(output_index)) const override OL_WARN_UNUSED_RESULT {
    if (gradients_already_computed_ == false) {
      OL_THROW_EXCEPTION(OptimalLearningException, "PingOnePotentialSampleExpectedImprovement::GetAnalyticGradient() called BEFORE EvaluateAndStoreAnalyticGradient. NO DATA!");
    }

    return grad_EI_[row_index];
  }

  virtual void EvaluateFunction(double const * restrict points_to_sample, double * restrict function_values) const noexcept override OL_NONNULL_POINTERS {
    bool configure_for_gradients = false;
    OnePotentialSampleExpectedImprovementEvaluator::StateType ei_state(ei_evaluator_, points_to_sample, configure_for_gradients);
    *function_values = ei_evaluator_.ComputeExpectedImprovement(&ei_state);
  }

 private:
  //! spatial dimension (e.g., entries per point of ``points_sampled``)
  int dim_;
  //! number of points in ``points_sampled``
  int num_sampled_;
  //! number of points being sampled concurrently (i.e., the "p" in q,p-EI). Must be 0 for the analytic case.
  const int num_being_sampled_ = 0;
  bool gradients_already_computed_;

  //! ``\sigma_n^2``, the noise variance
  std::vector<double> noise_variance_;
  //! coordinates of already-sampled points, ``X``
  std::vector<double> points_sampled_;
  //! function values at points_sampled, ``y``
  std::vector<double> points_sampled_value_;
  //! the gradient of EI at union_of_points, wrt union_of_points[0:num_to_sample]
  std::vector<double> grad_EI_;

  //! covariance class (for computing covariance and its gradients)
  SquareExponential sqexp_covariance_;
  //! gaussian process used for computations
  GaussianProcess gaussian_process_;
  //! expected improvement evaluator object that specifies the parameters & GP for EI evaluation
  OnePotentialSampleExpectedImprovementEvaluator ei_evaluator_;

  OL_DISALLOW_DEFAULT_AND_COPY_AND_ASSIGN(PingOnePotentialSampleExpectedImprovement);
};

/*!\rst
  Pings gradients (spatial) of GP components (e.g., mean, variance, cholesky of variance) 50 times with randomly generated test cases

  \param
    :epsilon: coarse, fine ``h`` sizes to use in finite difference computation
    :tolerance_fine: desired amount of deviation from the exact rate
    :tolerance_coarse: maximum allowable abmount of deviation from the exact rate
    :input_output_ratio: for ``||analytic_gradient||/||input|| < input_output_ratio``, ping testing is not performed, see PingDerivative()
  \return
    number of ping/test failures
\endrst*/
template <typename GPComponentEvaluator>
OL_WARN_UNUSED_RESULT int PingGPComponentTest(double epsilon[2], double tolerance_fine, double tolerance_coarse, double input_output_ratio) {
  int total_errors = 0;
  int errors_this_iteration = 0;
  const int dim = 3;

  int num_being_sampled = 0;
  int num_to_sample = 5;
  int num_sampled = 7;

  std::vector<double> lengths(dim);
  double alpha = 2.80723;

  MockExpectedImprovementEnvironment EI_environment;

  UniformRandomGenerator uniform_generator(2718);
  boost::uniform_real<double> uniform_double(0.5, 2.5);

  for (int i = 0; i < 50; ++i) {
    EI_environment.Initialize(dim, num_to_sample, num_being_sampled, num_sampled);

    for (int j = 0; j < dim; ++j) {
      lengths[j] = uniform_double(uniform_generator.engine);
    }

    GPComponentEvaluator gp_component_evaluator(lengths.data(), EI_environment.points_sampled(), EI_environment.points_sampled_value(), alpha, EI_environment.dim, EI_environment.num_to_sample, EI_environment.num_sampled);
    gp_component_evaluator.EvaluateAndStoreAnalyticGradient(EI_environment.points_to_sample(), nullptr);
    errors_this_iteration = PingDerivative(gp_component_evaluator, EI_environment.points_to_sample(), epsilon, tolerance_fine, tolerance_coarse, input_output_ratio);

    if (errors_this_iteration != 0) {
      OL_PARTIAL_FAILURE_PRINTF("on iteration %d\n", i);
    }
    total_errors += errors_this_iteration;
  }

  if (total_errors != 0) {
    OL_PARTIAL_FAILURE_PRINTF("%s gradient pings failed with %d errors\n", GPComponentEvaluator::kName, total_errors);
  } else {
    OL_PARTIAL_SUCCESS_PRINTF("%s gradient pings passed\n", GPComponentEvaluator::kName);
  }

  return total_errors;
}

}  // end unnamed namespace

/*!\rst
  Pings the gradients (spatial) of the GP mean 50 times with randomly generated test cases

  \return
    number of ping/test failures
\endrst*/
int PingGPMeanTest() {
  double epsilon_gp_mean[2] = {5.0e-3, 1.0e-3};
  int total_errors = PingGPComponentTest<PingGPPMean>(epsilon_gp_mean, 2.0e-3, 2.0e-3, 1.0e-18);
  return total_errors;
}

/*!\rst
  Pings the gradients (spatial) of the GP variance 50 times with randomly generated test cases

  \return
    number of ping/test failures
\endrst*/
int PingGPVarianceTest() {
  double epsilon_gp_variance[2] = {5.32879e-3, 0.942478e-3};
  int total_errors = PingGPComponentTest<PingGPPVariance>(epsilon_gp_variance, 2.0e-2, 4.0e-1, 1.0e-18);
  return total_errors;
}

/*!\rst
  Wrapper to ping the gradients (spatial) of the cholesky factorization.

  \return
    number of ping/test failures
\endrst*/
int PingGPCholeskyVarianceTest() {
  double epsilon_gp_variance[2] = {5.5e-3, 0.932e-3};
  int total_errors = PingGPComponentTest<PingGPPCholeskyVariance>(epsilon_gp_variance, 9.0e-3, 3.0e-1, 1.0e-18);
  return total_errors;
}

/*!\rst
  Pings the gradients (spatial) of the EI 50 times with randomly generated test cases
  Works with various EI evaluators (e.g., MC, analytic formulae)

  \param
    :num_to_sample: number of potential future samples; gradients are evaluated wrt these points (i.e., the "q" in q,p-EI)
    :num_being_sampled: number of points being sampled in concurrent experiments (i.e., the "p" in q,p-EI)
    :epsilon: coarse, fine ``h`` sizes to use in finite difference computation
    :tolerance_fine: desired amount of deviation from the exact rate
    :tolerance_coarse: maximum allowable abmount of deviation from the exact rate
    :input_output_ratio: for ``||analytic_gradient||/||input|| < input_output_ratio``, ping testing is not performed, see PingDerivative()
  \return
    number of ping/test failures
\endrst*/
template <typename EIEvaluator>
OL_WARN_UNUSED_RESULT int PingEITest(int num_to_sample, int num_being_sampled, double epsilon[2], double tolerance_fine, double tolerance_coarse, double input_output_ratio) {
  int total_errors = 0;
  int errors_this_iteration;
  const int dim = 3;

  int num_sampled = 7;

  std::vector<double> lengths(dim);
  double alpha = 2.80723;
  // set best_so_far to be larger than max(points_sampled_value) (but don't make it huge or stability will be suffer)
  double best_so_far = 7.0;
  const int num_mc_iter = 16;

  MockExpectedImprovementEnvironment EI_environment;

  UniformRandomGenerator uniform_generator(2718);
  boost::uniform_real<double> uniform_double(0.5, 2.5);

  for (int i = 0; i < 50; ++i) {
    EI_environment.Initialize(dim, num_to_sample, num_being_sampled, num_sampled);

    for (int j = 0; j < dim; ++j) {
      lengths[j] = uniform_double(uniform_generator.engine);
    }

    EIEvaluator EI_evaluator(lengths.data(), EI_environment.points_being_sampled(), EI_environment.points_sampled(), EI_environment.points_sampled_value(), alpha, best_so_far, EI_environment.dim, EI_environment.num_to_sample, EI_environment.num_being_sampled, EI_environment.num_sampled, num_mc_iter);
    EI_evaluator.EvaluateAndStoreAnalyticGradient(EI_environment.points_to_sample(), nullptr);
    errors_this_iteration = PingDerivative(EI_evaluator, EI_environment.points_to_sample(), epsilon, tolerance_fine, tolerance_coarse, input_output_ratio);

    if (errors_this_iteration != 0) {
      OL_PARTIAL_FAILURE_PRINTF("on iteration %d\n", i);
    }
    total_errors += errors_this_iteration;
  }

  if (total_errors != 0) {
    OL_PARTIAL_FAILURE_PRINTF("%s (%d,%d-EI) gradient pings failed with %d errors\n", EIEvaluator::kName, num_to_sample, num_being_sampled, total_errors);
  } else {
    OL_PARTIAL_SUCCESS_PRINTF("%s (%d,%d-EI) gradient pings passed\n", EIEvaluator::kName, num_to_sample, num_being_sampled);
  }

  return total_errors;
}

/*!\rst
  Pings the gradients (spatial) of the EI 50 times with randomly generated test cases

  \return
    number of ping/test failures
\endrst*/
int PingEIGeneralTest() {
  double epsilon_EI[2] = {1.0e-2, 1.0e-3};
  int total_errors = PingEITest<PingExpectedImprovement>(1, 0, epsilon_EI, 2.0e-3, 9.0e-2, 1.0e-18);

  total_errors += PingEITest<PingExpectedImprovement>(1, 5, epsilon_EI, 2.0e-3, 9.0e-2, 1.0e-18);

  total_errors += PingEITest<PingExpectedImprovement>(3, 2, epsilon_EI, 2.0e-3, 9.0e-2, 1.0e-18);

  total_errors += PingEITest<PingExpectedImprovement>(4, 0, epsilon_EI, 2.0e-3, 9.0e-2, 1.0e-18);
  return total_errors;
}

/*!\rst
  Pings the gradients (spatial) of the EI (one potential sample special case) 50 times with randomly generated test cases

  \return
    number of ping/test failures
\endrst*/
int PingEIOnePotentialSampleTest() {
  double epsilon_EI_one_potential_sample[2] = {5.0e-3, 9.0e-4};
  int total_errors = PingEITest<PingOnePotentialSampleExpectedImprovement>(1, 0, epsilon_EI_one_potential_sample, 2.0e-3, 7.0e-2, 1.0e-18);
  return total_errors;
}

/*!\rst
  Test cases where analytic EI would attempt to compute 0/0 without variance lower bounds.

  The bounds are OnePotentialSampleExpectedImprovementEvaluator::kMinimumVarianceEI and
  kMinimumVarianceGradEI. See those class docs for more details.

  These particular test cases arose from plotting EI (easy since dim = 1) and checking
  that EI and grad_EI were being computed appropriately at the specified locations.
  The test cases are purposely simple; the requirement was that they trigger behavior
  that would result in 0/0 without minimum variance thresholds.

  Without the aforementioned thresholds, 1D analytic EI could attempt
  ``0/0 = (best_so_far - gp_mean) / sqrt(gp_variance)``
  The easiest way to do cause these conditions is to compute EI at (or near) one of
  points_sampled such that ``gp_mean == best_so_far`` and ``gp_variance == 0``.
  (Although these conditions can arise elsewhere; try plotting the test case in the code.)

  \return
    number of test failures
\endrst*/
int EIOnePotentialSampleEdgeCasesTest() {
  int total_errors = 0;

  const int dim = 1;
  const double base_coord = 0.5;
  std::vector<double> points_sampled = {base_coord, 2.0 * base_coord};
  std::vector<double> points_sampled_value = {-1.809342, -1.09342};
  std::vector<double> noise_variance = {0.0, 0.0};
  auto best_so_far = *std::min_element(points_sampled_value.begin(), points_sampled_value.end());

  SquareExponential covariance(dim, 0.2, 0.3);
  // First a symmetric case: only one historical point
  GaussianProcess gaussian_process(covariance, points_sampled.data(), points_sampled_value.data(),
                                   noise_variance.data(), dim, 1);

  double point_to_sample = base_coord;

  OnePotentialSampleExpectedImprovementEvaluator ei_evaluator(gaussian_process, best_so_far);
  bool configure_for_gradients = true;
  OnePotentialSampleExpectedImprovementState ei_state(ei_evaluator, &point_to_sample,
                                                      configure_for_gradients);

  double ei;
  double grad_ei;

  ei = ei_evaluator.ComputeExpectedImprovement(&ei_state);
  ei_evaluator.ComputeGradExpectedImprovement(&ei_state, &grad_ei);

  // check that EI and gradient are 0 when computed at the one historical data point
  double tolerance = 4.0 * std::numeric_limits<double>::epsilon();
  if (!CheckDoubleWithinRelative(ei, 0.0, tolerance)) {
    ++total_errors;
  }
  if (!CheckDoubleWithinRelative(grad_ei, 0.0, tolerance)) {
    ++total_errors;
  }

  // Compute ei at point_to_sample +/- shifts and check for equality
  {
    double left_ei, right_ei;
    double left_grad_ei, right_grad_ei;
    std::vector<double> shifts = {1.0e-15, 4.0e-11, 3.14e-6, 8.89e-1, 2.71};

    for (auto shift : shifts) {
      point_to_sample = base_coord - shift;
      ei_state.SetCurrentPoint(ei_evaluator, &point_to_sample);
      left_ei = ei_evaluator.ComputeExpectedImprovement(&ei_state);
      ei_evaluator.ComputeGradExpectedImprovement(&ei_state, &left_grad_ei);

      point_to_sample = base_coord + shift;
      ei_state.SetCurrentPoint(ei_evaluator, &point_to_sample);
      right_ei = ei_evaluator.ComputeExpectedImprovement(&ei_state);
      ei_evaluator.ComputeGradExpectedImprovement(&ei_state, &right_grad_ei);

      if (!CheckDoubleWithinRelative(left_ei, right_ei, 0.0)) {
        ++total_errors;
      }
      if (!CheckDoubleWithinRelative(left_grad_ei, -right_grad_ei, 0.0)) {
          ++total_errors;
      }
    }  // end for shift : shifts
  }  // end ei symmetry check

  // Now introduce some asymmetry with a second point
  // Right side has a larger objetive value, so the EI minimum
  // is shifted *slightly* to the left of best_so_far.
  gaussian_process.AddPointsToGP(points_sampled.data() + dim, &points_sampled_value[1], &noise_variance[1], 1);

  double shift = 3.0e-12;
  point_to_sample = base_coord - shift;
  ei_state.SetCurrentPoint(ei_evaluator, &point_to_sample);
  ei = ei_evaluator.ComputeExpectedImprovement(&ei_state);
  ei_evaluator.ComputeGradExpectedImprovement(&ei_state, &grad_ei);

  if (!CheckDoubleWithinRelative(ei, 0.0, 0.0)) {
    ++total_errors;
  }
  if (!CheckDoubleWithinRelative(grad_ei, 0.0, 0.0)) {
    ++total_errors;
  }

  if (total_errors != 0) {
    OL_PARTIAL_FAILURE_PRINTF("1D analytic EI 0/0 edge case tests failed with %d errors\n", total_errors);
  } else {
    OL_PARTIAL_SUCCESS_PRINTF("1D analytic EI 0/0 edge case tests passed\n");
  }

  return total_errors;
}

/*!\rst
  Generates a set of 50 random test cases for expected improvement with only one potential sample.
  The general EI (which uses MC integration) is evaluated to reasonably high accuracy (while not taking too long to run)
  and compared against the analytic formula version for consistency.  The gradients (spatial) of EI are also checked.

  \return
    number of cases where analytic and monte-carlo EI do not match
\endrst*/
int RunEIConsistencyTests() {
  int total_errors = 0;

  const int num_mc_iter = 1000000;
  const int dim = 3;
  const int num_being_sampled = 0;
  const int num_to_sample = 1;
  const int num_sampled = 7;

  double alpha = 2.80723;
  // set best_so_far to be larger than max(points_sampled_value) (but don't make it huge or stability will be suffer)
  double best_so_far = 10.0;

  int max_num_threads = 4;
  int chunk_size = 5;

#pragma omp parallel num_threads(max_num_threads)
  {
    int tid = omp_get_thread_num();
    UniformRandomGenerator uniform_generator(31278 + tid);
    boost::uniform_real<double> uniform_double(0.5, 2.5);

    MockExpectedImprovementEnvironment EI_environment;

    std::vector<double> lengths(dim);
    std::vector<double> grad_EI_general(dim);
    std::vector<double> grad_EI_one_potential_sample(dim);
    double EI_general;
    double EI_one_potential_sample;

#pragma omp for nowait schedule(static, chunk_size) reduction(+:total_errors)
    for (int i = 0; i < 40; ++i) {
      EI_environment.Initialize(dim, num_to_sample, num_being_sampled, num_sampled, &uniform_generator);

      for (int j = 0; j < dim; ++j) {
        lengths[j] = uniform_double(uniform_generator.engine);
      }

      PingOnePotentialSampleExpectedImprovement EI_one_potential_sample_evaluator(lengths.data(), EI_environment.points_being_sampled(), EI_environment.points_sampled(), EI_environment.points_sampled_value(), alpha, best_so_far, EI_environment.dim, EI_environment.num_to_sample, EI_environment.num_being_sampled, EI_environment.num_sampled, num_mc_iter);
      EI_one_potential_sample_evaluator.EvaluateAndStoreAnalyticGradient(EI_environment.points_to_sample(), grad_EI_one_potential_sample.data());
      EI_one_potential_sample_evaluator.EvaluateFunction(EI_environment.points_to_sample(), &EI_one_potential_sample);

      PingExpectedImprovement EI_evaluator(lengths.data(), EI_environment.points_being_sampled(), EI_environment.points_sampled(), EI_environment.points_sampled_value(), alpha, best_so_far, EI_environment.dim, EI_environment.num_to_sample, EI_environment.num_being_sampled, EI_environment.num_sampled, num_mc_iter);
      EI_evaluator.EvaluateAndStoreAnalyticGradient(EI_environment.points_to_sample(), grad_EI_general.data());
      EI_evaluator.EvaluateFunction(EI_environment.points_to_sample(), &EI_general);

      int ei_errors_this_iteration = 0;
      if (!CheckDoubleWithinRelative(EI_general, EI_one_potential_sample, 5.0e-4)) {
        ++ei_errors_this_iteration;
      }
      if (ei_errors_this_iteration != 0) {
        OL_PARTIAL_FAILURE_PRINTF("in EI on iteration %d\n", i);
      }
      total_errors += ei_errors_this_iteration;

      int grad_ei_errors_this_iteration = 0;
      for (int j = 0; j < dim; ++j) {
        if (!CheckDoubleWithinRelative(grad_EI_general[j], grad_EI_one_potential_sample[j], 6.5e-3)) {
          ++grad_ei_errors_this_iteration;
        }
      }

      if (grad_ei_errors_this_iteration != 0) {
        OL_PARTIAL_FAILURE_PRINTF("in EI gradients on iteration %d\n", i);
      }
      total_errors += grad_ei_errors_this_iteration;
    }
  }

  if (total_errors != 0) {
    OL_PARTIAL_FAILURE_PRINTF("comparing MC EI to analytic EI failed with %d total_errors\n\n", total_errors);
  } else {
    OL_PARTIAL_SUCCESS_PRINTF("comparing MC EI to analytic EI passed\n");
  }

  return total_errors;
}

int RunGPTests() {
  int total_errors = 0;
  int current_errors = 0;

  {
    current_errors = PingGPMeanTest();
    if (current_errors != 0) {
      OL_PARTIAL_FAILURE_PRINTF("pinging GP mean failed with %d errors\n", current_errors);
    }
    total_errors += current_errors;
  }

  {
    current_errors = PingGPVarianceTest();
    if (current_errors != 0) {
      OL_PARTIAL_FAILURE_PRINTF("pinging GP variance failed with %d errors\n", current_errors);
    }
    total_errors += current_errors;
  }

  {
    current_errors = PingGPCholeskyVarianceTest();
    if (current_errors != 0) {
      OL_PARTIAL_FAILURE_PRINTF("pinging GP cholesky of variance failed with %d errors\n", current_errors);
    }
    total_errors += current_errors;
  }

  {
    current_errors = PingEIGeneralTest();
    if (current_errors != 0) {
      OL_PARTIAL_FAILURE_PRINTF("pinging EI failed with %d errors\n", current_errors);
    }
    total_errors += current_errors;
  }

  {
    current_errors = PingEIOnePotentialSampleTest();
    if (current_errors != 0) {
      OL_PARTIAL_FAILURE_PRINTF("pinging analytic (one potential sample) EI failed with %d errors\n", current_errors);
    }
    total_errors += current_errors;
  }

  {
    current_errors = EIOnePotentialSampleEdgeCasesTest();
    if (current_errors != 0) {
      OL_PARTIAL_FAILURE_PRINTF("analytic (one potential sample) EI 0/0 cases failed with %d errors\n", current_errors);
    }
    total_errors += current_errors;
  }

  if (total_errors != 0) {
    OL_PARTIAL_FAILURE_PRINTF("GP functions failed with %d errors\n\n", total_errors);
  } else {
    OL_PARTIAL_SUCCESS_PRINTF("GP functions passed\n");
  }

  return total_errors;
}

/*!\rst
  Tests that single & multithreaded EI optimization produce *the exact same* results.

  We do this by first setting up EI optimization in a single threaded scenario with 2 starting points and 2 random number generators.
  Optimization is run one from starting point 0 with RNG 0, and then again from starting point 1 with RNG 1.

  Then we run the optimization multithreaded (with 2 threads) over both starting points simultaneously.  One of the threads
  will see the winning (point, RNG) pair from the single-threaded won.  Hence one result point will match with the single threaded
  results exactly.

  Then we re-run the multithreaded optimization, swapping the position of the RNGs and starting points.  If thread 0 won in the
  previous test, thread 1 will win here (and vice versa).

  Note that it's tricky to run single-threaded optimization over both starting points simultaneously because we won't know which
  (point, RNG) pair won (which is required to ascertain the 'winner' since we are not computing EI accurately enough to avoid
  error).
\endrst*/
int MultithreadedEIOptimizationTest(ExpectedImprovementEvaluationMode ei_mode) {
  using DomainType = TensorProductDomain;
  const int num_sampled = 17;
  static const int kDim = 3;

  // q,p-EI computation parameters
  int num_to_sample = 2;
  int num_being_sampled = 3;
  if (ei_mode == ExpectedImprovementEvaluationMode::kAnalytic) {
    num_to_sample = 1;
    num_being_sampled = 0;
  }
  std::vector<double> points_being_sampled(kDim*num_being_sampled);

  // gradient descent parameters
  const double gamma = 0.5;
  const double pre_mult = 1.4;
  const double max_relative_change = 1.0;
  const double tolerance = 1.0e-7;

  const int max_gradient_descent_steps = 250;
  const int max_num_restarts = 3;
  const int num_steps_averaged = 0;
  GradientDescentParameters gd_params(0, max_gradient_descent_steps, max_num_restarts,
                                      num_steps_averaged, gamma, pre_mult,
                                      max_relative_change, tolerance);

  int max_mc_iterations = 967;

  int total_errors = 0;

  // seed randoms
  UniformRandomGenerator uniform_generator(314);
  boost::uniform_real<double> uniform_double_hyperparameter(1.0, 2.5);
  boost::uniform_real<double> uniform_double_lower_bound(-2.0, 0.5);
  boost::uniform_real<double> uniform_double_upper_bound(2.5, 5.5);

  std::vector<double> noise_variance(num_sampled, 0.0003);
  MockGaussianProcessPriorData<DomainType> mock_gp_data(SquareExponential(kDim, 1.0, 1.0), noise_variance, kDim,
                                                        num_sampled, uniform_double_lower_bound,
                                                        uniform_double_upper_bound, uniform_double_hyperparameter,
                                                        &uniform_generator);

  for (int j = 0; j < num_being_sampled; ++j) {
    mock_gp_data.domain_ptr->GeneratePointInDomain(&uniform_generator, points_being_sampled.data() + j*kDim);
  }

  const int pi_array[] = {314, 3141, 31415, 314159};
  static const int kMaxNumThreads = 2;
  std::vector<NormalRNG> normal_rng_vec(kMaxNumThreads);
  for (int j = 0; j < kMaxNumThreads; ++j) {
    normal_rng_vec[j].SetExplicitSeed(pi_array[j]);
  }

  std::vector<double> starting_points(kDim*kMaxNumThreads*num_to_sample);
  for (int j = 0; j < kMaxNumThreads; ++j) {
    for (int k = 0; k < num_to_sample; ++k) {
      mock_gp_data.domain_ptr->GeneratePointInDomain(&uniform_generator, starting_points.data() + j*kDim*num_to_sample + k*kDim);
    }
  }

  // we will optimize over the expanded region
  std::vector<ClosedInterval> domain_bounds(mock_gp_data.domain_bounds);
  ExpandDomainBounds(3.2, &domain_bounds);
  DomainType domain(domain_bounds.data(), kDim);

  // build truth data by using single threads
  bool found_flag = false;
  std::vector<double> best_next_point_single_thread(kDim*num_to_sample*kMaxNumThreads*kMaxNumThreads);
  int num_threads = 1;
  ThreadSchedule thread_schedule(num_threads, omp_sched_static);
  for (int j = 0; j < kMaxNumThreads; ++j) {
    NormalRNG normal_rng(pi_array[j]);
    int one_multistart = 1;  // truth values come from single threaded execution
    ComputeOptimalPointsToSampleViaMultistartGradientDescent(*mock_gp_data.gaussian_process_ptr, gd_params,
                                                             domain, thread_schedule,
                                                             starting_points.data() + j*kDim*num_to_sample,
                                                             points_being_sampled.data(), one_multistart,
                                                             num_to_sample, num_being_sampled,
                                                             mock_gp_data.best_so_far, max_mc_iterations,
                                                             &normal_rng, &found_flag,
                                                             best_next_point_single_thread.data() + j*kDim*num_to_sample);
    if (!found_flag) {
      ++total_errors;
    }

    normal_rng.SetExplicitSeed(pi_array[kMaxNumThreads - j - 1]);
    ComputeOptimalPointsToSampleViaMultistartGradientDescent(*mock_gp_data.gaussian_process_ptr, gd_params,
                                                             domain, thread_schedule,
                                                             starting_points.data() + j*kDim*num_to_sample,
                                                             points_being_sampled.data(), one_multistart,
                                                             num_to_sample, num_being_sampled,
                                                             mock_gp_data.best_so_far, max_mc_iterations,
                                                             &normal_rng, &found_flag,
                                                             best_next_point_single_thread.data() + j*kDim*num_to_sample + kDim*kMaxNumThreads*num_to_sample);
    if (!found_flag) {
      ++total_errors;
    }
  }

  // now multithreaded to generate test data
  std::vector<double> best_next_point_multithread(kDim*num_to_sample);
  thread_schedule.max_num_threads = kMaxNumThreads;
  found_flag = false;
  ComputeOptimalPointsToSampleViaMultistartGradientDescent(*mock_gp_data.gaussian_process_ptr, gd_params,
                                                           domain, thread_schedule, starting_points.data(),
                                                           points_being_sampled.data(), kMaxNumThreads,
                                                           num_to_sample, num_being_sampled,
                                                           mock_gp_data.best_so_far,
                                                           max_mc_iterations, normal_rng_vec.data(),
                                                           &found_flag, best_next_point_multithread.data());
  if (!found_flag) {
    ++total_errors;
  }

  // best_next_point_multithread must be PRECISELY one of the points determined by single threaded runs
  double error[kMaxNumThreads*kMaxNumThreads];
  for (int i = 0; i < kMaxNumThreads; ++i) {
    for (int j = 0; j < kMaxNumThreads; ++j) {
      error[i*kMaxNumThreads + j] = 0.0;
      for (int k = 0; k < num_to_sample; ++k) {
        for (int d = 0; d < kDim; ++d) {
          error[i*kMaxNumThreads + j] += std::fabs(best_next_point_multithread[k*kDim + d] -
                                                   best_next_point_single_thread[i*kDim*kMaxNumThreads*num_to_sample +
                                                                                 j*kDim*num_to_sample + k*kDim + d]);
        }
      }
    }
  }
  // normally double precision checks like this are bad
  // but here, we want to ensure that the multithreaded & singlethreaded paths executed THE EXACT SAME CODE IN THE SAME ORDER
  // and hence their results must be identical
  bool pass = false;
  for (int i = 0; i < kMaxNumThreads*kMaxNumThreads; ++i) {
    if (error[i] == 0.0) {
      pass = true;
      break;
    }
  }
  if (pass == false) {
    OL_PARTIAL_FAILURE_PRINTF("multi & single threaded results differ 1: ");
    PrintMatrix(error, 1, Square(kMaxNumThreads));
    ++total_errors;
  }

  // reset random state & flip the points & generators so that if thread 0 won before, thread 1 wins now (or vice versa)
  for (int j = 0; j < kMaxNumThreads; ++j) {
    normal_rng_vec[kMaxNumThreads-j-1].SetExplicitSeed(pi_array[j]);
  }

  std::vector<double> starting_points_flip(kDim*kMaxNumThreads*num_to_sample);
  for (int j = 0; j < kMaxNumThreads; ++j) {
    for (int k = 0; k < num_to_sample; ++k) {
      for (int d = 0; d < kDim; ++d) {
        starting_points_flip[(kMaxNumThreads-j-1)*kDim*num_to_sample + k*kDim + d] = starting_points[j*kDim*num_to_sample + k*kDim + d];
      }
    }
  }

  // check multithreaded results again
  found_flag = false;
  ComputeOptimalPointsToSampleViaMultistartGradientDescent(*mock_gp_data.gaussian_process_ptr, gd_params,
                                                           domain, thread_schedule, starting_points_flip.data(),
                                                           points_being_sampled.data(), kMaxNumThreads,
                                                           num_to_sample, num_being_sampled,
                                                           mock_gp_data.best_so_far,
                                                           max_mc_iterations, normal_rng_vec.data(),
                                                           &found_flag, best_next_point_multithread.data());
  if (!found_flag) {
    ++total_errors;
  }

  for (int i = 0; i < kMaxNumThreads; ++i) {
    for (int j = 0; j < kMaxNumThreads; ++j) {
      error[i*kMaxNumThreads + j] = 0.0;
      for (int k = 0; k < num_to_sample; ++k) {
        for (int d = 0; d < kDim; ++d) {
          error[i*kMaxNumThreads + j] += std::fabs(best_next_point_multithread[k*kDim + d] -
                                                   best_next_point_single_thread[i*kDim*kMaxNumThreads*num_to_sample +
                                                                                 j*kDim*num_to_sample + k*kDim + d]);
        }
      }
    }
  }
  // normally double precision checks like this are bad
  // but here, we want to ensure that the multithreaded & singlethreaded paths executed THE EXACT SAME CODE IN THE SAME ORDER
  // and hence their results must be identical
  pass = false;
  for (int i = 0; i < kMaxNumThreads*kMaxNumThreads; ++i) {
    if (error[i] == 0.0) {
      pass = true;
      break;
    }
  }
  if (pass == false) {
    OL_PARTIAL_FAILURE_PRINTF("multi & single threaded results differ 2: ");
    PrintMatrix(error, 1, Square(kMaxNumThreads));
    ++total_errors;
  }

  if (total_errors != 0) {
    OL_PARTIAL_FAILURE_PRINTF("Single/Multithreaded EI Optimization Consistency Check failed with %d errors\n", total_errors);
  } else {
    OL_PARTIAL_SUCCESS_PRINTF("Single/Multithreaded EI Optimization Consistency Check succeeded\n");
  }

  return total_errors;
}

namespace {  // contains tests of EI optimization

/*!\rst
  Test that EI optimization works as expected for the analytic or monte-carlo evaluator types on a TensorProductDomain.

  \param
    :ei_mode: which ei evaluator (analytic or monte-carlo) to use
  \return
    number of test failures (invalid results, unconverged results, etc.)
\endrst*/
OL_WARN_UNUSED_RESULT int ExpectedImprovementOptimizationTestCore(ExpectedImprovementEvaluationMode ei_mode) {
  using DomainType = TensorProductDomain;
  const int dim = 3;

  int total_errors = 0;
  int current_errors = 0;

  // gradient descent parameters
  const double gamma = 0.5;
  const double pre_mult = 1.4;
  const double max_relative_change = 1.0;
  const double tolerance = 1.0e-7;
  const int max_gradient_descent_steps = 1000;
  const int max_num_restarts = 10;
  const int num_steps_averaged = 0;
  const int num_multistarts = 20;
  GradientDescentParameters gd_params(num_multistarts, max_gradient_descent_steps,
                                      max_num_restarts, num_steps_averaged,
                                      gamma, pre_mult, max_relative_change, tolerance);

  // grid search parameters
  int num_grid_search_points = 10000;

  // 1,p-EI computation parameters
  const int num_to_sample = 1;
  int num_being_sampled = 0;
  int max_int_steps = 6000;

  // random number generators
  UniformRandomGenerator uniform_generator(314);
  boost::uniform_real<double> uniform_double_hyperparameter(0.4, 1.3);
  boost::uniform_real<double> uniform_double_lower_bound(-2.0, 0.5);
  boost::uniform_real<double> uniform_double_upper_bound(1.0, 2.5);

  const int64_t pi_array[] = {314, 3141, 31415, 314159, 3141592, 31415926, 314159265, 3141592653, 31415926535, 314159265359};
  static const int kMaxNumThreads = 4;
  ThreadSchedule thread_schedule(kMaxNumThreads, omp_sched_static);
  std::vector<NormalRNG> normal_rng_vec(kMaxNumThreads);
  for (int j = 0; j < kMaxNumThreads; ++j) {
    normal_rng_vec[j].SetExplicitSeed(pi_array[j]);
  }

  int num_sampled = 20;

  std::vector<double> noise_variance(num_sampled, 0.002);
  MockGaussianProcessPriorData<DomainType> mock_gp_data(SquareExponential(dim, 1.0, 1.0), noise_variance,
                                                        dim, num_sampled, uniform_double_lower_bound,
                                                        uniform_double_upper_bound,
                                                        uniform_double_hyperparameter,
                                                        &uniform_generator);

  // we will optimize over the expanded region
  std::vector<ClosedInterval> domain_bounds(mock_gp_data.domain_bounds);
  ExpandDomainBounds(1.5, &domain_bounds);
  DomainType domain(domain_bounds.data(), dim);

  // set up parallel experiments, if any
  if (ei_mode == ExpectedImprovementEvaluationMode::kAnalytic) {
    num_being_sampled = 0;
  } else {
    // using MC integration
    num_being_sampled = 2;

    gd_params.max_num_restarts = 3;
    gd_params.max_num_steps = 250;
    gd_params.tolerance = 1.0e-5;
  }
  std::vector<double> points_being_sampled(dim*num_being_sampled);

  if (ei_mode == ExpectedImprovementEvaluationMode::kMonteCarlo) {
    // generate two non-trivial parallel samples
    // picking these randomly could place them in regions where EI is 0, which means errors in the computation would
    // likely be masked (making for a bad test)
    bool found_flag = false;
    for (int j = 0; j < num_being_sampled; ++j) {
      ComputeOptimalPointsToSampleWithRandomStarts(*mock_gp_data.gaussian_process_ptr,
                                                   gd_params, domain, thread_schedule,
                                                   points_being_sampled.data(),
                                                   num_to_sample, j, mock_gp_data.best_so_far,
                                                   max_int_steps, &found_flag,
                                                   &uniform_generator, normal_rng_vec.data(),
                                                   points_being_sampled.data() + j*dim);
    }
    printf("setup complete, points_being_sampled:\n");
    PrintMatrixTrans(points_being_sampled.data(), num_being_sampled, dim);
  }

  // optimize EI
  bool found_flag = false;
  std::vector<double> grid_search_best_point(dim*num_to_sample);
  ComputeOptimalPointsToSampleViaLatinHypercubeSearch(*mock_gp_data.gaussian_process_ptr, domain,
                                                      thread_schedule, points_being_sampled.data(),
                                                      num_grid_search_points, num_to_sample,
                                                      num_being_sampled, mock_gp_data.best_so_far,
                                                      max_int_steps, &found_flag,
                                                      &uniform_generator, normal_rng_vec.data(),
                                                      grid_search_best_point.data());
  if (!found_flag) {
    ++total_errors;
  }

  std::vector<double> next_point(dim*num_to_sample);
  if (ei_mode == ExpectedImprovementEvaluationMode::kAnalytic) {
    found_flag = false;
    ComputeOptimalPointsToSampleWithRandomStarts(*mock_gp_data.gaussian_process_ptr, gd_params,
                                                 domain, thread_schedule, points_being_sampled.data(),
                                                 num_to_sample, num_being_sampled,
                                                 mock_gp_data.best_so_far, max_int_steps,
                                                 &found_flag,
                                                 &uniform_generator, normal_rng_vec.data(),
                                                 next_point.data());
    if (!found_flag) {
      ++total_errors;
    }
  } else {
    int num_multistarts_mc = 8;
    gd_params.num_multistarts = num_multistarts_mc;
    found_flag = false;
    std::vector<double> initial_guesses(num_multistarts_mc*dim);
    domain.GenerateUniformPointsInDomain(num_multistarts_mc - 1, &uniform_generator, initial_guesses.data() + dim);
    std::copy(grid_search_best_point.begin(), grid_search_best_point.end(), initial_guesses.begin());

    ComputeOptimalPointsToSampleViaMultistartGradientDescent(*mock_gp_data.gaussian_process_ptr,
                                                             gd_params, domain, thread_schedule,
                                                             initial_guesses.data(),
                                                             points_being_sampled.data(),
                                                             num_multistarts_mc, num_to_sample,
                                                             num_being_sampled,
                                                             mock_gp_data.best_so_far,
                                                             max_int_steps,
                                                             normal_rng_vec.data(), &found_flag,
                                                             next_point.data());
    if (!found_flag) {
      ++total_errors;
    }
  }

  printf("next best point  : "); PrintMatrixTrans(next_point.data(), num_to_sample, dim);
  printf("grid search point: "); PrintMatrixTrans(grid_search_best_point.data(), num_to_sample, dim);

  // results
  double ei_optimized, ei_grid_search;
  std::vector<double> grad_ei(dim*num_to_sample);

  // set up evaluators and state to check results
  double tolerance_result = tolerance;
  bool configure_for_gradients = true;
  if (ei_mode == ExpectedImprovementEvaluationMode::kAnalytic) {
    OnePotentialSampleExpectedImprovementEvaluator ei_evaluator(*mock_gp_data.gaussian_process_ptr,
                                                                mock_gp_data.best_so_far);
    OnePotentialSampleExpectedImprovementEvaluator::StateType ei_state(ei_evaluator, next_point.data(),
                                                                       configure_for_gradients);

    ei_optimized = ei_evaluator.ComputeExpectedImprovement(&ei_state);
    ei_evaluator.ComputeGradExpectedImprovement(&ei_state, grad_ei.data());

    ei_state.SetCurrentPoint(ei_evaluator, grid_search_best_point.data());
    ei_grid_search = ei_evaluator.ComputeExpectedImprovement(&ei_state);
  } else {
    max_int_steps = 1000000;
    tolerance_result = 2.0e-3;  // reduce b/c we cannot achieve full accuracy in the monte-carlo case
    // while still having this test run in a reasonable amt of time
    ExpectedImprovementEvaluator ei_evaluator(*mock_gp_data.gaussian_process_ptr,
                                              max_int_steps, mock_gp_data.best_so_far);
    ExpectedImprovementEvaluator::StateType ei_state(ei_evaluator, next_point.data(),
                                                     points_being_sampled.data(), num_to_sample,
                                                     num_being_sampled, configure_for_gradients,
                                                     normal_rng_vec.data());

    ei_optimized = ei_evaluator.ComputeExpectedImprovement(&ei_state);
    ei_evaluator.ComputeGradExpectedImprovement(&ei_state, grad_ei.data());

    ei_state.SetCurrentPoint(ei_evaluator, grid_search_best_point.data());
    ei_grid_search = ei_evaluator.ComputeExpectedImprovement(&ei_state);
  }

  printf("optimized EI: %.18E, grid_search_EI: %.18E\n", ei_optimized, ei_grid_search);
  printf("grad_EI: "); PrintMatrixTrans(grad_ei.data(), num_to_sample, dim);

  if (ei_optimized < ei_grid_search) {
    ++total_errors;
  }

  current_errors = 0;
  for (const auto& entry : grad_ei) {
    if (!CheckDoubleWithinRelative(entry, 0.0, tolerance_result)) {
      ++current_errors;
    }
  }
  total_errors += current_errors;

  return total_errors;
}

/*!\rst
  Test that EI optimization works as expected for the analytic or monte-carlo evaluator types on a SimplexIntersectTensorProductDomain.

  \param
    :ei_mode: which ei evaluator (analytic or monte-carlo) to use
  \return
    number of test failures (invalid results, unconverged results, etc.)
\endrst*/
OL_WARN_UNUSED_RESULT int ExpectedImprovementOptimizationSimplexTestCore(ExpectedImprovementEvaluationMode ei_mode) {
  using DomainType = SimplexIntersectTensorProductDomain;
  const int dim = 3;

  int total_errors = 0;
  int current_errors = 0;

  // gradient descent parameters
  const double gamma = 0.8;
  const double pre_mult = 0.02;
  const double max_relative_change = 0.99;
  const double tolerance = 1.0e-7;
  const int max_gradient_descent_steps = 1000;
  const int max_num_restarts = 10;
  const int num_steps_averaged = 0;
  const int num_multistarts = 20;
  GradientDescentParameters gd_params(num_multistarts, max_gradient_descent_steps,
                                      max_num_restarts, num_steps_averaged, gamma, pre_mult,
                                      max_relative_change, tolerance);

  // grid search parameters
  int num_grid_search_points = 10000;

  // 1,p-EI computation parameters
  const int num_to_sample = 1;
  int num_being_sampled = 0;
  int max_int_steps = 6000;

  // random number generators
  UniformRandomGenerator uniform_generator(314);
  boost::uniform_real<double> uniform_double_hyperparameter(0.05, 0.1);
  boost::uniform_real<double> uniform_double_lower_bound(0.11, 0.15);
  boost::uniform_real<double> uniform_double_upper_bound(0.3, 0.35);

  const int64_t pi_array[] = {314, 3141, 31415, 314159, 3141592, 31415926, 314159265, 3141592653, 31415926535, 314159265359};
  static const int kMaxNumThreads = 4;
  ThreadSchedule thread_schedule(kMaxNumThreads, omp_sched_static);
  std::vector<NormalRNG> normal_rng_vec(kMaxNumThreads);
  for (int j = 0; j < kMaxNumThreads; ++j) {
    normal_rng_vec[j].SetExplicitSeed(pi_array[j]);
  }

  int num_sampled = 20;

  std::vector<double> noise_variance(num_sampled, 0.002);
  MockGaussianProcessPriorData<DomainType> mock_gp_data(SquareExponential(dim, 1.0, 1.0),
                                                        noise_variance, dim, num_sampled,
                                                        uniform_double_lower_bound,
                                                        uniform_double_upper_bound,
                                                        uniform_double_hyperparameter,
                                                        &uniform_generator);

  // we will optimize over the expanded region
  std::vector<ClosedInterval> domain_bounds(mock_gp_data.domain_bounds);
  ExpandDomainBounds(2.2, &domain_bounds);
  // intersect domain with bounding box of unit simplex
  for (auto& interval : domain_bounds) {
    interval.min = std::fmax(interval.min, 0.0);
    interval.max = std::fmin(interval.max, 1.0);
  }
  DomainType domain(domain_bounds.data(), dim);

  // set up parallel experiments, if any
  if (ei_mode == ExpectedImprovementEvaluationMode::kAnalytic) {
    num_being_sampled = 0;
    gd_params.pre_mult = 1.0;
  } else {
    // using MC integration
    num_being_sampled = 2;

    gd_params.max_num_restarts = 4;
    gd_params.max_num_steps = 250;
    gd_params.tolerance = 1.0e-4;
  }
  std::vector<double> points_being_sampled(dim*num_being_sampled);

  if (ei_mode == ExpectedImprovementEvaluationMode::kMonteCarlo) {
    // generate two non-trivial parallel samples
    // picking these randomly could place them in regions where EI is 0, which means errors in the computation would
    // likely be masked (making for a bad test)
    bool found_flag = false;
    for (int j = 0; j < num_being_sampled; ++j) {
      ComputeOptimalPointsToSampleWithRandomStarts(*mock_gp_data.gaussian_process_ptr,
                                                   gd_params, domain, thread_schedule,
                                                   points_being_sampled.data(),
                                                   num_to_sample, j, mock_gp_data.best_so_far,
                                                   max_int_steps, &found_flag,
                                                   &uniform_generator, normal_rng_vec.data(),
                                                   points_being_sampled.data() + j*dim);
    }
    printf("setup complete, points_being_sampled:\n");
    PrintMatrixTrans(points_being_sampled.data(), num_being_sampled, dim);
  }

  // optimize EI
  bool found_flag = false;
  std::vector<double> grid_search_best_point(dim*num_to_sample);
  ComputeOptimalPointsToSampleViaLatinHypercubeSearch(*mock_gp_data.gaussian_process_ptr, domain,
                                                      thread_schedule, points_being_sampled.data(),
                                                      num_grid_search_points, num_to_sample,
                                                      num_being_sampled, mock_gp_data.best_so_far,
                                                      max_int_steps, &found_flag,
                                                      &uniform_generator, normal_rng_vec.data(),
                                                      grid_search_best_point.data());
  if (!found_flag) {
    ++total_errors;
  }

  std::vector<double> next_point(dim*num_to_sample);
  if (ei_mode == ExpectedImprovementEvaluationMode::kAnalytic) {
    found_flag = false;
    ComputeOptimalPointsToSampleWithRandomStarts(*mock_gp_data.gaussian_process_ptr, gd_params,
                                                 domain, thread_schedule,
                                                 points_being_sampled.data(), num_to_sample,
                                                 num_being_sampled, mock_gp_data.best_so_far,
                                                 max_int_steps, &found_flag,
                                                 &uniform_generator, normal_rng_vec.data(),
                                                 next_point.data());
    if (!found_flag) {
      ++total_errors;
    }
  } else {
    int num_multistarts_mc = 6;
    gd_params.num_multistarts = num_multistarts_mc;
    found_flag = false;
    std::vector<double> initial_guesses(num_multistarts_mc*dim);
    int num_points_actual = domain.GenerateUniformPointsInDomain(num_multistarts_mc, &uniform_generator, initial_guesses.data());
    if (num_points_actual != num_multistarts_mc) {
      ++total_errors;
    }
    std::copy(grid_search_best_point.begin(), grid_search_best_point.end(), initial_guesses.begin());

    ComputeOptimalPointsToSampleViaMultistartGradientDescent(*mock_gp_data.gaussian_process_ptr, gd_params,
                                                             domain, thread_schedule, initial_guesses.data(),
                                                             points_being_sampled.data(), num_multistarts_mc,
                                                             num_to_sample, num_being_sampled,
                                                             mock_gp_data.best_so_far, max_int_steps,
                                                             normal_rng_vec.data(),
                                                             &found_flag, next_point.data());
    if (!found_flag) {
      ++total_errors;
    }
  }

  printf("next best point  : "); PrintMatrixTrans(next_point.data(), num_to_sample, dim);
  printf("grid search point: "); PrintMatrixTrans(grid_search_best_point.data(), num_to_sample, dim);

  // results
  double ei_optimized, ei_grid_search;
  std::vector<double> grad_ei(dim*num_to_sample);

  // set up evaluators and state to check results
  double tolerance_result = tolerance;
  bool configure_for_gradients = true;
  if (ei_mode == ExpectedImprovementEvaluationMode::kAnalytic) {
    OnePotentialSampleExpectedImprovementEvaluator ei_evaluator(*mock_gp_data.gaussian_process_ptr,
                                                                mock_gp_data.best_so_far);
    OnePotentialSampleExpectedImprovementEvaluator::StateType ei_state(ei_evaluator, next_point.data(),
                                                                       configure_for_gradients);

    ei_optimized = ei_evaluator.ComputeExpectedImprovement(&ei_state);
    ei_evaluator.ComputeGradExpectedImprovement(&ei_state, grad_ei.data());

    ei_state.SetCurrentPoint(ei_evaluator, grid_search_best_point.data());
    ei_grid_search = ei_evaluator.ComputeExpectedImprovement(&ei_state);
  } else {
    max_int_steps = 1000000;
    tolerance_result = 3.5e-3;  // reduce b/c we cannot achieve full accuracy in the monte-carlo case
    // while still having this test run in a reasonable amt of time
    ExpectedImprovementEvaluator ei_evaluator(*mock_gp_data.gaussian_process_ptr,
                                              max_int_steps, mock_gp_data.best_so_far);
    ExpectedImprovementEvaluator::StateType ei_state(ei_evaluator, next_point.data(),
                                                     points_being_sampled.data(),
                                                     num_to_sample, num_being_sampled,
                                                     configure_for_gradients, normal_rng_vec.data());

    ei_optimized = ei_evaluator.ComputeExpectedImprovement(&ei_state);
    ei_evaluator.ComputeGradExpectedImprovement(&ei_state, grad_ei.data());

    ei_state.SetCurrentPoint(ei_evaluator, grid_search_best_point.data());
    ei_grid_search = ei_evaluator.ComputeExpectedImprovement(&ei_state);
  }

  printf("optimized EI: %.18E, grid_search_EI: %.18E\n", ei_optimized, ei_grid_search);
  printf("grad_EI: "); PrintMatrixTrans(grad_ei.data(), num_to_sample, dim);

  if (ei_optimized < ei_grid_search) {
    ++total_errors;
  }

  current_errors = 0;
  for (const auto& entry : grad_ei) {
    if (!CheckDoubleWithinRelative(entry, 0.0, tolerance_result)) {
      ++current_errors;
    }
  }
  total_errors += current_errors;

  return total_errors;
}

}  // end unnamed namespace

/*!\rst
  At the moment, this test is very bare-bones.  It checks:

  1. method succeeds
  2. points returned are all inside the specified domain
  3. points returned are not within epsilon of each other (i.e., distinct)
  4. result of gradient-descent optimization is *no worse* than result of a random search
  5. final grad EI is sufficiently small

  The test sets up a toy problem by repeatedly drawing from a GP with made-up hyperparameters.
  Then it runs EI optimization, attempting to sample 3 points simultaneously.
\endrst*/
int ExpectedImprovementOptimizationMultipleSamplesTest() {
  using DomainType = TensorProductDomain;
  const int dim = 3;

  int total_errors = 0;
  int current_errors = 0;

  // gradient descent parameters
  const double gamma = 0.5;
  const double pre_mult = 1.5;
  const double max_relative_change = 1.0;
  const double tolerance = 1.0e-5;
  const int max_gradient_descent_steps = 250;
  const int max_num_restarts = 3;
  const int num_steps_averaged = 0;
  const int num_multistarts = 20;
  GradientDescentParameters gd_params(num_multistarts, max_gradient_descent_steps,
                                      max_num_restarts, num_steps_averaged, gamma, pre_mult,
                                      max_relative_change, tolerance);

  // grid search parameters
  int num_grid_search_points = 1000;

  // q,p-EI computation parameters
  const int num_to_sample = 3;
  const int num_being_sampled = 0;

  std::vector<double> points_being_sampled(dim*num_being_sampled);
  int max_int_steps = 6000;

  // random number generators
  UniformRandomGenerator uniform_generator(314);
  boost::uniform_real<double> uniform_double_hyperparameter(0.4, 1.3);
  boost::uniform_real<double> uniform_double_lower_bound(-2.0, 0.5);
  boost::uniform_real<double> uniform_double_upper_bound(1.0, 2.5);

  const int64_t pi_array[] = {314, 3141, 31415, 314159, 3141592, 31415926, 314159265, 3141592653, 31415926535, 314159265359};
  static const int kMaxNumThreads = 4;
  ThreadSchedule thread_schedule(kMaxNumThreads, omp_sched_static);
  std::vector<NormalRNG> normal_rng_vec(kMaxNumThreads);
  for (int j = 0; j < kMaxNumThreads; ++j) {
    normal_rng_vec[j].SetExplicitSeed(pi_array[j]);
  }

  const int num_sampled = 20;
  std::vector<double> noise_variance(num_sampled, 0.002);
  MockGaussianProcessPriorData<DomainType> mock_gp_data(SquareExponential(dim, 1.0, 1.0),
                                                        noise_variance, dim, num_sampled,
                                                        uniform_double_lower_bound,
                                                        uniform_double_upper_bound,
                                                        uniform_double_hyperparameter,
                                                        &uniform_generator);

  // we will optimize over the expanded region
  std::vector<ClosedInterval> domain_bounds(mock_gp_data.domain_bounds);
  ExpandDomainBounds(1.5, &domain_bounds);
  DomainType domain(domain_bounds.data(), dim);

  // optimize EI using grid search to set the baseline
  bool found_flag = false;
  std::vector<double> grid_search_best_point_set(dim*num_to_sample);
  ComputeOptimalPointsToSampleViaLatinHypercubeSearch(*mock_gp_data.gaussian_process_ptr, domain,
                                                      thread_schedule, points_being_sampled.data(),
                                                      num_grid_search_points, num_to_sample,
                                                      num_being_sampled, mock_gp_data.best_so_far,
                                                      max_int_steps, &found_flag,
                                                      &uniform_generator, normal_rng_vec.data(),
                                                      grid_search_best_point_set.data());
  if (!found_flag) {
    ++total_errors;
  }

  // optimize EI using gradient descent
  found_flag = false;
  bool lhc_search_only = false;
  std::vector<double> best_points_to_sample(dim*num_to_sample);
  ComputeOptimalPointsToSample(*mock_gp_data.gaussian_process_ptr, gd_params, domain,
                               thread_schedule, points_being_sampled.data(),
                               num_to_sample, num_being_sampled, mock_gp_data.best_so_far,
                               max_int_steps, lhc_search_only,
                               num_grid_search_points, &found_flag, &uniform_generator,
                               normal_rng_vec.data(), best_points_to_sample.data());
  if (!found_flag) {
    ++total_errors;
  }

  // check points are in domain
  RepeatedDomain<DomainType> repeated_domain(domain, num_to_sample);
  if (!repeated_domain.CheckPointInside(best_points_to_sample.data())) {
    ++current_errors;
  }
#ifdef OL_ERROR_PRINT
  if (current_errors != 0) {
    OL_ERROR_PRINTF("ERROR: points were not in domain!  points:\n");
    PrintMatrixTrans(best_points_to_sample.data(), num_to_sample, dim);
    OL_ERROR_PRINTF("domain:\n");
    PrintDomainBounds(domain_bounds.data(), dim);
  }
#endif
  total_errors += current_errors;

  // check points are distinct; points within tolerance are considered non-distinct
  const double distinct_point_tolerance = 1.0e-5;
  current_errors = CheckPointsAreDistinct(best_points_to_sample.data(), num_to_sample, dim, distinct_point_tolerance);
#ifdef OL_ERROR_PRINT
  if (current_errors != 0) {
    OL_ERROR_PRINTF("ERROR: points were not distinct!  points:\n");
    PrintMatrixTrans(best_points_to_sample.data(), num_to_sample, dim);
  }
#endif
  total_errors += current_errors;

  // results
  double ei_optimized, ei_grid_search;
  std::vector<double> grad_ei(dim*num_to_sample);

  // set up evaluators and state to check results
  double tolerance_result = tolerance;
  {
    max_int_steps = 1000000;  // evaluate the final results with high accuracy
    tolerance_result = 2.0e-3;  // reduce b/c we cannot achieve full accuracy in the monte-carlo case
    // while still having this test run in a reasonable amt of time
    bool configure_for_gradients = true;
    ExpectedImprovementEvaluator ei_evaluator(*mock_gp_data.gaussian_process_ptr,
                                              max_int_steps, mock_gp_data.best_so_far);
    ExpectedImprovementEvaluator::StateType ei_state(ei_evaluator, best_points_to_sample.data(),
                                                     points_being_sampled.data(), num_to_sample,
                                                     num_being_sampled, configure_for_gradients,
                                                     normal_rng_vec.data());

    ei_optimized = ei_evaluator.ComputeExpectedImprovement(&ei_state);
    ei_evaluator.ComputeGradExpectedImprovement(&ei_state, grad_ei.data());

    ExpectedImprovementEvaluator::StateType ei_state_grid_search(ei_evaluator,
                                                                 grid_search_best_point_set.data(),
                                                                 points_being_sampled.data(), num_to_sample,
                                                                 num_being_sampled, configure_for_gradients,
                                                                 normal_rng_vec.data());
    ei_grid_search = ei_evaluator.ComputeExpectedImprovement(&ei_state_grid_search);
  }

  printf("optimized EI: %.18E, grid_search_EI: %.18E\n", ei_optimized, ei_grid_search);
  printf("grad_EI: "); PrintMatrixTrans(grad_ei.data(), num_to_sample, dim);

  if (ei_optimized < ei_grid_search) {
    ++total_errors;
  }

  current_errors = 0;
  for (const auto& entry : grad_ei) {
    if (!CheckDoubleWithinRelative(entry, 0.0, tolerance_result)) {
      ++current_errors;
    }
  }
  total_errors += current_errors;

  return total_errors;
}

int EvaluateEIAtPointListTest() {
  using DomainType = TensorProductDomain;
  const int dim = 3;

  int total_errors = 0;

  // grid search parameters
  int num_grid_search_points = 100000;

  // q,p-EI computation parameters
  int num_to_sample = 1;
  int num_being_sampled = 0;
  int max_int_steps = 0;

  // random number generators
  UniformRandomGenerator uniform_generator(314);
  boost::uniform_real<double> uniform_double_hyperparameter(0.4, 1.3);
  boost::uniform_real<double> uniform_double_lower_bound(-2.0, 0.5);
  boost::uniform_real<double> uniform_double_upper_bound(2.0, 3.5);

  const int64_t pi_array[] = {314, 3141, 31415, 314159, 3141592, 31415926, 314159265, 3141592653, 31415926535, 314159265359};
  static const int kMaxNumThreads = 4;
  ThreadSchedule thread_schedule(kMaxNumThreads, omp_sched_static);
  std::vector<NormalRNG> normal_rng_vec(kMaxNumThreads);
  for (int j = 0; j < kMaxNumThreads; ++j) {
    normal_rng_vec[j].SetExplicitSeed(pi_array[j]);
  }

  int num_sampled = 20;  // arbitrary
  std::vector<double> noise_variance(num_sampled, 0.002);
  MockGaussianProcessPriorData<DomainType> mock_gp_data(SquareExponential(dim, 1.0, 1.0), noise_variance, dim, num_sampled, uniform_double_lower_bound, uniform_double_upper_bound, uniform_double_hyperparameter, &uniform_generator);

  // no parallel experiments
  num_being_sampled = 0;
  std::vector<double> points_being_sampled(dim*num_being_sampled);

  bool found_flag = false;
  std::vector<double> grid_search_best_point(dim*num_to_sample);
  std::vector<double> function_values(num_grid_search_points);
  std::vector<double> initial_guesses(dim*num_to_sample*num_grid_search_points);
  num_grid_search_points = mock_gp_data.domain_ptr->GenerateUniformPointsInDomain(num_grid_search_points, &uniform_generator, initial_guesses.data());

  EvaluateEIAtPointList(*mock_gp_data.gaussian_process_ptr, thread_schedule, initial_guesses.data(),
                        points_being_sampled.data(), num_grid_search_points, num_to_sample,
                        num_being_sampled, mock_gp_data.best_so_far, max_int_steps, &found_flag,
                        normal_rng_vec.data(), function_values.data(), grid_search_best_point.data());
  if (!found_flag) {
    ++total_errors;
  }

  // find the max function_value and the index at which it occurs
  auto max_value_ptr = std::max_element(function_values.begin(), function_values.end());
  auto max_index = std::distance(function_values.begin(), max_value_ptr);

  // check that EvaluateEIAtPointList found the right point
  for (int i = 0; i < dim*num_to_sample; ++i) {
    if (!CheckDoubleWithin(grid_search_best_point[i], initial_guesses[max_index*dim + i], 0.0)) {
      ++total_errors;
    }
  }

  // now check multi-threaded & single threaded give the same result
  {
    std::vector<double> grid_search_best_point_single_thread(dim*num_to_sample);
    std::vector<double> function_values_single_thread(num_grid_search_points);
    ThreadSchedule single_thread_schedule(1, omp_sched_static);
    found_flag = false;
    EvaluateEIAtPointList(*mock_gp_data.gaussian_process_ptr, single_thread_schedule,
                          initial_guesses.data(), points_being_sampled.data(),
                          num_grid_search_points, num_to_sample, num_being_sampled,
                          mock_gp_data.best_so_far, max_int_steps,
                          &found_flag, normal_rng_vec.data(),
                          function_values_single_thread.data(),
                          grid_search_best_point_single_thread.data());

    // check against multi-threaded result matches single
    for (int i = 0; i < dim*num_to_sample; ++i) {
      if (!CheckDoubleWithin(grid_search_best_point[i], grid_search_best_point_single_thread[i], 0.0)) {
        ++total_errors;
      }
    }

    // check all function values match too
    for (int i = 0; i < num_grid_search_points; ++i) {
      if (!CheckDoubleWithin(function_values[i], function_values_single_thread[i], 0.0)) {
        ++total_errors;
      }
    }
  }

  return total_errors;
}

int ExpectedImprovementOptimizationTest(DomainTypes domain_type, ExpectedImprovementEvaluationMode ei_mode) {
  switch (domain_type) {
    case DomainTypes::kTensorProduct: {
      switch (ei_mode) {
        case ExpectedImprovementEvaluationMode::kAnalytic:
        case ExpectedImprovementEvaluationMode::kMonteCarlo: {
          return ExpectedImprovementOptimizationTestCore(ei_mode);
        }
        default: {
          OL_ERROR_PRINTF("%s: INVALID ei_mode choice: %d\n", OL_CURRENT_FUNCTION_NAME, ei_mode);
          return 1;
        }
      }  // end switch over ei_mode
    }  // end case kTensorProduct
    case DomainTypes::kSimplex: {
      switch (ei_mode) {
        case ExpectedImprovementEvaluationMode::kAnalytic:
        case ExpectedImprovementEvaluationMode::kMonteCarlo: {
          return ExpectedImprovementOptimizationSimplexTestCore(ei_mode);
        }
        default: {
          OL_ERROR_PRINTF("%s: INVALID ei_mode choice: %d\n", OL_CURRENT_FUNCTION_NAME, ei_mode);
          return 1;
        }
      }  // end switch over ei_mode
    }  // end case kSimplex
    default: {
      OL_ERROR_PRINTF("%s: INVALID domain_type choice: %d\n", OL_CURRENT_FUNCTION_NAME, domain_type);
      return 1;
    }
  }  // end switch over domain_type
}

}  // end namespace optimal_learning
