/*!
  \file gpp_expected_improvement_gpu_test.hpp
  \rst
  Functions for testing expected improvement functions on GPU.

  Tests are broken into three main groups:

  * ping (unit) tests for EI outputs
  * consistency test against analytical 1,0-EI result
  * compare with CPU(MC) results

  The ping tests are set up the same way as the ping tests in gpp_covariance_test; using the function evaluator and ping
  framework defined in gpp_test_utils.

\endrst*/

#ifndef MOE_OPTIMAL_LEARNING_CPP_GPP_EXPECTED_IMPROVEMENT_GPU_TEST_HPP_
#define MOE_OPTIMAL_LEARNING_CPP_GPP_EXPECTED_IMPROVEMENT_GPU_TEST_HPP_

#include <vector>

#include "gpp_common.hpp"
#include "gpp_domain.hpp"
#include "gpp_test_utils.hpp"
#include "gpp_exception.hpp"
#include "gpp_expected_improvement_gpu.hpp"
#include "gpp_math.hpp"

namespace optimal_learning {
/*!\rst
  Supports evaluating the expected improvement on GPU, CudaExpectedImprovementEvaluator::ComputeExpectedImprovement() and
  its gradient, CudaExpectedImprovementEvaluator::ComputeGradExpectedImprovement()

  The gradient is taken wrt ``points_to_sample[dim]``, so this is the ``input_matrix``, ``X_{d,i}``.
  The other inputs to EI are not differentiated against, so they are taken as input and stored by the constructor.

  The output of EI is a scalar.
\endrst*/
class PingCudaExpectedImprovement final : public PingableMatrixInputVectorOutputInterface {
 public:
  constexpr static char const * const kName = "EI with MC on GPU";

  PingCudaExpectedImprovement(double const * restrict lengths, double const * restrict points_being_sampled, double const * restrict points_sampled, double const * restrict points_sampled_value, double alpha, double best_so_far, int dim, int num_to_sample, int num_being_sampled, int num_sampled, int num_mc_iter) OL_NONNULL_POINTERS;

  virtual void GetInputSizes(int * num_rows, int * num_cols) const noexcept override OL_NONNULL_POINTERS;

  virtual int GetGradientsSize() const noexcept override OL_WARN_UNUSED_RESULT;

  virtual int GetOutputSize() const noexcept override OL_WARN_UNUSED_RESULT;

  virtual void EvaluateAndStoreAnalyticGradient(double const * restrict points_to_sample, double * restrict gradients) noexcept override OL_NONNULL_POINTERS_LIST(2);

  virtual double GetAnalyticGradient(int row_index, int OL_UNUSED(column_index), int OL_UNUSED(output_index)) const override OL_WARN_UNUSED_RESULT;

  virtual void EvaluateFunction(double const * restrict points_to_sample, double * restrict function_values) const noexcept override OL_NONNULL_POINTERS;

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
  CudaExpectedImprovementEvaluator ei_evaluator_;

  OL_DISALLOW_DEFAULT_AND_COPY_AND_ASSIGN(PingCudaExpectedImprovement);
};

/*!\rst
  Tests that the general EI + grad EI computation (using MC integration) is consistent
  with the special analytic case of EI when there is only *ONE* potential point
  to sample.

  \return
    number of test failures: 0 if all is working well.
\endrst*/
OL_WARN_UNUSED_RESULT int RunCudaEIConsistencyTests();

/*!\rst
  Tests that the general EI + grad EI computation on CPU (using MC integration) is consistent
  with the computation on GPU.

  \return
    number of test failures: 0 if all is working well.
\endrst*/
OL_WARN_UNUSED_RESULT int RunCudaEIvsCpuEI();

}  // end namespace optimal_learning
#endif  // MOE_OPTIMAL_LEARNING_CPP_GPP_EXPECTED_IMPROVEMENT_GPU_TEST_HPP_

