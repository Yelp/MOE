/*!
  \file gpp_expected_improvement_gpu.cpp
  \rst
  This file contains implementations of GPU related functions. They are actually C++ wrappers for
  CUDA C functions defined in gpu/gpp_cuda_math.cu.
\endrst*/

#include "gpp_expected_improvement_gpu.hpp"

#include <stdint.h>

#include <algorithm>
#include <vector>

#include "gpp_common.hpp"
#include "gpp_exception.hpp"
#include "gpp_logging.hpp"
#include "gpp_math.hpp"
#include "gpp_random.hpp"

#ifdef OL_GPU_ENABLED

#include "gpu/gpp_cuda_math.hpp"
#include "driver_types.h"
#include "cuda_runtime.h"

#endif

namespace optimal_learning {

#ifdef OL_GPU_ENABLED

CudaDevicePointer::CudaDevicePointer(int num_doubles_in) : num_doubles(num_doubles_in) {
  if (num_doubles_in > 0) {
      CudaError error = CudaAllocateMemForDoubleVector(num_doubles, &ptr);
      if (error.err != cudaSuccess) {
          ptr = nullptr;
          ThrowException(OptimalLearningCudaException(error));
      }
  } else {
      ptr = nullptr;
  }
}

CudaDevicePointer::~CudaDevicePointer() {
    CudaFreeMem(ptr);
}

OptimalLearningCudaException::OptimalLearningCudaException(const CudaError& error)
      : OptimalLearningException(error.file_and_line_info, error.func_info, cudaGetErrorString(error.err)) {
}

double CudaExpectedImprovementEvaluator::ComputeExpectedImprovement(StateType * ei_state) const {
  double EI_val;
  int num_union = ei_state->num_union;
  gaussian_process_->ComputeMeanOfPoints(ei_state->points_to_sample_state, ei_state->to_sample_mean.data());
  gaussian_process_->ComputeVarianceOfPoints(&(ei_state->points_to_sample_state), ei_state->cholesky_to_sample_var.data());
  int leading_minor_index = ComputeCholeskyFactorL(num_union, ei_state->cholesky_to_sample_var.data());
  if (unlikely(leading_minor_index != 0)) {
    OL_THROW_EXCEPTION(SingularMatrixException,
                       "GP-Variance matrix singular. Check for duplicate points_to_sample/being_sampled or points_to_sample/being_sampled duplicating points_sampled with 0 noise.",
                       ei_state->cholesky_to_sample_var.data(), num_union, leading_minor_index);
  }
  uint64_t seed_in = (ei_state->uniform_rng->GetEngine())();
  OL_CUDA_ERROR_THROW(CudaGetEI(ei_state->to_sample_mean.data(), ei_state->cholesky_to_sample_var.data(),
                                num_union, num_mc_, seed_in, best_so_far_, ei_state->configure_for_test,
                                ei_state->random_number_ei.data(), &EI_val, ei_state->gpu_mu.ptr,
                                ei_state->gpu_chol_var.ptr, ei_state->gpu_random_number_ei.ptr,
                                ei_state->gpu_ei_storage.ptr));
  return EI_val;
}

void CudaExpectedImprovementEvaluator::ComputeGradExpectedImprovement(StateType * ei_state,
                                                                      double * restrict grad_ei) const {
  if (ei_state->num_derivatives == 0) {
    OL_THROW_EXCEPTION(OptimalLearningException, "configure_for_gradients set to false, gradient computation is disabled!");
  }
  const int num_union = ei_state->num_union;
  const int num_to_sample = ei_state->num_to_sample;
  gaussian_process_->ComputeMeanOfPoints(ei_state->points_to_sample_state, ei_state->to_sample_mean.data());
  gaussian_process_->ComputeGradMeanOfPoints(ei_state->points_to_sample_state, ei_state->grad_mu.data());
  gaussian_process_->ComputeVarianceOfPoints(&(ei_state->points_to_sample_state), ei_state->cholesky_to_sample_var.data());
  int leading_minor_index = ComputeCholeskyFactorL(num_union, ei_state->cholesky_to_sample_var.data());
  if (unlikely(leading_minor_index != 0)) {
    OL_THROW_EXCEPTION(SingularMatrixException,
                       "GP-Variance matrix singular. Check for duplicate points_to_sample/being_sampled or points_to_sample/being_sampled duplicating points_sampled with 0 noise.",
                       ei_state->cholesky_to_sample_var.data(), num_union, leading_minor_index);
  }

  gaussian_process_->ComputeGradCholeskyVarianceOfPoints(&(ei_state->points_to_sample_state),
                                                         ei_state->cholesky_to_sample_var.data(),
                                                         ei_state->grad_chol_decomp.data());
  uint64_t seed_in = (ei_state->uniform_rng->GetEngine())();

  OL_CUDA_ERROR_THROW(CudaGetGradEI(ei_state->to_sample_mean.data(), ei_state->cholesky_to_sample_var.data(),
                                    ei_state->grad_mu.data(), ei_state->grad_chol_decomp.data(), num_union,
                                    num_to_sample, dim_, num_mc_, seed_in, best_so_far_, ei_state->configure_for_test,
                                    ei_state->random_number_grad_ei.data(), grad_ei, ei_state->gpu_mu.ptr,
                                    ei_state->gpu_chol_var.ptr, ei_state->gpu_grad_mu.ptr, ei_state->gpu_grad_chol_var.ptr,
                                    ei_state->gpu_random_number_grad_ei.ptr, ei_state->gpu_grad_ei_storage.ptr));
}

void CudaExpectedImprovementEvaluator::SetupGPU(int devID) {
  OL_CUDA_ERROR_THROW(CudaSetDevice(devID));
}

CudaExpectedImprovementEvaluator::CudaExpectedImprovementEvaluator(const GaussianProcess& gaussian_process_in,
                                                                   int num_mc_in, double best_so_far, int devID_in)
      : dim_(gaussian_process_in.dim()),
        num_mc_(num_mc_in),
        best_so_far_(best_so_far),
        gaussian_process_(&gaussian_process_in) {
    SetupGPU(devID_in);
  }

CudaExpectedImprovementEvaluator::~CudaExpectedImprovementEvaluator() {
  cudaDeviceReset();
}

CudaExpectedImprovementState::CudaExpectedImprovementState(const EvaluatorType& ei_evaluator,
                                                           double const * restrict points_to_sample,
                                                           double const * restrict points_being_sampled,
                                                           int num_to_sample_in, int num_being_sampled_in,
                                                           bool configure_for_gradients,
                                                           UniformRandomGenerator * uniform_rng_in)
    : dim(ei_evaluator.dim()),
      num_to_sample(num_to_sample_in),
      num_being_sampled(num_being_sampled_in),
      num_derivatives(configure_for_gradients ? num_to_sample : 0),
      num_union(num_to_sample + num_being_sampled),
      union_of_points(BuildUnionOfPoints(points_to_sample, points_being_sampled, num_to_sample, num_being_sampled, dim)),
      points_to_sample_state(*ei_evaluator.gaussian_process(), union_of_points.data(), num_union, num_derivatives),
      uniform_rng(uniform_rng_in),
      to_sample_mean(num_union),
      grad_mu(dim*num_derivatives),
      cholesky_to_sample_var(Square(num_union)),
      grad_chol_decomp(dim*Square(num_union)*num_derivatives),
      configure_for_test(false),
      gpu_mu(num_union),
      gpu_chol_var(Square(num_union)),
      gpu_grad_mu(dim * num_derivatives),
      gpu_grad_chol_var(dim * Square(num_union) * num_derivatives),
      gpu_ei_storage(kEINumThreads * kEINumBlocks),
      gpu_grad_ei_storage(kGradEINumThreads * kGradEINumBlocks * dim * num_derivatives),
      gpu_random_number_ei(0),
      gpu_random_number_grad_ei(0),
      random_number_ei(0),
      random_number_grad_ei(0) {
}

CudaExpectedImprovementState::CudaExpectedImprovementState(const EvaluatorType& ei_evaluator,
                                                           double const * restrict points_to_sample,
                                                           double const * restrict points_being_sampled,
                                                           int num_to_sample_in, int num_being_sampled_in,
                                                           bool configure_for_gradients,
                                                           UniformRandomGenerator * uniform_rng_in,
                                                           bool configure_for_test_in)
    : dim(ei_evaluator.dim()),
      num_to_sample(num_to_sample_in),
      num_being_sampled(num_being_sampled_in),
      num_derivatives(configure_for_gradients ? num_to_sample : 0),
      num_union(num_to_sample + num_being_sampled),
      union_of_points(BuildUnionOfPoints(points_to_sample, points_being_sampled, num_to_sample, num_being_sampled, dim)),
      points_to_sample_state(*ei_evaluator.gaussian_process(), union_of_points.data(), num_union, num_derivatives),
      uniform_rng(uniform_rng_in),
      to_sample_mean(num_union),
      grad_mu(dim*num_derivatives),
      cholesky_to_sample_var(Square(num_union)),
      grad_chol_decomp(dim*Square(num_union)*num_derivatives),
      configure_for_test(configure_for_test_in),
      gpu_mu(num_union),
      gpu_chol_var(Square(num_union)),
      gpu_grad_mu(dim * num_derivatives),
      gpu_grad_chol_var(dim * Square(num_union) * num_derivatives),
      gpu_ei_storage(kEINumThreads * kEINumBlocks),
      gpu_grad_ei_storage(kGradEINumThreads * kGradEINumBlocks * dim * num_derivatives),
      gpu_random_number_ei(configure_for_test ? GetVectorSize(ei_evaluator.num_mc(), kEINumThreads, kEINumBlocks, num_union) : 0),
      gpu_random_number_grad_ei(configure_for_test ? GetVectorSize(ei_evaluator.num_mc(), kGradEINumThreads, kGradEINumBlocks, num_union) : 0),
      random_number_ei(configure_for_test ? GetVectorSize(ei_evaluator.num_mc(), kEINumThreads, kEINumBlocks, num_union) : 0),
      random_number_grad_ei(configure_for_test ? GetVectorSize(ei_evaluator.num_mc(), kGradEINumThreads, kGradEINumBlocks, num_union) : 0) {
}

std::vector<double> CudaExpectedImprovementState::BuildUnionOfPoints(double const * restrict points_to_sample,
                                                                     double const * restrict points_being_sampled,
                                                                     int num_to_sample, int num_being_sampled, int dim) noexcept {
  std::vector<double> union_of_points(dim*(num_to_sample + num_being_sampled));
  std::copy(points_to_sample, points_to_sample + dim*num_to_sample, union_of_points.data());
  std::copy(points_being_sampled, points_being_sampled + dim*num_being_sampled, union_of_points.data() + dim*num_to_sample);
  return union_of_points;
}

int CudaExpectedImprovementState::GetVectorSize(int num_mc_itr, int num_threads, int num_blocks, int num_points) noexcept {
  return ((static_cast<int>(num_mc_itr / (num_threads * num_blocks)) + 1) * (num_threads * num_blocks) * num_points);
}

void CudaExpectedImprovementState::UpdateCurrentPoint(const EvaluatorType& ei_evaluator, double const * restrict points_to_sample) {
  // update points_to_sample in union_of_points
  std::copy(points_to_sample, points_to_sample + num_to_sample*dim, union_of_points.data());

  // evaluate derived quantities for the GP
  points_to_sample_state.SetupState(*ei_evaluator.gaussian_process(), union_of_points.data(), num_union, num_derivatives);
}

void CudaExpectedImprovementState::SetupState(const EvaluatorType& ei_evaluator, double const * restrict points_to_sample) {
  // update quantities derived from points_to_sample
  UpdateCurrentPoint(ei_evaluator, points_to_sample);
}
#endif  // OL_GPU_ENABLED

}  // end namespace optimal_learning

