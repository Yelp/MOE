/*!
  \file gpp_expected_improvement_gpu.cpp
  \rst
  gpu code for calculating Expected Improvemnet & gradient of Expected Improvement
\endrst*/

#include "gpp_expected_improvement_gpu.hpp"

#include <vector>

#include "gpp_common.hpp"
#include "gpp_logging.hpp"
#include "gpp_random.hpp"
#include "gpp_math.hpp"
#include "gpp_exception.hpp"

#ifdef OL_GPU_ENABLED
#include "gpu/gpp_cuda_math.hpp"
#include "driver_types.h"
#include "cuda_runtime.h"

#define OL_CUDA_THROW_EXCEPTION(_ERR) ThrowException(OptimalLearningException((_ERR).line_info, (_ERR).func_info, cudaGetErrorString((_ERR).err)));
#define OL_CUDA_ERROR_THROW(_ERR) {if((_ERR).err != cudaSuccess) {OL_CUDA_THROW_EXCEPTION(_ERR)}}
#endif

namespace optimal_learning {
#ifdef OL_GPU_ENABLED
CudaDevicePointer::CudaDevicePointer(int num_doubles_in)
  : num_doubles(num_doubles_in) {
    if (num_doubles_in > 0) {
        CudaError _err = cuda_allocate_mem_for_double_vector(num_doubles, &ptr);
        if (_err.err != cudaSuccess) {
            ptr = nullptr;
            OL_CUDA_THROW_EXCEPTION(_err)
        }
    } else {
        ptr = nullptr;
    }
  }

CudaDevicePointer::~CudaDevicePointer() {
  if (ptr != nullptr) {
      cuda_free_mem(ptr);
  }
}

double CudaExpectedImprovementEvaluator::ComputeExpectedImprovement(StateType * ei_state) const {
  double EI_val;
  int num_union = ei_state->num_union;
  gaussian_process_->ComputeMeanOfPoints(ei_state->points_to_sample_state, ei_state->to_sample_mean.data());
  gaussian_process_->ComputeVarianceOfPoints(&(ei_state->points_to_sample_state), ei_state->cholesky_to_sample_var.data());
  ComputeCholeskyFactorL(num_union, ei_state->cholesky_to_sample_var.data());
  unsigned int seed_in = (ei_state->normal_rng->GetEngine())();
  CudaError _err = cuda_get_EI(ei_state->to_sample_mean.data(), ei_state->cholesky_to_sample_var.data(), best_so_far_, num_union, (ei_state->dev_mu).ptr, (ei_state->dev_L).ptr, (ei_state->dev_EIs).ptr, seed_in, num_mc, &EI_val);
  OL_CUDA_ERROR_THROW(_err)
  return EI_val;
}

void CudaExpectedImprovementEvaluator::ComputeGradExpectedImprovement(StateType * ei_state, double * restrict grad_EI) const {
  if (ei_state->num_derivatives == 0) {
    OL_THROW_EXCEPTION(OptimalLearningException, "configure_for_gradients set to false, gradient computation is disabled!");
  }
  const int num_union = ei_state->num_union;
  const int num_to_sample = ei_state->num_to_sample;
  gaussian_process_->ComputeMeanOfPoints(ei_state->points_to_sample_state, ei_state->to_sample_mean.data());
  gaussian_process_->ComputeGradMeanOfPoints(ei_state->points_to_sample_state, ei_state->grad_mu.data());
  gaussian_process_->ComputeVarianceOfPoints(&(ei_state->points_to_sample_state), ei_state->cholesky_to_sample_var.data());
  ComputeCholeskyFactorL(num_union, ei_state->cholesky_to_sample_var.data());

  gaussian_process_->ComputeGradCholeskyVarianceOfPoints(&(ei_state->points_to_sample_state), ei_state->cholesky_to_sample_var.data(), ei_state->grad_chol_decomp.data());
  unsigned int seed_in = (ei_state->normal_rng->GetEngine())();

  CudaError _err = cuda_get_gradEI(ei_state->to_sample_mean.data(), ei_state->grad_mu.data(), ei_state->cholesky_to_sample_var.data(), ei_state->grad_chol_decomp.data(), best_so_far_, num_union, num_to_sample, dim_, (ei_state->dev_mu).ptr, (ei_state->dev_grad_mu).ptr, (ei_state->dev_L).ptr, (ei_state->dev_grad_L).ptr, (ei_state->dev_grad_EIs).ptr, seed_in, num_mc, grad_EI);
  OL_CUDA_ERROR_THROW(_err)
}

void CudaExpectedImprovementEvaluator::setupGPU(int devID) {
  CudaError _err = cuda_set_device(devID);
  OL_CUDA_ERROR_THROW(_err)
}

CudaExpectedImprovementEvaluator::~CudaExpectedImprovementEvaluator() {
  cudaDeviceReset();
}

CudaExpectedImprovementState::CudaExpectedImprovementState(const EvaluatorType& ei_evaluator, double const * restrict points_to_sample, double const * restrict points_being_sampled, int num_to_sample_in, int num_being_sampled_in, bool configure_for_gradients, NormalRNG * normal_rng_in)
  : dim(ei_evaluator.dim()),
    num_to_sample(num_to_sample_in),
    num_being_sampled(num_being_sampled_in),
    num_derivatives(configure_for_gradients ? num_to_sample : 0),
    num_union(num_to_sample + num_being_sampled),
    union_of_points(BuildUnionOfPoints(points_to_sample, points_being_sampled, num_to_sample, num_being_sampled, dim)),
    points_to_sample_state(*ei_evaluator.gaussian_process(), union_of_points.data(), num_union, num_derivatives),
    normal_rng(normal_rng_in),
    to_sample_mean(num_union),
    grad_mu(dim*num_derivatives),
    cholesky_to_sample_var(Square(num_union)),
    grad_chol_decomp(dim*Square(num_union)*num_derivatives),
    dev_mu(num_union),
    dev_L(Square(num_union)),
    dev_grad_mu(dim * num_derivatives),
    dev_grad_L(dim * Square(num_union) * num_derivatives),
    dev_EIs(OL_EI_THREAD_NO * OL_EI_BLOCK_NO),
    dev_grad_EIs(OL_GRAD_EI_THREAD_NO * OL_GRAD_EI_BLOCK_NO * dim * num_derivatives) {
    }

#else

double CudaExpectedImprovementEvaluator::ComputeExpectedImprovement(StateType * ei_state) const {
  OL_THROW_EXCEPTION(OptimalLearningException, "GPU component is disabled or unavailable, cannot call gpu function!\n");
}

void CudaExpectedImprovementEvaluator::ComputeGradExpectedImprovement(StateType * ei_state, double * restrict grad_EI) const {
  OL_THROW_EXCEPTION(OptimalLearningException, "GPU component is disabled or unavailable, cannot call gpu function!\n");
}

void CudaExpectedImprovementEvaluator::setupGPU(int devID) {
  OL_THROW_EXCEPTION(OptimalLearningException, "GPU component is disabled or unavailable, cannot call gpu function!\n");
}

CudaExpectedImprovementEvaluator::~CudaExpectedImprovementEvaluator() {
}

CudaExpectedImprovementState::CudaExpectedImprovementState(const EvaluatorType& ei_evaluator, double const * restrict points_to_sample, double const * restrict points_being_sampled, int num_to_sample_in, int num_being_sampled_in, bool configure_for_gradients, NormalRNG * normal_rng_in)
  : dim(ei_evaluator.dim()),
    num_to_sample(num_to_sample_in),
    num_being_sampled(num_being_sampled_in),
    num_derivatives(configure_for_gradients ? num_to_sample : 0),
    num_union(num_to_sample + num_being_sampled),
    union_of_points(BuildUnionOfPoints(points_to_sample, points_being_sampled, num_to_sample, num_being_sampled, dim)),
    points_to_sample_state(*ei_evaluator.gaussian_process(), union_of_points.data(), num_union, num_derivatives),
    normal_rng(normal_rng_in),
    to_sample_mean(num_union),
    grad_mu(dim*num_derivatives),
    cholesky_to_sample_var(Square(num_union)),
    grad_chol_decomp(dim*Square(num_union)*num_derivatives) {
        OL_THROW_EXCEPTION(OptimalLearningException, "GPU component is disabled or unavailable, cannot call gpu function!\n");
    }
#endif
}  // end namespace optimal_learning

