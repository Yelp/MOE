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
#include <cuda_runtime.h>
#include "gpu/gpp_cuda_math.hpp"
#endif

namespace optimal_learning {
#ifdef OL_GPU_ENABLED
    double CudaExpectedImprovementEvaluator::ComputeExpectedImprovement(StateType * ei_state) const {
      int num_union = ei_state->num_union;
      gaussian_process_->ComputeMeanOfPoints(ei_state->points_to_sample_state, ei_state->to_sample_mean.data());
      gaussian_process_->ComputeVarianceOfPoints(&(ei_state->points_to_sample_state), ei_state->cholesky_to_sample_var.data());
      ComputeCholeskyFactorL(num_union, ei_state->cholesky_to_sample_var.data());
      unsigned int seed_in = (ei_state->normal_rng->GetEngine())();
      return cuda_get_EI(ei_state->to_sample_mean.data(), ei_state->cholesky_to_sample_var.data(), best_so_far_, num_union, ei_state->dev_mu, ei_state->dev_L, ei_state->dev_EIs, seed_in, num_mc);
    }

    void CudaExpectedImprovementEvaluator::ComputeGradExpectedImprovement(StateType * ei_state, double * restrict grad_EI) const {
      const int num_union = ei_state->num_union;
      const int num_to_sample = ei_state->num_to_sample;
      gaussian_process_->ComputeMeanOfPoints(ei_state->points_to_sample_state, ei_state->to_sample_mean.data());
      gaussian_process_->ComputeGradMeanOfPoints(ei_state->points_to_sample_state, ei_state->grad_mu.data());
      gaussian_process_->ComputeVarianceOfPoints(&(ei_state->points_to_sample_state), ei_state->cholesky_to_sample_var.data());
      ComputeCholeskyFactorL(num_union, ei_state->cholesky_to_sample_var.data());

      gaussian_process_->ComputeGradCholeskyVarianceOfPoints(&(ei_state->points_to_sample_state), ei_state->cholesky_to_sample_var.data(), ei_state->grad_chol_decomp.data());
      unsigned int seed_in = (ei_state->normal_rng->GetEngine())();

      cuda_get_gradEI(ei_state->to_sample_mean.data(), ei_state->grad_mu.data(), ei_state->cholesky_to_sample_var.data(), ei_state->grad_chol_decomp.data(), best_so_far_, num_union, num_to_sample, dim_, ei_state->dev_mu, ei_state->dev_grad_mu, ei_state->dev_L, ei_state->dev_grad_L, ei_state->dev_grad_EIs, seed_in, num_mc, grad_EI);
    }

    // hackhack: need some error handle
    // It's nice to have "findCudaDevice()" functionality, however the required header is in Cuda SDK Sample folder, which need not be existed for Cuda to run
    void CudaExpectedImprovementEvaluator::setupGPU(int devID) {
        cudaSetDevice(devID);
    }

    void CudaExpectedImprovementEvaluator::resetGPU() {
        cudaDeviceReset();
    }

    CudaExpectedImprovementEvaluator::~CudaExpectedImprovementEvaluator() {
          resetGPU();
    }

    CudaExpectedImprovementState::~CudaExpectedImprovementState() {
          cuda_memory_deallocation();
    }

    void CudaExpectedImprovementState::cuda_memory_allocation() {
        cuda_allocate_mem(num_union, num_to_sample, dim, &dev_mu, &dev_grad_mu, &dev_L, &dev_grad_L, &dev_grad_EIs, &dev_EIs);
    }

    void CudaExpectedImprovementState::cuda_memory_deallocation() {
        cuda_free_mem(dev_mu, dev_grad_mu, dev_L, dev_grad_L, dev_grad_EIs, dev_EIs);
    }

#else
    double CudaExpectedImprovementEvaluator::ComputeExpectedImprovement(StateType * ei_state) const {
        OL_THROW_EXCEPTION(RuntimeException, "GPU component is disabled or unavailable, cannot call gpu function!");
        return 0.0;
    }
    void CudaExpectedImprovementEvaluator::ComputeGradExpectedImprovement(StateType * ei_state, double * restrict grad_EI) const {
        OL_THROW_EXCEPTION(RuntimeException, "GPU component is disabled or unavailable, cannot call gpu function!");
    }
    void CudaExpectedImprovementEvaluator::setupGPU(int devID) {
        OL_THROW_EXCEPTION(RuntimeException, "GPU component is disabled or unavailable, cannot call gpu function!");
    }
    void CudaExpectedImprovementEvaluator::resetGPU() {
        OL_THROW_EXCEPTION(RuntimeException, "GPU component is disabled or unavailable, cannot call gpu function!");
    }
    CudaExpectedImprovementEvaluator::~CudaExpectedImprovementEvaluator() {
        OL_THROW_EXCEPTION(RuntimeException, "GPU component is disabled or unavailable, cannot call gpu function!");
    }
    CudaExpectedImprovementState::~CudaExpectedImprovementState() {
        OL_THROW_EXCEPTION(RuntimeException, "GPU component is disabled or unavailable, cannot call gpu function!");
    }
    void CudaExpectedImprovementState::cuda_memory_allocation() {
        OL_THROW_EXCEPTION(RuntimeException, "GPU component is disabled or unavailable, cannot call gpu function!");
    }
    void CudaExpectedImprovementState::cuda_memory_deallocation() {
        OL_THROW_EXCEPTION(RuntimeException, "GPU component is disabled or unavailable, cannot call gpu function!");
    }
#endif
}

