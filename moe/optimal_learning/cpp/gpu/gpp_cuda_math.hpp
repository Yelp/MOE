/*!
  \file gpp_cuda_math.hpp
  \rst
  This file contains declaration of gpu functions (host code) that are called by C++ code. The functions include calculating ExpectedImprovement, gradient of ExpectedImprovement, and gpu utility functions (memory allocation, setup gpu device, etc)
\endrst*/

#ifndef MOE_OPTIMAL_LEARNING_CPP_GPU_GPP_CUDA_MATH_HPP_
#define MOE_OPTIMAL_LEARNING_CPP_GPU_GPP_CUDA_MATH_HPP_

#include <stdint.h>

#include <driver_types.h>

/*!\rst
  Macro to allow ``restrict`` as a keyword for ``C++`` compilation and ``CUDA/nvcc`` compilation.
  See related entry in ``gpp_common.hpp`` for more details.
\endrst*/
#if defined(__CUDACC__) || defined(__cplusplus)
#define restrict __restrict__
#endif

namespace optimal_learning {

//! Number of blocks assigned for computing Expected Improvement on GPU
static const unsigned int kEINumBlocks = 32;
//! Number of threads per block assigned for computing Expected Improvement on GPU
static const unsigned int kEINumThreads = 256;
//! Number of blocks assigned for computing Gradient of Expected Improvement on GPU
static const unsigned int kGradEINumBlocks = 32;
//! Number of threads per block assigned for computing Gradient of Expected Improvement on GPU
static const unsigned int kGradEINumThreads = 256;

/*!\rst
  This C struct contains error information that are used by exception handling in gpp_expected_improvement_gpu.hpp/cpp.
  File/line and function information are empty strings if the error code is cudaSuccess (i.e., no error).
\endrst*/
struct CudaError {
  //! error returned by CUDA API functions (basically enum type)
  cudaError_t err;
  //! file and line info of the function which returned error
  char const * file_and_line_info;
  //! name of the function that returned error
  char const * func_info;
};

//! CudaError struct encoding a successful CUDA operation.
static const CudaError kCudaSuccess = {cudaSuccess, "", ""};

/*!\rst
  Compute Expected Improvement by Monte-Carlo using GPU, and this function is only meant to be used by
  CudaExpectedImprovementEvaluator::ComputeExpectedImprovement(...) in gpp_expected_improvement_gpu.hpp/cpp

  \param
    :mu[num_union]: the mean of the GP evaluated at points interested
    :chol_var[num_union][num_union]: cholesky factorization of the GP variance evaluated at points interested
    :num_union: number of the points interested
    :num_mc: number of iterations for Monte-Carlo simulation
    :best: best function evaluation obtained so far
    :base_seed: base seed for the GPU's RNG; will be offset by GPU thread index (see curand)
    :configure_for_test: whether record random_number_ei or not
  \output
    :gpu_mu[num_union]: device pointer to memory storing mu on GPU
    :gpu_chol_var[num_union][num_union]: device pointer to memory storing chol_var on GPU
    :random_number_ei[num_union][num_iteration][num_threads][num_blocks]: random numbers used for
      computing EI, for testing purpose only
    :gpu_random_number_ei[num_union][num_iteration][num_threads][num_blocks]: device pointer to memory storing
      random numbers used for computing EI, for testing purpose only
    :gpu_ei_storage[num_threads][num_blocks]: device pointer to memory storing values of EI on GPU
    :ei_val[1]: pointer to value of Expected Improvement
  \return
    CudaError state, which contains error information, file name, line and function name of the function that occurs error
\endrst*/
extern "C" CudaError CudaGetEI(double const * restrict mu, double const * restrict chol_var,
                               int num_union, int num_mc, double best,
                               uint64_t base_seed, bool configure_for_test,
                               double * restrict gpu_mu, double * restrict gpu_chol_var,
                               double * restrict random_number_ei,
                               double * restrict gpu_random_number_ei,
                               double * restrict gpu_ei_storage,
                               double * restrict ei_val);

/*!\rst
  Compute Gradient of Expected Improvement by Monte-Carlo using GPU, and this function is only meant to be used by
  CudaExpectedImprovementEvaluator::ComputeGradExpectedImprovement(...) in gpp_expected_improvement_gpu.hpp/cpp

  \param
    :mu[num_union]: the mean of the GP evaluated at points interested
    :grad_mu[dim][num_to_sample]: the gradient of mean of the GP evaluated at points interested
    :chol_var[num_union][num_union]: cholesky factorization of the GP variance evaluated at points interested
    :grad_chol_var[dim][num_union][num_union][num_to_sample]: gradient of cholesky factorization of
      the GP variance evaluated at points interested
    :num_union: number of the union of points (aka q+p)
    :num_to_sample: number of points to sample (aka q)
    :dim: dimension of point space
    :num_mc: number of iterations for Monte-Carlo simulation
    :best: best function evaluation obtained so far
    :base_seed: base seed for the GPU's RNG; will be offset by GPU thread index (see curand)
    :configure_for_test: whether record random_number_grad_ei or not
  \output
    :gpu_mu[num_union]: device pointer to memory storing mu on GPU
    :gpu_grad_mu[dim][num_to_sample]: device pointer to memory storing grad_mu on GPU
    :gpu_chol_var[num_union][num_union]: device pointer to memory storing chol_var on GPU
    :gpu_grad_chol_var[dim][num_union][num_union][num_to_sample]: device pointer to memory storing grad_chol_var on GPU
    :random_number_grad_ei[num_union][num_threads][num_blocks]: random numbers used for computing gradEI, for testing purpose only
    :gpu_random_number_grad_ei[num_union][num_threads][num_blocks]: device pointer to memory storing random
      numbers used for computing gradEI, for testing purpose only
    :gpu_grad_ei_storage[dim][num_to_sample][num_threads][num_blocks]: device pointer to memory storing values of gradient EI on GPU
    :grad_ei[dim][num_to_sample]: pointer to gradient of Expected Improvement
  \return
    CudaError state, which contains error information, file name, line and function name of the function that occurs error
\endrst*/
extern "C" CudaError CudaGetGradEI(double const * restrict mu, double const * restrict grad_mu,
                                   double const * restrict chol_var, double const * restrict grad_chol_var,
                                   int num_union, int num_to_sample, int dim, int num_mc,
                                   double best, uint64_t base_seed, bool configure_for_test,
                                   double * restrict gpu_mu, double * restrict gpu_grad_mu,
                                   double * restrict gpu_chol_var, double * restrict gpu_grad_chol_var,
                                   double * restrict random_number_grad_ei,
                                   double * restrict gpu_random_number_grad_ei,
                                   double * restrict gpu_grad_ei_storage,
                                   double * restrict grad_ei);

/*!\rst
  Allocate GPU device memory for storing an array; analogous to ``malloc()`` in ``C``.
  Thin wrapper around ``cudaMalloc()`` that handles errors.
  See: ``http://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html``

  Do not dereference ``address_of_ptr_to_gpu_memory`` outside the GPU device.
  Do not dereference ``address_of_ptr_to_gpu_memory`` if the error code (``return_value.err``) is not ``cudaSuccess``.

  \param
    :size: number of bytes to allocate
    :address_of_ptr_to_gpu_memory: address of the pointer to alllocated device memory on the GPU
  \return
    CudaError state, which contains error information, file name, line and function name of the function that occurs error
\endrst*/
extern "C" CudaError CudaMallocDeviceMemory(size_t size, void ** restrict address_of_ptr_to_gpu_memory);

/*!\rst
  Free GPU device memory on the GPU; analogous to ``free()`` in ``C``.
  Thin wrapper around ``cudaFree()`` that handles errors.
  See: ``http://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html``

  \param
    :ptr_to_gpu_memory: pointer to memory on GPU to free; MUST have been returned by a previous call to ``cudaMalloc()``.
  \return
    CudaError state, which contains error information, file name, line and function name of the function that occurs error
\endrst*/
extern "C" CudaError CudaFreeDeviceMemory(void * restrict ptr_to_gpu_memory);

/*!\rst
  Setup GPU device, and all GPU function calls will be operated on the GPU activated by this function.

  \param
    :devID: the ID of GPU device to setup
  \return
    CudaError state, which contains error information, file name, line and function name of the function that occurs error
\endrst*/
extern "C" CudaError CudaSetDevice(int devID);

}   // end namespace optimal_learning

#endif  // MOE_OPTIMAL_LEARNING_CPP_GPU_GPP_CUDA_MATH_HPP_
