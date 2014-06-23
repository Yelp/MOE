/*!\rst
  \file gpp_cuda_math.hpp
  \rst
  This file contains declaration of gpu functions (host code) that are called by C++ code. The functions include calculating ExpectedImprovement, gradient of ExpectedImprovement, and gpu utility functions (memory allocation, setup gpu device, etc)
\endrst*/

#ifndef MOE_OPTIMAL_LEARNING_CPP_GPU_GPP_CUDA_MATH_HPP
#define MOE_OPTIMAL_LEARNING_CPP_GPU_GPP_CUDA_MATH_HPP

#define OL_EI_THREAD_NO 256
#define OL_EI_BLOCK_NO 32
#define OL_GRAD_EI_THREAD_NO 256
#define OL_GRAD_EI_BLOCK_NO 32

#include "driver_types.h"

namespace optimal_learning {

/*
  This C struct contains error information that are used by exception handling in gpp_expected_improvement_gpu.hpp/cpp
*/
struct CudaError{
  cudaError_t err;
  char const * line_info;
  char const * func_info;
};

extern "C" CudaError cuda_get_EI(double * __restrict__ mu, double * __restrict__ L, double best, int num_union_of_pts, double * __restrict__ dev_mu, double * __restrict__ dev_L, double * __restrict__ dev_EIs, unsigned int seed, int num_mc, double* __restrict__ ei_val);

extern "C" CudaError cuda_get_gradEI(double * __restrict__ mu, double * __restrict__ grad_mu, double * __restrict__ L, double * __restrict__ grad_L, double best, int num_union_of_pts, int num_to_sample, int dimension, double * __restrict__ dev_mu, double * __restrict__ dev_grad_mu, double * __restrict__ dev_L, double * __restrict__ dev_grad_L, double * __restrict__ dev_grad_EIs, unsigned int seed, int num_mc, double * __restrict__ grad_EI);

extern "C" CudaError cuda_allocate_mem_for_double_vector(int num_doubles, double** __restrict__ ptr_to_ptr);

extern "C" void cuda_free_mem(double* __restrict__ ptr);

extern "C" CudaError cuda_set_device(int devID);

}   // end namespace optimal_learning
#endif  // MOE_OPTIMAL_LEARNING_CPP_GPU_GPP_CUDA_MATH_HPP

