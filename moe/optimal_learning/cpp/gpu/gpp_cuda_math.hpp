/*!\rst
  \file gpp_cuda_math.hpp
  \rst
  This file contains declaration of gpu functions (host code) that are called by C++ code. The functions include calculating ExpectedImprovement, gradient of ExpectedImprovement, and gpu utility functions (memory allocation, setup gpu device, etc)
\endrst*/

#ifndef MOE_OPTIMAL_LEARNING_CPP_GPU_GPP_CUDA_MATH_HPP_
#define MOE_OPTIMAL_LEARNING_CPP_GPU_GPP_CUDA_MATH_HPP_

#include "driver_types.h"

namespace optimal_learning {

static unsigned int EI_thread_no = 256;
static unsigned int EI_block_no = 32;
static unsigned int gradEI_thread_no = 256;
static unsigned int gradEI_block_no = 32;

/*
  This C struct contains error information that are used by exception handling in gpp_expected_improvement_gpu.hpp/cpp
*/
struct CudaError {
  cudaError_t err;
  char const * line_info;
  char const * func_info;
};

extern "C" CudaError cuda_get_EI(double * __restrict__ mu, double * __restrict__ chol_var, double best, int num_union, double * __restrict__ gpu_mu, double * __restrict__ gpu_chol_var, double * __restrict__ gpu_EI_storage, unsigned int seed, int num_mc, double* __restrict__ ei_val, double* __restrict__ gpu_random_number_EI, double* __restrict__ random_number_EI, bool configure_for_test);

extern "C" CudaError cuda_get_gradEI(double * __restrict__ mu, double * __restrict__ grad_mu, double * __restrict__ chol_var, double * __restrict__ grad_chol_var, double best, int num_union, int num_to_sample, int dimension, double * __restrict__ gpu_mu, double * __restrict__ gpu_grad_mu, double * __restrict__ gpu_chol_var, double * __restrict__ gpu_grad_chol_var, double * __restrict__ gpu_grad_EI_storage, unsigned int seed, int num_mc, double * __restrict__ grad_EI, double* __restrict__
         gpu_random_number_gradEI, double* __restrict__ random_number_gradEI, bool configure_for_test);

extern "C" CudaError cuda_allocate_mem_for_double_vector(int num_doubles, double** __restrict__ address_of_ptr_to_gpu_memory);

extern "C" void cuda_free_mem(double* __restrict__ ptr_to_gpu_memory);

extern "C" CudaError cuda_set_device(int devID);

}   // end namespace optimal_learning
#endif  // MOE_OPTIMAL_LEARNING_CPP_GPU_GPP_CUDA_MATH_HPP_
