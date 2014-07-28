/*!\rst
  \file gpp_cuda_math.hpp
  \rst
  This file contains declaration of gpu functions (host code) that are called by C++ code. The functions include calculating ExpectedImprovement, gradient of ExpectedImprovement, and gpu utility functions (memory allocation, setup gpu device, etc)
\endrst*/

#ifndef MOE_OPTIMAL_LEARNING_CPP_GPU_GPP_CUDA_MATH_HPP_
#define MOE_OPTIMAL_LEARNING_CPP_GPU_GPP_CUDA_MATH_HPP_

#include "driver_types.h"

namespace optimal_learning {

//! Number of blocks assigned for computing Expected Improvement on GPU
static unsigned int kEINumBlocks = 32;
//! Number of threads per block assigned for computing Expected Improvement on GPU
static unsigned int kEINumThreads = 256;
//! Number of blocks assigned for computing Gradient of Expected Improvement on GPU
static unsigned int kGradEINumBlocks = 32;
//! Number of threads per block assigned for computing Gradient of Expected Improvement on GPU
static unsigned int kGradEINumThreads = 256;

/*!\rst
  This C struct contains error information that are used by exception handling in gpp_expected_improvement_gpu.hpp/cpp
\endrst*/
struct CudaError {
  //! error returned by CUDA API functions(basiclly enum type)
  cudaError_t err;
  //! file and line info of the function which returned error
  char const * file_and_line_info;
  //! name of the function that returned error
  char const * func_info;
};

extern "C" CudaError CudaGetEI(double * __restrict__ mu, double * __restrict__ chol_var, double best, int num_union, double * __restrict__ gpu_mu, double * __restrict__ gpu_chol_var, double * __restrict__ gpu_ei_storage, unsigned int seed, int num_mc, double* __restrict__ ei_val, double* __restrict__ gpu_random_number_ei, double* __restrict__ random_number_ei, bool configure_for_test);

extern "C" CudaError CudaGetGradEI(double * __restrict__ mu, double * __restrict__ grad_mu, double * __restrict__ chol_var, double * __restrict__ grad_chol_var, double best, int num_union, int num_to_sample, int dimension, double * __restrict__ gpu_mu, double * __restrict__ gpu_grad_mu, double * __restrict__ gpu_chol_var, double * __restrict__ gpu_grad_chol_var, double * __restrict__ gpu_grad_ei_storage, unsigned int seed, int num_mc, double * __restrict__ grad_ei, double* __restrict__
         gpu_random_number_grad_ei, double* __restrict__ random_number_grad_ei, bool configure_for_test);

extern "C" CudaError CudaAllocateMemForDoubleVector(int num_doubles, double** __restrict__ address_of_ptr_to_gpu_memory);

extern "C" void CudaFreeMem(double* __restrict__ ptr_to_gpu_memory);

extern "C" CudaError CudaSetDevice(int devID);

}   // end namespace optimal_learning
#endif  // MOE_OPTIMAL_LEARNING_CPP_GPU_GPP_CUDA_MATH_HPP_
