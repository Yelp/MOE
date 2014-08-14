/*!
  \file gpp_cuda_math.hpp
  \rst
  This file contains declaration of gpu functions (host code) that are called by C++ code. The functions include calculating ExpectedImprovement, gradient of ExpectedImprovement, and gpu utility functions (memory allocation, setup gpu device, etc)
\endrst*/

#ifndef MOE_OPTIMAL_LEARNING_CPP_GPU_GPP_CUDA_MATH_HPP_
#define MOE_OPTIMAL_LEARNING_CPP_GPU_GPP_CUDA_MATH_HPP_

#include <stdint.h>

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
  //! error returned by CUDA API functions (basically enum type)
  cudaError_t err;
  //! file and line info of the function which returned error
  char const * file_and_line_info;
  //! name of the function that returned error
  char const * func_info;
};

/*!\rst
  Compute Expected Improvement by Monte-Carlo using GPU, and this function is only meant to be used by
  CudaExpectedImprovementEvaluator::ComputeExpectedImprovement(...) in gpp_expected_improvement_gpu.hpp/cpp

  \param
    :mu[num_union]: the mean of the GP evaluated at points interested
    :chol_var[num_union][num_union]: cholesky factorization of the GP variance evaluated at points interested
    :num_union: number of the points interested
    :num_mc: number of iterations for Monte-Carlo simulation
    :seed: seed for RNG
    :best: best function evaluation obtained so far
    :configure_for_test: whether record random_number_ei or not
    :random_number_ei[num_union][num_iteration][num_threads][num_blocks]: random numbers used for
      computing EI, for testing purpose only
    :ei_val[1]: pointer to value of Expected Improvement
    :gpu_mu[num_union]: pointer to memory storing mu on GPU
    :gpu_chol_var[num_union][num_union]: pointer to memory storing chol_var on GPU
    :gpu_random_number_ei[num_union][num_iteration][num_threads][num_blocks]: pointer to memory storing
      random numbers used for computing EI, for testing purpose only
    :gpu_ei_storage[num_threads][num_blocks]: pointer to memory storing values of EI on GPU
  \output
    :ei_val[1]: value of Expected Improvement modified, and equals to computed value of EI
    :gpu_random_number_ei[num_union][num_iteration][num_threads][num_blocks]: pointer to memory storing
      random numbers used for computing EI, for testing purpose only
    :random_number_ei[num_union][num_iteration][num_threads][num_blocks]: random numbers used for
      computing EI, for testing purpose only
  \return
    CudaError state, which contains error information, file name, line and function name of the function that occurs error
\endrst*/
extern "C" CudaError CudaGetEI(double * __restrict__ mu, double * __restrict__ chol_var, int num_union,
                               int num_mc, uint64_t seed, double best, bool configure_for_test,
                               double* __restrict__ random_number_ei, double* __restrict__ ei_val,
                               double * __restrict__ gpu_mu, double * __restrict__ gpu_chol_var,
                               double* __restrict__ gpu_random_number_ei, double * __restrict__ gpu_ei_storage);

/*!\rst
  Compute Gradient of Expected Improvement by Monte-Carlo using GPU, and this function is only meant to be used by 
  CudaExpectedImprovementEvaluator::ComputeGradExpectedImprovement(...) in gpp_expected_improvement_gpu.hpp/cpp

  \param
    :mu[num_union]: the mean of the GP evaluated at points interested
    :chol_var[num_union][num_union]: cholesky factorization of the GP variance evaluated at points interested
    :grad_mu[dim][num_to_sample]: the gradient of mean of the GP evaluated at points interested
    :grad_chol_var[dim][num_union][num_union][num_to_sample]: gradient of cholesky factorization of
      the GP variance evaluated at points interested
    :num_union: number of the union of points (aka q+p)
    :num_to_sample: number of points to sample (aka q)
    :dim: dimension of point space
    :num_mc: number of iterations for Monte-Carlo simulation
    :seed: seed for RNG
    :best: best function evaluation obtained so far
    :configure_for_test: whether record random_number_grad_ei or not
    :random_number_grad_ei[num_union][num_threads][num_blocks]: random numbers used for computing gradEI,
      for testing purpose only
    :grad_ei[dim][num_to_sample]: pointer to value of gradient of Expected Improvement
    :gpu_mu[num_union]: pointer to memory storing mu on GPU
    :gpu_chol_var[num_union][num_union]: pointer to memory storing chol_var on GPU
    :gpu_grad_mu[dim][num_to_sample]: pointer to memory storing grad_mu on GPU
    :gpu_grad_chol_var[dim][num_union][num_union][num_to_sample]: pointer to memory storing grad_chol_var on GPU
    :gpu_random_number_grad_ei[num_union][num_threads][num_blocks]: pointer to memory storing random
      numbers used for computing gradEI, for testing purpose only
    :gpu_grad_ei_storage[dim][num_to_sample][num_threads][num_blocks]: pointer to memory storing values of gradient EI on GPU
  \output
    :random_number_grad_ei[num_union][num_threads][num_blocks]: random numbers used for computing gradEI, for testing purpose only
    :grad_ei[dim][num_to_sample]: pointer to value of gradient of Expected Improvement
    :gpu_random_number_grad_ei[num_union][num_threads][num_blocks]: pointer to memory storing random
      numbers used for computing gradEI, for testing purpose only
  \return
    CudaError state, which contains error information, file name, line and function name of the function that occurs error
\endrst*/
extern "C" CudaError CudaGetGradEI(double * __restrict__ mu, double * __restrict__ chol_var, double * __restrict__ grad_mu,
                                   double * __restrict__ grad_chol_var, int num_union, int num_to_sample, int dim, int num_mc,
                                   uint64_t seed, double best, bool configure_for_test, double* __restrict__ random_number_grad_ei,
                                   double * __restrict__ grad_ei, double * __restrict__ gpu_mu, double * __restrict__ gpu_chol_var,
                                   double * __restrict__ gpu_grad_mu, double * __restrict__ gpu_grad_chol_var,
                                   double* __restrict__ gpu_random_number_grad_ei, double * __restrict__ gpu_grad_ei_storage);

/*!\rst
  Allocate GPU memory for storing an array. This is same as malloc in C, with error handling.

  \param
    :num_doubles: number of double numbers contained in the array
    :address_of_ptr_to_gpu_memory: address of the pointer to memory on GPU
  \return
    CudaError state, which contains error information, file name, line and function name of the function that occurs error
\endrst*/
extern "C" CudaError CudaAllocateMemForDoubleVector(int num_doubles, double** __restrict__ address_of_ptr_to_gpu_memory);

/*!\rst
  Free GPU memory, same as free() in C.

  \param
    :ptr_to_gpu_memory: pointer to memory on GPU to free
\endrst*/
extern "C" void CudaFreeMem(double* __restrict__ ptr_to_gpu_memory);

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
