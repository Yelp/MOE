/*!
  \file gpp_cuda_math.hpp
  \rst
  Functions compiled by nvcc are all declared here. 
\endrst*/

#ifndef MOE_OPTIMAL_LEARNING_CPP_GPU_GPP_CUDA_MATH_HPP
#define MOE_OPTIMAL_LEARNING_CPP_GPU_GPP_CUDA_MATH_HPP


namespace optimal_learning {
    extern "C" double cuda_get_EI(double * __restrict__ mu, double * __restrict__ L, double best, int num_union_of_pts, double * __restrict__ dev_mu, double * __restrict__ dev_L, double * __restrict__ dev_EIs, unsigned int seed, int num_mc);
    extern "C" void cuda_get_gradEI(double * __restrict__ mu, double * __restrict__ grad_mu, double * __restrict__ L, double * __restrict__ grad_L, double best, int num_union_of_pts, int num_to_sample, int dimension, double * __restrict__ dev_mu, double * __restrict__ dev_grad_mu, double * __restrict__ dev_L, double * __restrict__ dev_grad_L, double * __restrict__ dev_grad_EIs, unsigned int seed, int num_mc, double * __restrict__ grad_EI);
    extern "C" void cuda_allocate_mem(int num_union_of_pts, int num_to_sample, int dimension, double** __restrict__ pointer_dev_mu, double** __restrict__ pointer_dev_grad_mu, double** __restrict__ pointer_dev_L, double** __restrict__ pointer_dev_grad_L, double** __restrict__ pointer_dev_grad_EIs, double ** __restrict__ pointer_dev_EIs); 
    extern "C" void cuda_free_mem(double* __restrict__ dev_mu, double* __restrict__ dev_grad_mu, double* __restrict__ dev_L, double* __restrict__ dev_grad_L, double* __restrict__ dev_grad_EIs, double* __restrict__ dev_EIs); 

}   // end namespace optimal_learning
#endif  // MOE_OPTIMAL_LEARNING_CPP_GPU_GPP_CUDA_MATH_HPP

