/*!
  \file gpp_cuda_math.cu
  \rst
  This file contains implementations of all GPU functions. There are both device code (executed on
  GPU device) and host code (executed on CPU), and they are compiled by NVCC, which is a NVIDIA CUDA
  compiler.
\endrst*/

#include "gpp_cuda_math.hpp"

#include <stdint.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>

#include <algorithm>

/*!\rst
  Macro to stringify the expansion of a macro. For example, say we are on line 53:

  * ``#__LINE__ --> "__LINE__"``
  * ``OL_CUDA_STRINGIFY_EXPANSION(__LINE__) --> "53"``

  ``OL_CUDA_STRINGIFY_EXPANSION_INNER`` is not meant to be used directly;
  but we need ``#x`` in a macro for this expansion to work.

  This is a standard trick; see bottom of:
  http://gcc.gnu.org/onlinedocs/cpp/Stringification.html
\endrst*/
#define OL_CUDA_STRINGIFY_EXPANSION_INNER(x) #x
#define OL_CUDA_STRINGIFY_EXPANSION(x) OL_CUDA_STRINGIFY_EXPANSION_INNER(x)

/*!\rst
  Macro to stringify and format the current file and line number. For
  example, if the macro is invoked from line 893 of file gpp_foo.cpp,
  this macro produces the compile-time string-constant:
  ``(gpp_foo.cpp: 893)``
\endrst*/
#define OL_CUDA_STRINGIFY_FILE_AND_LINE "(" __FILE__ ": " OL_CUDA_STRINGIFY_EXPANSION(__LINE__) ")"

/*!\rst
  Macro that checks error message (with type cudaError_t) returned by CUDA API functions, and if there is error occurred,
  the macro produces a C struct containing error message, function name where error occured, file name and line info, and 
  then terminate the function.
\endrst*/
#define OL_CUDA_ERROR_RETURN(X) do {cudaError_t _error_code = (X); if (_error_code != cudaSuccess) {CudaError _err = {_error_code, OL_CUDA_STRINGIFY_FILE_AND_LINE, __func__}; return _err;}} while (0)

namespace optimal_learning {

namespace {  // functions run on gpu device
/*!\rst
  Special case of GeneralMatrixVectorMultiply.  As long as A has zeros in the strict upper-triangle,
  GeneralMatrixVectorMultiply will work too (but take ``>= 2x`` as long).

  Computes results IN-PLACE.
  Avoids accessing the strict upper triangle of A.

  Should be equivalent to BLAS call:
  ``dtrmv('L', trans, 'N', size_m, A, size_m, x, 1);``
\endrst*/
__device__ void CudaTriangularMatrixVectorMultiply(double const * __restrict__ A, int size_m, double * __restrict__ x) {
  double temp;
  A += size_m * (size_m-1);
  for (int j = size_m-1; j >= 0; --j) {  // i.e., j >= 0
    temp = x[j];
    for (int i = size_m-1; i >= j+1; --i) {
      // handles sub-diagonal contributions from j-th column
      x[i] += temp*A[i];
    }
    x[j] *= A[j];  // handles j-th on-diagonal component
    A -= size_m;
  }
}

/*!\rst
  This is reduced version of GeneralMatrixVectorMultiply(...) in gpp_linear_algebra.cpp, and this function computes
  y = y - A * x (aka alpha = -1.0, beta = 1.0)
\endrst*/
__device__ void CudaGeneralMatrixVectorMultiply(double const * __restrict__ A, double const * __restrict__ x, int size_m, int size_n, int lda, double * __restrict__ y) {
  double temp;
  for (int i = 0; i < size_n; ++i) {
    temp = -1.0 * x[i];
    for (int j = 0; j < size_m; ++j) {
      y[j] += A[j]*temp;
    }
    A += lda;
  }
}

/*!\rst
  This inline function copies [begin, begin+1, ..., end-1] elements from one array to the other, if bound < end, then end = bound
\endrst*/
__forceinline__ __device__ void CudaCopyElements(int begin, int end, int bound, double const * __restrict__ origin, double * __restrict__ destination) {
    int local_end = end < bound ? end : bound;
    for (int idx = begin; idx < local_end; ++idx) {
        destination[idx] = origin[idx];
    }
}

/*!\rst
  Device code to compute Expected Improvement by Monte-Carlo on GPU

  \param
    :mu[num_union]: the mean of the GP evaluated at points interested
    :chol_var[num_union][num_union]: cholesky factorization of the GP variance evaluated at points interested
    :num_union: number of the points interested
    :num_iteration: number of iterations performed on each thread for MC evaluation
    :best: best function evaluation obtained so far
    :seed: seed for RNG
    :ei_storage[num_threads][num_blocks]: array storing values of EI on GPU
    :gpu_random_number_ei[num_union][num_iteration][num_threads][num_blocks]: array storing random
      numbers used for computing EI, for testing purpose only
    :configure_for_test: whether record random_number_ei or not
  \output
    :ei_storage[num_threads][num_blocks]: each thread write result of computed EI to its corresponding position
    :gpu_random_number_ei[num_union][num_iteration][num_threads][num_blocks]: write random numbers
      used for computing EI into the array, for testing purpose only
\endrst*/
__global__ void CudaComputeEIGpu(double const * __restrict__ mu, double const * __restrict__ chol_var,
                                 int num_union, int num_iteration, double best, uint64_t seed,
                                 double * __restrict__ ei_storage, double* __restrict__ gpu_random_number_ei,
                                 bool configure_for_test) {
  // copy mu, chol_var to shared memory mu_local & chol_var_local
  // For multiple dynamically sized arrays in a single kernel, declare a single extern unsized array, and use
  // pointers into it to divide it into multiple arrays
  // refer to http://devblogs.nvidia.com/parallelforall/using-shared-memory-cuda-cc/
  extern __shared__ double storage[];
  double * chol_var_local = storage;
  double * mu_local = chol_var_local + num_union * num_union;
  const int idx = threadIdx.x;
  const int IDX = threadIdx.x + blockDim.x * blockIdx.x;
  int chunk_size = (num_union * num_union - 1)/ blockDim.x + 1;
  CudaCopyElements(chunk_size * idx, chunk_size * (idx + 1), num_union * num_union, chol_var, chol_var_local);
  chunk_size = (num_union - 1)/ blockDim.x + 1;
  CudaCopyElements(chunk_size * idx, chunk_size * (idx + 1), num_union,  mu, mu_local);
  __syncthreads();

  // MC start
  // RNG setup
  uint64_t local_seed = seed + IDX;
  curandState random_state;
  // seed a random number generator
  curand_init(local_seed, 0, 0, &random_state);

  double *normals = reinterpret_cast<double *>(malloc(sizeof(*chol_var_local) * num_union));
  double agg = 0.0;
  double improvement_this_step;
  double EI;

  for (int mc = 0; mc < num_iteration; ++mc) {
    improvement_this_step = 0.0;
    for (int i = 0; i < num_union; ++i) {
        normals[i] = curand_normal_double(&random_state);
        // If configure_for_test is true, random numbers used in MC computations will be saved as output.
        // In fact we will let EI compuation on CPU use the same sequence of random numbers saved here,
        // so that EI compuation on CPU & GPU can be compared directly for unit test purpose.
        if (configure_for_test) {
            gpu_random_number_ei[IDX * num_iteration * num_union + mc * num_union + i] = normals[i];
        }
    }
    CudaTriangularMatrixVectorMultiply(chol_var_local, num_union, normals);
    for (int i = 0; i < num_union; ++i) {
        EI = best - (mu_local[i] + normals[i]);
        improvement_this_step = fmax(EI, improvement_this_step);
    }
    agg += improvement_this_step;
  }
  ei_storage[IDX] = agg / static_cast<double>(num_iteration);
  free(normals);
}

/*!\rst
  Device code to compute Gradient of Expected Improvement by Monte-Carlo on GPU

  \param
    :mu[num_union]: the mean of the GP evaluated at points interested
    :chol_var[num_union][num_union]: cholesky factorization of the GP variance evaluated at points interested
    :grad_mu[dim][num_to_sample]: the gradient of mean of the GP evaluated at points interested
    :grad_chol_var[dim][num_union][num_union][num_to_sample]: gradient of cholesky factorization of the GP variance
      evaluated at points interested
    :num_union: number of the union of points (aka q+p)
    :num_to_sample: number of points to sample (aka q)
    :dim: dimension of point space
    :num_iteration: number of iterations performed on each thread for MC evaluation
    :best: best function evaluation obtained so far
    :seed: seed for RNG
    :grad_ei_storage[dim][num_to_sample][num_threads][num_blocks]: A vector storing result of grad_ei from each thread
    :gpu_random_number_grad_ei[num_union][num_itreration][num_threads][num_blocks]: array storing
      random numbers used for computing gradEI, for testing purpose only
    :configure_for_test: whether record random_number_grad_ei or not
  \output
    :grad_ei_storage[dim][num_to_sample][num_threads][num_blocks]: each thread write result of grad_ei
      to its corresponding positions
    :gpu_random_number_grad_ei[num_union][num_iteration][num_threads][num_blocks]: write random numbers
      used for computing gradEI to the array, for testing purpose only
\endrst*/
__global__ void CudaComputeGradEIGpu(double const * __restrict__ mu, double const * __restrict__ chol_var,
                                     double const * __restrict__ grad_mu, double const * __restrict__ grad_chol_var,
                                     int num_union, int num_to_sample, int dim, int num_iteration, double best,
                                     uint64_t seed,  double * __restrict__ grad_ei_storage,
                                     double* __restrict__ gpu_random_number_grad_ei, bool configure_for_test) {
  // copy mu, chol_var, grad_mu, grad_chol_var to shared memory
  extern __shared__ double storage[];
  double * mu_local = storage;
  double * chol_var_local = mu_local + num_union;
  double * grad_mu_local = chol_var_local + num_union * num_union;
  double * grad_chol_var_local = grad_mu_local + num_to_sample * dim;
  const int idx = threadIdx.x;
  const int IDX = threadIdx.x + blockDim.x * blockIdx.x;
  int chunk_size = (num_to_sample * num_union * num_union * dim - 1)/ blockDim.x + 1;
  CudaCopyElements(chunk_size * idx, chunk_size * (idx + 1), num_to_sample * num_union * num_union * dim,
                  grad_chol_var, grad_chol_var_local);
  chunk_size = (num_union * num_union - 1)/ blockDim.x + 1;
  CudaCopyElements(chunk_size * idx, chunk_size * (idx + 1), num_union * num_union, chol_var, chol_var_local);
  chunk_size = (num_to_sample * dim - 1)/ blockDim.x + 1;
  CudaCopyElements(chunk_size * idx, chunk_size * (idx + 1), num_to_sample * dim, grad_mu, grad_mu_local);
  chunk_size = (num_union - 1)/ blockDim.x + 1;
  CudaCopyElements(chunk_size * idx, chunk_size * (idx + 1), num_union, mu, mu_local);
  __syncthreads();

  int i, k, mc, winner;
  double EI, improvement_this_step;
  // RNG setup
  uint64_t local_seed = seed + IDX;
  curandState random_state;
  curand_init(local_seed, 0, 0, &random_state);
  double* normals = reinterpret_cast<double*>(malloc(sizeof(*mu_local) * num_union));
  double* normals_copy = reinterpret_cast<double*>(malloc(sizeof(*mu_local) * num_union));
  // initialize grad_ei_storage
  for (int i = 0; i < (num_to_sample * dim); ++i) {
      grad_ei_storage[IDX*num_to_sample*dim + i] = 0.0;
  }
  // MC step start
  for (mc = 0; mc < num_iteration; ++mc) {
      improvement_this_step = 0.0;
      winner = -1;
      for (i = 0; i < num_union; ++i) {
          normals[i] = curand_normal_double(&random_state);
          normals_copy[i] = normals[i];
            // If configure_for_test is true, random numbers used in MC computations will be saved as output.
            // In fact we will let grad_ei compuation on CPU use the same sequence of random numbers saved here,
            // so that grad_ei compuation on CPU & GPU can be compared directly for unit test purpose.
          if (configure_for_test) {
              gpu_random_number_grad_ei[IDX * num_iteration * num_union + mc * num_union + i] = normals[i];
          }
      }
      CudaTriangularMatrixVectorMultiply(chol_var_local, num_union, normals);
      for (i = 0; i < num_union; ++i) {
          EI = best - (mu_local[i] + normals[i]);
          if (EI > improvement_this_step) {
              improvement_this_step = EI;
              winner = i;
          }
      }
      if (improvement_this_step > 0.0) {
          if (winner < num_to_sample) {
              for (k = 0; k < dim; ++k) {
                  grad_ei_storage[IDX*num_to_sample*dim + winner * dim + k] -= grad_mu_local[winner * dim + k];
              }
          }
          for (i = 0; i < num_to_sample; ++i) {   // derivative w.r.t ith point
              CudaGeneralMatrixVectorMultiply(grad_chol_var_local + i * num_union * num_union * dim +
                                              winner * num_union * dim, normals_copy, dim, num_union,
                                              dim, grad_ei_storage + IDX * num_to_sample * dim + i * dim);
          }
      }
  }

  for (int i = 0; i < num_to_sample*dim; ++i) {
      grad_ei_storage[IDX*num_to_sample*dim + i] /= static_cast<double>(num_iteration);
  }
  free(normals);
  free(normals_copy);
}

}  // end unnamed namespace

CudaError CudaAllocateMemForDoubleVector(int num_doubles, double** __restrict__ address_of_ptr_to_gpu_memory) {
  CudaError _success = {cudaSuccess, OL_CUDA_STRINGIFY_FILE_AND_LINE, __func__};
  int mem_size = num_doubles * sizeof(**address_of_ptr_to_gpu_memory);
  OL_CUDA_ERROR_RETURN(cudaMalloc(reinterpret_cast<void**>(address_of_ptr_to_gpu_memory), mem_size));
  return _success;
}

void CudaFreeMem(double* __restrict__ ptr_to_gpu_memory) {
  cudaFree(ptr_to_gpu_memory);
}

CudaError CudaGetEI(double * __restrict__ mu, double * __restrict__ chol_var, int num_union, int num_mc,
                    uint64_t seed, double best, bool configure_for_test, double * __restrict__ random_number_ei,
                    double * __restrict__ ei_val, double * __restrict__ gpu_mu, double * __restrict__ gpu_chol_var,
                    double* __restrict__ gpu_random_number_ei, double * __restrict__ gpu_ei_storage) {
  *ei_val = 0.0;
  CudaError _success = {cudaSuccess, OL_CUDA_STRINGIFY_FILE_AND_LINE, __func__};

  // We assign kEINumBlocks blocks and kEINumThreads threads/block for EI computation, so there are
  // (kEINumBlocks * kEINumThreads) threads in total to execute kernel function in parallel
  dim3 threads(kEINumThreads);
  dim3 grid(kEINumBlocks);
  double ei_storage[kEINumThreads * kEINumBlocks];
  int num_iteration = num_mc / (kEINumThreads * kEINumBlocks) + 1;   // make sure num_iteration is always >= 1

  int mem_size_mu = num_union * sizeof(*mu);
  int mem_size_chol_var = num_union * num_union * sizeof(*mu);
  int mem_size_ei_storage = kEINumThreads * kEINumBlocks * sizeof(*mu);
  // copy mu, chol_var to GPU
  OL_CUDA_ERROR_RETURN(cudaMemcpy(gpu_mu, mu, mem_size_mu, cudaMemcpyHostToDevice));
  OL_CUDA_ERROR_RETURN(cudaMemcpy(gpu_chol_var, chol_var, mem_size_chol_var, cudaMemcpyHostToDevice));
  // execute kernel
  CudaComputeEIGpu <<< grid, threads, num_union*sizeof(*mu)+num_union*num_union*sizeof(*mu) >>>
                   (gpu_mu, gpu_chol_var, num_union, num_iteration, best, seed, gpu_ei_storage,
                    gpu_random_number_ei, configure_for_test);
  OL_CUDA_ERROR_RETURN(cudaPeekAtLastError());
  // copy gpu_ei_storage back to CPU
  OL_CUDA_ERROR_RETURN(cudaMemcpy(ei_storage, gpu_ei_storage, mem_size_ei_storage, cudaMemcpyDeviceToHost));
  // copy gpu_random_number_ei back to CPU if configure_for_test is on
  if (configure_for_test) {
      int mem_size_random_number_ei = num_iteration * kEINumThreads * kEINumBlocks * num_union * sizeof(*mu);
      OL_CUDA_ERROR_RETURN(cudaMemcpy(random_number_ei, gpu_random_number_ei, mem_size_random_number_ei, cudaMemcpyDeviceToHost));
  }
  // average ei_storage
  double ave = 0.0;
  for (int i = 0; i < (kEINumThreads*kEINumBlocks); ++i) {
      ave += ei_storage[i];
  }
  *ei_val = ave / static_cast<double>(kEINumThreads*kEINumBlocks);
  return _success;
}

CudaError CudaGetGradEI(double * __restrict__ mu, double * __restrict__ chol_var, double * __restrict__ grad_mu,
                        double * __restrict__ grad_chol_var, int num_union, int num_to_sample, int dim, int num_mc,
                        uint64_t seed, double best, bool configure_for_test, double* __restrict__ random_number_grad_ei,
                        double * __restrict__ grad_ei, double * __restrict__ gpu_mu, double * __restrict__ gpu_chol_var,
                        double * __restrict__ gpu_grad_mu, double * __restrict__ gpu_grad_chol_var,
                        double* __restrict__ gpu_random_number_grad_ei, double * __restrict__ gpu_grad_ei_storage) {
  CudaError _success = {cudaSuccess, OL_CUDA_STRINGIFY_FILE_AND_LINE, __func__};

  double grad_ei_storage[num_to_sample * dim * kGradEINumThreads * kGradEINumBlocks];
  std::fill(grad_ei, grad_ei + num_to_sample * dim, 0.0);

  // We assign kGradEINumBlocks blocks and kGradEINumThreads threads/block for grad_ei computation,
  // so there are (kGradEINumBlocks * kGradEINumThreads) threads in total to execute kernel function
  // in parallel
  dim3 threads(kGradEINumThreads);
  dim3 grid(kGradEINumBlocks);
  int num_iteration = num_mc / (kGradEINumThreads * kGradEINumBlocks) + 1;   // make sure num_iteration is always >= 1

  int mem_size_mu = num_union * sizeof(*mu);
  int mem_size_grad_mu = num_to_sample * dim * sizeof(*mu);
  int mem_size_chol_var = num_union * num_union *sizeof(*mu);
  int mem_size_grad_chol_var = num_to_sample * num_union * num_union * dim * sizeof(*mu);
  int mem_size_grad_ei_storage= kGradEINumThreads * kGradEINumBlocks * num_to_sample * dim * sizeof(*mu);

  OL_CUDA_ERROR_RETURN(cudaMemcpy(gpu_mu, mu, mem_size_mu, cudaMemcpyHostToDevice));
  OL_CUDA_ERROR_RETURN(cudaMemcpy(gpu_grad_mu, grad_mu, mem_size_grad_mu, cudaMemcpyHostToDevice));
  OL_CUDA_ERROR_RETURN(cudaMemcpy(gpu_chol_var, chol_var, mem_size_chol_var, cudaMemcpyHostToDevice));
  OL_CUDA_ERROR_RETURN(cudaMemcpy(gpu_grad_chol_var, grad_chol_var, mem_size_grad_chol_var, cudaMemcpyHostToDevice));

  // execute kernel
  // inputs: gpu_mu, gpu_chol_var, gpu_grad_mu, gpu_grad_chol_var, best, num_union, num_to_sample, dim, num_iteration, seed
  // output: gpu_grad_ei_storage
  CudaComputeGradEIGpu <<< grid, threads, mem_size_mu+mem_size_chol_var+mem_size_grad_mu+mem_size_grad_chol_var >>>
                       (gpu_mu, gpu_chol_var, gpu_grad_mu, gpu_grad_chol_var, num_union, num_to_sample, dim,
                        num_iteration, best, seed, gpu_grad_ei_storage, gpu_random_number_grad_ei, configure_for_test);
  OL_CUDA_ERROR_RETURN(cudaPeekAtLastError());

  OL_CUDA_ERROR_RETURN(cudaMemcpy(grad_ei_storage, gpu_grad_ei_storage, mem_size_grad_ei_storage, cudaMemcpyDeviceToHost));
  // copy gpu_random_number_grad_ei back to CPU if configure_for_test is on
  if (configure_for_test) {
      int mem_size_random_number_grad_ei = num_iteration * kGradEINumThreads * kGradEINumBlocks * num_union * sizeof(*mu);
      OL_CUDA_ERROR_RETURN(cudaMemcpy(random_number_grad_ei, gpu_random_number_grad_ei, mem_size_random_number_grad_ei, cudaMemcpyDeviceToHost));
  }

  // The code block below extracts grad_ei from grad_ei_storage, which is output from the function
  // "CudaGetGradEI" run on gpu. The way to do that is for each component of grad_ei, we find all
  // the threads calculating the corresponding component and average over the threads.
  for (int n = 0; n < (kGradEINumThreads*kGradEINumBlocks); ++n) {
      for (int i = 0; i < num_to_sample*dim; ++i) {
          grad_ei[i] += grad_ei_storage[n*num_to_sample*dim + i];
      }
  }
  for (int i = 0; i < num_to_sample*dim; ++i) {
      grad_ei[i] /= static_cast<double>(kGradEINumThreads*kGradEINumBlocks);
  }
  return _success;
}

CudaError CudaSetDevice(int devID) {
  CudaError _success = {cudaSuccess, OL_CUDA_STRINGIFY_FILE_AND_LINE, __func__};
  OL_CUDA_ERROR_RETURN(cudaSetDevice(devID));
  return _success;
}

}    // end namespace optimal_learning
