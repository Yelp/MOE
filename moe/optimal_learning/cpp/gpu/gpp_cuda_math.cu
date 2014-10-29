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
#include <vector>

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
__device__ void CudaTriangularMatrixVectorMultiply(double const * restrict A, int size_m, double * restrict x) {
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
__device__ void CudaGeneralMatrixVectorMultiply(double const * restrict A, double const * restrict x, int size_m, int size_n, int lda, double * restrict y) {
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
__forceinline__ __device__ void CudaCopyElements(int begin, int end, int bound, double const * restrict origin, double * restrict destination) {
    int local_end = end < bound ? end : bound;
    for (int idx = begin; idx < local_end; ++idx) {
        destination[idx] = origin[idx];
    }
}

/*!\rst
  GPU kernel function of computing Expected Improvement using Monte-Carlo.

  **Shared Memory Requirements**

  This method requires the caller to allocate 3 arrays: chol_var_local, mu_local and normals, with

  ``(num_union * num_union + num_union + num_union * num_threads)``

  doubles in total in shared memory. The order of the arrays placed in this shared memory is like
  ``[chol_var_local, mu_local, normals]``

  Currently size of shared memory per block is set to 48K, to give you a sense, that is approximately
  6144 doubles, for example, this caller works when num_union = 22 without blowing up shared memory
  (currently num_threads = 256).

  :chol_var_local[num_union][num_union]: copy of chol_var in shared memory for each block
  :mu_local[num_union]: copy of mu in shared memory for each block
  :normals[num_union][num_threads]: shared memory for storage of normal random numbers for each block

  \param
    :mu[num_union]: the mean of the GP evaluated at points interested
    :chol_var[num_union][num_union]: cholesky factorization of the GP variance evaluated at points interested
    :num_union: number of the points interested
    :num_iteration: number of iterations performed on each thread for MC evaluation
    :best: best function evaluation obtained so far
    :base_seed: base seed for the GPU's RNG; will be offset by GPU thread index (see curand)
    :configure_for_test: whether record random_number_ei or not
  \output
    :gpu_random_number_ei[num_union][num_iteration][num_threads][num_blocks]: array storing random
      numbers used for computing EI, for testing only
    :ei_storage[num_threads][num_blocks]: each thread's computed EI component written to its corresponding position
\endrst*/
__global__ void CudaComputeEIGpu(double const * restrict mu, double const * restrict chol_var,
                                 int num_union, int num_iteration, double best,
                                 uint64_t base_seed, bool configure_for_test,
                                 double * restrict gpu_random_number_ei,
                                 double * restrict ei_storage) {
  // copy mu, chol_var to shared memory mu_local & chol_var_local
  // For multiple dynamically sized arrays in a single kernel, declare a single extern unsized array, and use
  // pointers into it to divide it into multiple arrays
  // refer to http://devblogs.nvidia.com/parallelforall/using-shared-memory-cuda-cc/
  extern __shared__ double storage[];
  double * restrict chol_var_local = storage;
  double * restrict mu_local = chol_var_local + num_union * num_union;
  const int idx = threadIdx.x;
  const int IDX = threadIdx.x + blockDim.x * blockIdx.x;
  int chunk_size = (num_union * num_union - 1)/ blockDim.x + 1;
  CudaCopyElements(chunk_size * idx, chunk_size * (idx + 1), num_union * num_union, chol_var, chol_var_local);
  chunk_size = (num_union - 1)/ blockDim.x + 1;
  CudaCopyElements(chunk_size * idx, chunk_size * (idx + 1), num_union,  mu, mu_local);
  __syncthreads();
  double * restrict normals = mu_local + num_union + idx * num_union;

  // MC start
  // RNG setup
  uint64_t local_seed = base_seed + IDX;
  curandState random_state;
  // seed a random number generator
  curand_init(local_seed, 0, 0, &random_state);

  double agg = 0.0;
  double improvement_this_step;
  double EI;

  for (int mc = 0; mc < num_iteration; ++mc) {
    improvement_this_step = 0.0;
    for (int i = 0; i < num_union; ++i) {
        normals[i] = curand_normal_double(&random_state);
        // If configure_for_test is true, random numbers used in MC computations will be saved as output.
        // In fact we will let EI compuation on CPU use the same sequence of random numbers saved here,
        // so that EI compuation on CPU & GPU can be compared directly for unit testing.
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
}

/*!\rst
  Device code to compute Gradient of Expected Improvement by Monte-Carlo on GPU.

  **Shared Memory Requirements**

  This method requires the caller to allocate 5 arrays: mu_local, chol_var_local, grad_mu_local,
  grad_chol_var_local and normals, with

  ``(num_union + num_union * num_union + dim * num_to_sample + dim * num_union * num_union *
    num_to_sample + 2 * num_union * num_threads)``

  doubles in total in shared memory.
  The order of the arrays placed in this shared memory is like ``[mu_local, chol_var_local, grad_mu_local,
  grad_chol_var_local, normals]``

  Currently size of shared memory per block is set to 48K, to give you a sense, that is approximately
  6144 doubles, for example, this caller works for num_union = num_to_sample = 8, dim = 3 without
  blowing up shared memory (currently num_threads = 256).

  :mu_local[num_union]: copy of mu in shared memory for each block
  :chol_var_local[num_union][num_union]: copy of chol_var in shared memory for each block
  :grad_mu_local[dim][num_to_sample]: copy of grad_mu in shared memory for each block
  :grad_chol_var_local[dim][num_union][num_union][num_to_sample]: copy of grad_chol_var_local in shared memory for each block
  :normals[2 * num_union][num_threads]: shared memory for storage of normal random numbers for each block, and for each thread
    it gets 2 * num_union normal random numbers, with one set of normals occupying the first num_union doubles, and we store a copy
    of them in the rest of the spaces.

  \param
    :mu[num_union]: the mean of the GP evaluated at points interested
    :grad_mu[dim][num_to_sample]: the gradient of mean of the GP evaluated at points interested
    :chol_var[num_union][num_union]: cholesky factorization of the GP variance evaluated at points interested
    :grad_chol_var[dim][num_union][num_union][num_to_sample]: gradient of cholesky factorization of the GP variance
      evaluated at points interested
    :num_union: number of the union of points (aka q+p)
    :num_to_sample: number of points to sample (aka q)
    :dim: dimension of point space
    :num_iteration: number of iterations performed on each thread for MC evaluation
    :best: best function evaluation obtained so far
    :base_seed: base seed for the GPU's RNG; will be offset by GPU thread index (see curand)
    :configure_for_test: whether record random_number_grad_ei or not
  \output
    :gpu_random_number_grad_ei[num_union][num_itreration][num_threads][num_blocks]: array storing
      random numbers used for computing gradEI, for testing only
    :grad_ei_storage[dim][num_to_sample][num_threads][num_blocks]: each thread write result of grad_ei
      to its corresponding positions
\endrst*/
__global__ void CudaComputeGradEIGpu(double const * restrict mu, double const * restrict grad_mu,
                                     double const * restrict chol_var, double const * restrict grad_chol_var,
                                     int num_union, int num_to_sample, int dim, int num_iteration, double best,
                                     uint64_t base_seed, bool configure_for_test,
                                     double * restrict gpu_random_number_grad_ei,
                                     double * restrict grad_ei_storage) {
  // copy mu, chol_var, grad_mu, grad_chol_var to shared memory
  extern __shared__ double storage[];
  double * restrict mu_local = storage;
  double * restrict chol_var_local = mu_local + num_union;
  double * restrict grad_mu_local = chol_var_local + num_union * num_union;
  double * restrict grad_chol_var_local = grad_mu_local + num_to_sample * dim;
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
  double * restrict normals = grad_chol_var_local + num_union * num_union * num_to_sample * dim + idx * num_union * 2;
  double * restrict normals_copy = normals + num_union;

  int i, k, mc, winner;
  double EI, improvement_this_step;
  // RNG setup
  uint64_t local_seed = base_seed + IDX;
  curandState random_state;
  curand_init(local_seed, 0, 0, &random_state);
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
            // In fact we will let gradEI compuation on CPU use the same sequence of random numbers saved here,
            // so that gradEI compuation on CPU & GPU can be compared directly for unit testing.
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
              CudaGeneralMatrixVectorMultiply(grad_chol_var_local + i*num_union*num_union*dim + winner*num_union*dim,
                                              normals_copy, dim, num_union, dim,
                                              grad_ei_storage + IDX*num_to_sample*dim + i*dim);
          }
      }
  }

  for (int i = 0; i < num_to_sample*dim; ++i) {
      grad_ei_storage[IDX*num_to_sample*dim + i] /= static_cast<double>(num_iteration);
  }
}

}  // end unnamed namespace

CudaError CudaGetEI(double const * restrict mu, double const * restrict chol_var,
                    int num_union, int num_mc, double best,
                    uint64_t base_seed, bool configure_for_test,
                    double * restrict gpu_mu, double * restrict gpu_chol_var,
                    double * restrict random_number_ei,
                    double * restrict gpu_random_number_ei,
                    double * restrict gpu_ei_storage,
                    double * restrict ei_val) {
  // We assign kEINumBlocks blocks and kEINumThreads threads/block for EI computation, so there are
  // (kEINumBlocks * kEINumThreads) threads in total to execute kernel function in parallel
  dim3 threads(kEINumThreads);
  dim3 grid(kEINumBlocks);
  std::vector<double> ei_storage(kEINumThreads * kEINumBlocks);

  int num_iteration = num_mc / (kEINumThreads * kEINumBlocks) + 1;   // make sure num_iteration is always >= 1

  int mem_size_mu = num_union * sizeof(*mu);
  int mem_size_chol_var = num_union * num_union * sizeof(*mu);
  int mem_size_ei_storage = kEINumThreads * kEINumBlocks * sizeof(*mu);
  // copy mu, chol_var to GPU
  OL_CUDA_ERROR_RETURN(cudaMemcpy(gpu_mu, mu, mem_size_mu, cudaMemcpyHostToDevice));
  OL_CUDA_ERROR_RETURN(cudaMemcpy(gpu_chol_var, chol_var, mem_size_chol_var, cudaMemcpyHostToDevice));
  // execute kernel
  CudaComputeEIGpu <<< grid, threads, num_union*sizeof(*mu)+num_union*num_union*sizeof(*mu)+num_union*kEINumThreads*sizeof(*mu) >>>
                   (gpu_mu, gpu_chol_var,
                    num_union, num_iteration, best, base_seed, configure_for_test,
                    gpu_random_number_ei, gpu_ei_storage);
  OL_CUDA_ERROR_RETURN(cudaPeekAtLastError());
  // copy gpu_ei_storage back to CPU
  OL_CUDA_ERROR_RETURN(cudaMemcpy(ei_storage.data(), gpu_ei_storage, mem_size_ei_storage, cudaMemcpyDeviceToHost));
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

  return kCudaSuccess;
}

CudaError CudaGetGradEI(double const * restrict mu, double const * restrict grad_mu,
                        double const * restrict chol_var, double const * restrict grad_chol_var,
                        int num_union, int num_to_sample, int dim, int num_mc,
                        double best, uint64_t base_seed, bool configure_for_test,
                        double * restrict gpu_mu, double * restrict gpu_grad_mu,
                        double * restrict gpu_chol_var, double * restrict gpu_grad_chol_var,
                        double * restrict random_number_grad_ei,
                        double * restrict gpu_random_number_grad_ei,
                        double * restrict gpu_grad_ei_storage,
                        double * restrict grad_ei) {
  std::vector<double> grad_ei_storage(num_to_sample * dim * kGradEINumThreads * kGradEINumBlocks);

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
  int mem_size_grad_ei_storage = kGradEINumThreads * kGradEINumBlocks * num_to_sample * dim * sizeof(*mu);

  OL_CUDA_ERROR_RETURN(cudaMemcpy(gpu_mu, mu, mem_size_mu, cudaMemcpyHostToDevice));
  OL_CUDA_ERROR_RETURN(cudaMemcpy(gpu_grad_mu, grad_mu, mem_size_grad_mu, cudaMemcpyHostToDevice));
  OL_CUDA_ERROR_RETURN(cudaMemcpy(gpu_chol_var, chol_var, mem_size_chol_var, cudaMemcpyHostToDevice));
  OL_CUDA_ERROR_RETURN(cudaMemcpy(gpu_grad_chol_var, grad_chol_var, mem_size_grad_chol_var, cudaMemcpyHostToDevice));

  // execute kernel
  // inputs: gpu_mu, gpu_chol_var, gpu_grad_mu, gpu_grad_chol_var, best, num_union, num_to_sample, dim, num_iteration, base_seed
  // output: gpu_grad_ei_storage
  CudaComputeGradEIGpu <<< grid, threads, mem_size_mu+mem_size_chol_var+mem_size_grad_mu+mem_size_grad_chol_var+num_union*kGradEINumThreads*2*sizeof(*mu) >>>
                       (gpu_mu, gpu_grad_mu, gpu_chol_var, gpu_grad_chol_var,
                        num_union, num_to_sample, dim,
                        num_iteration, best, base_seed, configure_for_test,
                        gpu_random_number_grad_ei, gpu_grad_ei_storage);
  OL_CUDA_ERROR_RETURN(cudaPeekAtLastError());

  OL_CUDA_ERROR_RETURN(cudaMemcpy(grad_ei_storage.data(), gpu_grad_ei_storage, mem_size_grad_ei_storage, cudaMemcpyDeviceToHost));
  // copy gpu_random_number_grad_ei back to CPU if configure_for_test is on
  if (configure_for_test) {
      int mem_size_random_number_grad_ei = num_iteration * kGradEINumThreads * kGradEINumBlocks * num_union * sizeof(*mu);
      OL_CUDA_ERROR_RETURN(cudaMemcpy(random_number_grad_ei, gpu_random_number_grad_ei, mem_size_random_number_grad_ei, cudaMemcpyDeviceToHost));
  }

  // The code block below extracts grad_ei from grad_ei_storage, which is output from the function
  // "CudaGetGradEI" run on gpu. The way to do that is for each component of grad_ei, we find all
  // the threads calculating the corresponding component and average over the threads.
  std::fill(grad_ei, grad_ei + num_to_sample * dim, 0.0);
  for (int n = 0; n < (kGradEINumThreads*kGradEINumBlocks); ++n) {
      for (int i = 0; i < num_to_sample*dim; ++i) {
          grad_ei[i] += grad_ei_storage[n*num_to_sample*dim + i];
      }
  }
  for (int i = 0; i < num_to_sample*dim; ++i) {
      grad_ei[i] /= static_cast<double>(kGradEINumThreads*kGradEINumBlocks);
  }

  return kCudaSuccess;
}

CudaError CudaMallocDeviceMemory(size_t size, void ** restrict address_of_ptr_to_gpu_memory) {
  OL_CUDA_ERROR_RETURN(cudaMalloc(address_of_ptr_to_gpu_memory, size));
  return kCudaSuccess;
}

CudaError CudaFreeDeviceMemory(void * restrict ptr_to_gpu_memory) {
  OL_CUDA_ERROR_RETURN(cudaFree(ptr_to_gpu_memory));
  return kCudaSuccess;
}

CudaError CudaSetDevice(int devID) {
  OL_CUDA_ERROR_RETURN(cudaSetDevice(devID));
  // Cuda API to set memory config preference: in our code we prefer to use more shared memory
  OL_CUDA_ERROR_RETURN(cudaDeviceSetCacheConfig(cudaFuncCachePreferShared));

  return kCudaSuccess;
}

}    // end namespace optimal_learning
