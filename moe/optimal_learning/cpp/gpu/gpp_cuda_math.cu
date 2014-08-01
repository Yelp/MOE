#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#include "gpp_cuda_math.hpp"

#define OL_CUDA_STRINGIFY_EXPANSION_INNER(x) #x
#define OL_CUDA_STRINGIFY_EXPANSION(x) OL_CUDA_STRINGIFY_EXPANSION_INNER(x)
#define OL_CUDA_STRINGIFY_FILE_AND_LINE "(" __FILE__ ": " OL_CUDA_STRINGIFY_EXPANSION(__LINE__) ")"
#define OL_CUDA_ERROR_RETURN(X) do {if((X) != cudaSuccess) {CudaError _err = {(X), OL_CUDA_STRINGIFY_FILE_AND_LINE, __func__}; return _err;}} while (0);

namespace optimal_learning {

namespace {  // functions run on gpu device
/*
Special case of GeneralMatrixVectorMultiply.  As long as A has zeros in the strict upper-triangle,
GeneralMatrixVectorMultiply will work too (but take >= 2x as long).

Computes results IN-PLACE.
Avoids accessing the strict upper triangle of A.

Should be equivalent to BLAS call:
dtrmv('chol_var', trans, 'N', size_m, A, size_m, x, 1);

comment: This function is copied from gpp_linear_algebra.cpp
*/
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

/*
y = y - A * x (aka alpha = -1.0, beta = 1.0)

Computes matrix-vector product y = alpha * A * x + beta * y or y = alpha * A^T * x + beta * y
Since A is stored column-major, we need to treat the matrix-vector product as a weighted sum
of the columns of A, where x provides the weights.

That is, a matrix-vector product can be thought of as: (trans = 'T')
[  a_row1  ][   ]
[  a_row2  ][ x ]
[    ...   ][   ]
[  a_rowm  ][   ]
That is, y_i is the dot product of the i-th row of A with x.

OR the "dual" view: (trans = 'N')
[        |        |     |        ][ x_1 ]
[ a_col1 | a_col2 | ... | a_coln ][ ... ] = x_1*a_col1 + ... + x_n*a_coln
[        |        |     |        ][ x_n ]
That is, y is the weighted sum of columns of A.

Should be equivalent to BLAS call:
dgemv(trans='N', size_m, size_n, alpha, A, size_m, x, 1, beta, y, 1);

comment: This function is copied from gpp_linear_algebra.cpp
*/
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

// This inline function copies element from one array to the other, it also checks if index is out of bound before initiating the copy operation.
inline __device__ void CudaCopyElement(int index, int bound, double const * __restrict__ origin, double * __restrict__ destination) {
    if (index < bound) {
        destination[index] = origin[index];
    }
}

// EI_storage: A vector storing calculation result of EI from each thread
__global__ void CudaComputeEIGpu(double const * __restrict__ chol_var, double const * __restrict__ mu, int num_union, int num_iteration, double best, unsigned int seed, double * __restrict__ EI_storage, double* __restrict__ gpu_random_number_EI, bool configure_for_test) {
  // copy mu, chol_var to shared memory mu_local & chol_var_local 
  // For multiple dynamically sized arrays in a single kernel, declare a single extern unsized array, and use
  // pointers into it to divide it into multiple arrays
  // refer to http://devblogs.nvidia.com/parallelforall/using-shared-memory-cuda-cc/
  extern __shared__ double storage[];
  double * chol_var_local = storage;
  double * mu_local = &chol_var_local[num_union * num_union];
  const int idx = threadIdx.x;
  const int IDX = threadIdx.x + blockDim.x * blockIdx.x;
  const int loop_no = num_union * num_union / blockDim.x;
  for (int k = 0; k <= loop_no; ++k) {
    CudaCopyElement(k*blockDim.x+idx, num_union*num_union, chol_var, chol_var_local);
    CudaCopyElement(k*blockDim.x+idx, num_union, mu, mu_local);
  }
  __syncthreads();
  double * ei_random_normal_container = &mu_local[num_union];
  // double *normals = reinterpret_cast<double *>(malloc(sizeof(double)*num_union));
  // normals[0] = 1;

  // MC start
  // RNG setup
  unsigned int local_seed = seed + IDX;
  curandState random_state;
  // seed a random number generator
  curand_init(local_seed, 0, 0, &random_state);

  double agg = 0.0;
  double improvement_this_step;
  double EI;

  for (int mc = 0; mc < num_iteration; ++mc) {
    improvement_this_step = 0.0;
    for (int i = 0; i < num_union; ++i) {
        ei_random_normal_container[idx * num_union + i] = curand_normal_double(&random_state);
        // If configure_for_test is ture, random numbers used in MC computations will be saved as output.
        // In fact we will let EI compuation on CPU use the same sequence of random numbers saved here,
        // so that EI compuation on CPU & GPU can be compared directly for unit test purpose.
        if (configure_for_test) {
            gpu_random_number_EI[IDX * num_iteration * num_union + mc * num_union + i] = ei_random_normal_container[idx * num_union + i];
        }
    }

    CudaTriangularMatrixVectorMultiply(chol_var_local, num_union, ei_random_normal_container + idx * num_union);

    for (int i = 0; i < num_union; ++i) {
        EI = best - (mu_local[i] + ei_random_normal_container[idx * num_union + i]);
        improvement_this_step = fmax(EI, improvement_this_step);
    }
    agg += improvement_this_step;
  }
  EI_storage[IDX] = agg / static_cast<double>(num_iteration);
  // free(normals);
}

// grad_EI_storage[dim][num_to_sample][num_threads]: A vector storing result of grad_EI from each thread
__global__ void CudaComputeGradEIGpu(double const * __restrict__ mu, double const * __restrict__ chol_var, double const * __restrict__ grad_mu, double const * __restrict__ grad_chol_var, double best, int num_union, int num_to_sample, int dim, int num_iteration, unsigned int seed,  double * __restrict__ grad_EI_storage, double* __restrict__ gpu_random_number_gradEI, bool configure_for_test) {
  // copy mu, chol_var, grad_mu, grad_chol_var to shared memory
  extern __shared__ double storage[];
  double * mu_local = storage;
  double * chol_var_local = &mu_local[num_union];
  double * grad_mu_local = &chol_var_local[num_union * num_union];
  double * grad_chol_var_local = &grad_mu_local[num_to_sample * dim];
  const int idx = threadIdx.x;
  const int IDX = threadIdx.x + blockDim.x * blockIdx.x;
  const int loop_no = num_to_sample * num_union * num_union * dim / blockDim.x;
  for (int k = 0; k <= loop_no; ++k) {
      CudaCopyElement(k*blockDim.x+idx, num_to_sample*num_union*num_union*dim, grad_chol_var, grad_chol_var_local);
      CudaCopyElement(k*blockDim.x+idx, num_union*num_union, chol_var, chol_var_local);
      CudaCopyElement(k*blockDim.x+idx, num_to_sample*dim, grad_mu, grad_mu_local);
      CudaCopyElement(k*blockDim.x+idx, num_union, mu, mu_local);
  }
  __syncthreads();
  double * grad_ei_random_normal_container = &grad_chol_var_local[num_union * num_union * num_to_sample * dim];
  // double* normals = reinterpret_cast<double*>(malloc(sizeof(double) * num_union));
  // double* normals_copy = reinterpret_cast<double*>(malloc(sizeof(double) * num_union));
  // normals[0] = 1;
  // normals_copy[0] = 1;

  int i, k, mc, winner;
  double EI, improvement_this_step;
  // RNG setup
  unsigned int local_seed = seed + IDX;
  curandState random_state;
  curand_init(local_seed, 0, 0, &random_state);
  // initialize grad_EI_storage
  for (int i = 0; i < (num_to_sample * dim); ++i) {
      grad_EI_storage[IDX*num_to_sample*dim + i] = 0.0;
  }
  // MC step start
  for (mc = 0; mc < num_iteration; ++mc) {
      improvement_this_step = 0.0;
      winner = -1;
      for (i = 0; i < num_union; ++i) {
          grad_ei_random_normal_container[idx * num_union * 2 + i] = curand_normal_double(&random_state);
          grad_ei_random_normal_container[idx * num_union * 2 + num_union + i] = grad_ei_random_normal_container[idx * num_union * 2 + i];
            // If configure_for_test is ture, random numbers used in MC computations will be saved as output.
            // In fact we will let gradEI compuation on CPU use the same sequence of random numbers saved here,
            // so that gradEI compuation on CPU & GPU can be compared directly for unit test purpose.
          if (configure_for_test) {
              gpu_random_number_gradEI[IDX * num_iteration * num_union + mc * num_union + i] = grad_ei_random_normal_container[idx * num_union * 2 + i];
          }
      }
      CudaTriangularMatrixVectorMultiply(chol_var_local, num_union, grad_ei_random_normal_container + idx * num_union * 2);
      for (i = 0; i < num_union; ++i) {
          EI = best - (mu_local[i] + grad_ei_random_normal_container[idx * num_union * 2 + i]);
          if (EI > improvement_this_step) {
              improvement_this_step = EI;
              winner = i;
          }
      }
      if (improvement_this_step > 0.0) {
          if (winner < num_to_sample) {
              for (k = 0; k < dim; ++k) {
                  grad_EI_storage[IDX*num_to_sample*dim + winner * dim + k] -= grad_mu_local[winner * dim + k];
              }
          }
          for (i = 0; i < num_to_sample; ++i) {   // derivative w.r.t ith point
              CudaGeneralMatrixVectorMultiply(grad_chol_var_local + i*num_union*num_union*dim + winner*num_union*dim, grad_ei_random_normal_container + idx * num_union * 2 + num_union, dim, num_union, dim, grad_EI_storage + IDX*num_to_sample*dim + i*dim);
          }
      }
  }

  for (int i = 0; i < num_to_sample*dim; ++i) {
      grad_EI_storage[IDX*num_to_sample*dim + i] /= static_cast<double>(num_iteration);
  }
  // free(normals);
  // free(normals_copy);
}
}  // end unnamed namespace

CudaError CudaAllocateMemForDoubleVector(int num_doubles, double** __restrict__ address_of_ptr_to_gpu_memory) {
  CudaError _success = {cudaSuccess, OL_CUDA_STRINGIFY_FILE_AND_LINE, __func__};
  int mem_size = num_doubles * sizeof(double);
  OL_CUDA_ERROR_RETURN(cudaMalloc(reinterpret_cast<void**>(address_of_ptr_to_gpu_memory), mem_size))
  return _success;
}

void CudaFreeMem(double* __restrict__ ptr_to_gpu_memory) {
  cudaFree(ptr_to_gpu_memory);
}

CudaError CudaGetEI(double * __restrict__ mu, double * __restrict__ chol_var, double best, int num_union, double * __restrict__ gpu_mu, double * __restrict__ gpu_chol_var, double * __restrict__ gpu_EI_storage, unsigned int seed, int num_mc, double* __restrict__ ei_val, double* __restrict__ gpu_random_number_EI, double* __restrict__ random_number_EI, bool configure_for_test) {
  *ei_val = 0.0;
  CudaError _success = {cudaSuccess, OL_CUDA_STRINGIFY_FILE_AND_LINE, __func__};

  // We assign ei_block_no blocks and ei_thread_no threads/block for EI computation, so there are (ei_block_no * ei_thread_no) threads in total to execute kernel function in parallel
  dim3 threads(ei_thread_no);
  dim3 grid(ei_block_no);
  double EI_storage[ei_thread_no * ei_block_no];
  int num_iteration = num_mc / (ei_thread_no * ei_block_no) + 1;   // make sure num_iteration is always >= 1

  int mem_size_mu = num_union * sizeof(double);
  int mem_size_chol_var = num_union * num_union * sizeof(double);
  int mem_size_EI_storage = ei_thread_no * ei_block_no * sizeof(double);
  // copy mu, chol_var to GPU
  OL_CUDA_ERROR_RETURN(cudaMemcpy(gpu_mu, mu, mem_size_mu, cudaMemcpyHostToDevice))
  OL_CUDA_ERROR_RETURN(cudaMemcpy(gpu_chol_var, chol_var, mem_size_chol_var, cudaMemcpyHostToDevice))
  // execute kernel
  CudaComputeEIGpu <<< grid, threads, num_union*sizeof(double)+num_union*num_union*sizeof(double)+num_union*ei_thread_no*sizeof(double) >>> (gpu_chol_var, gpu_mu, num_union, num_iteration, best, seed, gpu_EI_storage, gpu_random_number_EI, configure_for_test);
  OL_CUDA_ERROR_RETURN(cudaPeekAtLastError())
  // copy gpu_EI_storage back to CPU
  OL_CUDA_ERROR_RETURN(cudaMemcpy(EI_storage, gpu_EI_storage, mem_size_EI_storage, cudaMemcpyDeviceToHost))
  // copy gpu_random_number_EI back to CPU if configure_for_test is on
  if (configure_for_test) {
      int mem_size_random_number_EI = num_iteration * ei_thread_no * ei_block_no * num_union * sizeof(double);
      OL_CUDA_ERROR_RETURN(cudaMemcpy(random_number_EI, gpu_random_number_EI, mem_size_random_number_EI, cudaMemcpyDeviceToHost))
  }
  // average EI_storage
  double ave = 0.0;
  for (int i = 0; i < (ei_thread_no*ei_block_no); ++i) {
      ave += EI_storage[i];
  }
  *ei_val = ave / static_cast<double>(ei_thread_no*ei_block_no);
  return _success;
}

// grad_EI[dim][num_to_sample]
CudaError CudaGetGradEI(double * __restrict__ mu, double * __restrict__ grad_mu, double * __restrict__ chol_var, double * __restrict__ grad_chol_var, double best, int num_union, int num_to_sample, int dim, double * __restrict__ gpu_mu, double * __restrict__ gpu_grad_mu, double * __restrict__ gpu_chol_var, double * __restrict__ gpu_grad_chol_var, double * __restrict__ gpu_grad_EI_storage, unsigned int seed, int num_mc, double * __restrict__ grad_EI, double* __restrict__ gpu_random_number_gradEI, double* __restrict__ random_number_gradEI, bool configure_for_test) {
  CudaError _success = {cudaSuccess, OL_CUDA_STRINGIFY_FILE_AND_LINE, __func__};

  double grad_EI_storage[num_to_sample * dim * grad_ei_thread_no * grad_ei_block_no];
  for (int i = 0; i < num_to_sample*dim; ++i) {
      grad_EI[i] = 0.0;
  }
  // We assign grad_ei_block_no blocks and grad_ei_thread_no threads/block for gradEI computation, so there are (grad_ei_block_no * grad_ei_thread_no) threads in total to execute kernel function in parallel
  dim3 threads(grad_ei_thread_no);
  dim3 grid(grad_ei_block_no);
  int num_iteration = num_mc / (grad_ei_thread_no * grad_ei_block_no) + 1;   // make sure num_iteration is always >= 1

  int mem_size_mu = num_union * sizeof(double);
  int mem_size_grad_mu = num_to_sample * dim * sizeof(double);
  int mem_size_chol_var = num_union * num_union *sizeof(double);
  int mem_size_grad_chol_var = num_to_sample * num_union * num_union * dim * sizeof(double);
  int mem_size_grad_EI_storage= grad_ei_thread_no * grad_ei_block_no * num_to_sample * dim * sizeof(double);

  OL_CUDA_ERROR_RETURN(cudaMemcpy(gpu_mu, mu, mem_size_mu, cudaMemcpyHostToDevice))
  OL_CUDA_ERROR_RETURN(cudaMemcpy(gpu_grad_mu, grad_mu, mem_size_grad_mu, cudaMemcpyHostToDevice))
  OL_CUDA_ERROR_RETURN(cudaMemcpy(gpu_chol_var, chol_var, mem_size_chol_var, cudaMemcpyHostToDevice))
  OL_CUDA_ERROR_RETURN(cudaMemcpy(gpu_grad_chol_var, grad_chol_var, mem_size_grad_chol_var, cudaMemcpyHostToDevice))

  // execute kernel
  // inputs: gpu_mu, gpu_chol_var, gpu_grad_mu, gpu_grad_chol_var, best, num_union, num_to_sample, dim, num_iteration, seed
  // output: gpu_grad_EI_storage
  CudaComputeGradEIGpu <<< grid, threads, mem_size_mu+mem_size_chol_var+mem_size_grad_mu+mem_size_grad_chol_var+num_union*grad_ei_thread_no*2*sizeof(double) >>> (gpu_mu, gpu_chol_var, gpu_grad_mu, gpu_grad_chol_var, best, num_union, num_to_sample, dim, num_iteration, seed, gpu_grad_EI_storage, gpu_random_number_gradEI, configure_for_test);
  OL_CUDA_ERROR_RETURN(cudaPeekAtLastError())

  OL_CUDA_ERROR_RETURN(cudaMemcpy(grad_EI_storage, gpu_grad_EI_storage, mem_size_grad_EI_storage, cudaMemcpyDeviceToHost))
  // copy gpu_random_number_gradEI back to CPU if configure_for_test is on
  if (configure_for_test) {
      int mem_size_random_number_gradEI = num_iteration * grad_ei_thread_no * grad_ei_block_no * num_union * sizeof(double);
      OL_CUDA_ERROR_RETURN(cudaMemcpy(random_number_gradEI, gpu_random_number_gradEI, mem_size_random_number_gradEI, cudaMemcpyDeviceToHost))
  }

  // The code block below extracts grad_EI from grad_EI_storage, which is output from the function
  // "CudaGetGradEI" run on gpu. The way to do that is for each component of grad_EI, we find all
  // the threads calculating the corresponding component and average over the threads.
  for (int n = 0; n < (grad_ei_thread_no*grad_ei_block_no); ++n) {
      for (int i = 0; i < num_to_sample*dim; ++i) {
          grad_EI[i] += grad_EI_storage[n*num_to_sample*dim + i];
      }
  }
  for (int i = 0; i < num_to_sample*dim; ++i) {
      grad_EI[i] /= static_cast<double>(grad_ei_thread_no*grad_ei_block_no);
  }
  return _success;
}

CudaError CudaSetDevice(int devID) {
  CudaError _success = {cudaSuccess, OL_CUDA_STRINGIFY_FILE_AND_LINE, __func__};
  OL_CUDA_ERROR_RETURN(cudaSetDevice(devID))
  OL_CUDA_ERROR_RETURN(cudaDeviceSetCacheConfig(cudaFuncCachePreferShared))
  return _success;
}

}    // end namespace optimal_learning

