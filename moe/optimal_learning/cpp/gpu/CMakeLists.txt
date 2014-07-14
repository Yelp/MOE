cmake_minimum_required(VERSION 2.8.9)
find_package(CUDA 5.0 REQUIRED)
CUDA_INCLUDE_DIRECTORIES(
    ${CUDA_INCLUDE_DIRS}
    ${MOE_CUDA_SDK_INCLUDE_DIRS}
    )

# Provide a list of actual .cu files
set(CUDA_SRCS
    gpp_cuda_math.cu
    )

# Possibility to stear compiler flags - eg different compute capabilites 
list(APPEND CUDA_NVCC_FLAGS --gpu-architecture sm_20)

CUDA_ADD_LIBRARY(
    GPU
    ${CUDA_SRCS}
    SHARED
    )
