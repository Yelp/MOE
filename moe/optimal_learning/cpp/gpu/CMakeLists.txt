cmake_minimum_required(VERSION 2.8.9)
find_package(CUDA)
CUDA_INCLUDE_DIRECTORIES(
    ${CUDA_INCLUDE_DIRS}
    /fs/home/jw865/samples/common/inc
    )
set(CUDA_SRCS
    gpp_cuda_math.cu
    )
list(APPEND CUDA_NVCC_FLAGS --gpu-architecture sm_20)

CUDA_ADD_LIBRARY(
    CUDA_MATH
    ${CUDA_SRCS}
    SHARED
    )
