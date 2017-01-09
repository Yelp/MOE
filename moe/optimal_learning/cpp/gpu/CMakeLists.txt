cmake_minimum_required(VERSION 2.8.9)
find_package(CUDA 5.0 REQUIRED)

if (DEFINED MOE_CUDA_SDK_INCLUDE_DIRS)
    if (IS_DIRECTORY ${MOE_CUDA_SDK_INCLUDE_DIRS})
        CUDA_INCLUDE_DIRECTORIES(
            ${CUDA_INCLUDE_DIRS}
            ${MOE_CUDA_SDK_INCLUDE_DIRS}
            )
    else()
        message(FATAL_ERROR "MOE_CUDA_SDK_INCLUDE_DIRS is not a valid directory: " ${MOE_CUDA_SDK_INCLUDE_DIRS})
    endif()
endif()

# Provide a list of actual .cu files
set(CUDA_SRCS
    gpp_cuda_math.cu
    )

# Compiler flags - eg specify different compute capabilites 
list(APPEND CUDA_NVCC_FLAGS --gpu-architecture sm_20)

CUDA_ADD_LIBRARY(
    OL_GPU
    ${CUDA_SRCS}
    SHARED
    )
