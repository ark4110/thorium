cmake_minimum_required(VERSION 3.0.2)
project(cuda_emission CXX CUDA)

set(SOURCE_FILES
    cuda_emission.h
    cuda_emission.cpp
    malley_cuda.cuh
    malley_cuda.cu
    parallel_cuda.cuh
    parallel_cuda.cu
    )

add_library(cuda_emission STATIC ${SOURCE_FILES})

target_compile_options(cuda_emission PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode ${Thorium_CUDA_Compute_Config}>)