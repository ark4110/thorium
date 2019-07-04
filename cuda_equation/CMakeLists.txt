cmake_minimum_required(VERSION 3.0.2)
project(cuda_equation CXX CUDA)

set(SOURCE_FILES
    cuda_equation.h
    cuda_equation.cpp
    radiance_cuda.h
    radiance_cuda.cuh
    radiance_cuda.cu
    )

add_library(cuda_equation STATIC ${SOURCE_FILES})

target_compile_options(cuda_equation PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-gencode ${Thorium_CUDA_Compute_Config}>)