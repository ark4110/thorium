cmake_minimum_required(VERSION 3.0.2)
project(cuda_equation)

set(SOURCE_FILES
    cuda_equation.h
    cuda_equation.cpp
    radiance_cuda.h
    radiance_cuda.cuh
    radiance_cuda.cu
    )

cuda_add_library(cuda_equation STATIC ${SOURCE_FILES})
