cmake_minimum_required(VERSION 3.0.2)
project(cuda_emission)

set(SOURCE_FILES
    cuda_emission.h
    cuda_emission.cpp
    malley_cuda.cuh
    system.cuh
    system.cu
    )

cuda_add_library(cuda_emission STATIC ${SOURCE_FILES})

