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

set_target_properties(cuda_equation PROPERTIES CUDA_ARCHITECTURES ${Thorium_CUDA_Compute_Config})
set_target_properties(cuda_equation PROPERTIES CUDA_STANDARD 20)

if (CMAKE_BUILD_TYPE MATCHES Release)
    target_compile_options(cuda_equation PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
            --generate-line-info
            >)
endif()

if (CMAKE_BUILD_TYPE MATCHES Debug)
    target_compile_options(cuda_equation PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
            -G
            >)
endif()
