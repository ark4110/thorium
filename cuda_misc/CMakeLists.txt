cmake_minimum_required(VERSION 3.0.2)
project(cuda_misc CXX CUDA)

set(SOURCE_FILES
    cuda_data_util.cuh
    cuda_scene_engine.h
    cuda_scene_engine.cpp
    cuda_scene_engine.cuh
    cuda_scene_engine.cu
    cuda_util.h
    cuda_util.cpp
    )

add_library(cuda_misc STATIC ${SOURCE_FILES})

set_target_properties(cuda_misc PROPERTIES CUDA_ARCHITECTURES ${Thorium_CUDA_Compute_Config})
set_target_properties(cuda_misc PROPERTIES CUDA_STANDARD 20)

if (CMAKE_BUILD_TYPE MATCHES Release)
    target_compile_options(cuda_misc PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
            --generate-line-info
            >)
endif()

if (CMAKE_BUILD_TYPE MATCHES Debug)
    target_compile_options(cuda_misc PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
            -G
            >)
endif()
