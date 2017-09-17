cmake_minimum_required(VERSION 3.0.2)
project(cuda_misc)

set(SOURCE_FILES
    cuda_scene_engine.h
    cuda_scene_engine.cpp
    cuda_scene_engine.cuh
    cuda_scene_engine.cu
    optix_hit_converter.h
    optix_hit_converter.cuh
    optix_hit_converter.cu
    )

cuda_add_library(cuda_misc STATIC ${SOURCE_FILES})
