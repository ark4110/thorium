cmake_minimum_required(VERSION 3.0.2)
project(cuda_ray_caster)

set(SOURCE_FILES
    cuda_system.h
    cuda_system.cpp
    system.cuh
    system.cu
    )

cuda_add_library(cuda_ray_caster STATIC ${SOURCE_FILES})

