cmake_minimum_required(VERSION 3.0.2)
project(cuda_ray_caster)

set(SOURCE_FILES
    cuda_system.h
    cuda_system.cpp
    system.cuh
    system.cu
    task_relocation.h
    task_relocation.cpp
    optix_system.h
    optix_system.cpp
    optix_hit_converter.h
    optix_hit_converter.cuh
    optix_hit_converter.cu
    )

include_directories(
  "${OptiX_INCLUDE}"
  )

cuda_add_library(cuda_ray_caster STATIC ${SOURCE_FILES})

