cmake_minimum_required(VERSION 3.0.2)
project(cuda_ray_caster CXX CUDA)

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

add_library(cuda_ray_caster STATIC ${SOURCE_FILES})
set_target_properties(cuda_ray_caster PROPERTIES CUDA_ARCHITECTURES ${Thorium_CUDA_Compute_Config})
set_target_properties(cuda_ray_caster PROPERTIES CUDA_STANDARD 20)

if (CMAKE_BUILD_TYPE MATCHES Release)
  target_compile_options(cuda_ray_caster PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
          --generate-line-info
          >)
endif()

if (CMAKE_BUILD_TYPE MATCHES Debug)
  target_compile_options(cuda_ray_caster PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
          -G
          >)
endif()
