cmake_minimum_required(VERSION 3.0.2)
project (Thorium)

set(CMAKE_MODULE_PATH
  "${CMAKE_SOURCE_DIR}/ext/optix/CMake"
  ${CMAKE_MODULE_PATH}
  )

set(CMAKE_INCLUDE_CURRENT_DIR ON)
include_directories(".")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ../lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ../bin)

find_package(CUDA REQUIRED)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3 -gencode arch=compute_20,code=sm_20 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_50,code=sm_50)
find_package(OptiX REQUIRED)
include_directories("/usr/local/cuda/samples/common/inc")

add_subdirectory("ext/libpredict")
set(LIBPREDICT_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/ext/libpredict/include")

add_subdirectory("ballistics")
add_subdirectory("emission")
add_subdirectory("math")
add_subdirectory("ray_caster")
add_subdirectory("cuda_ray_caster")
add_subdirectory("cuda_emission")
add_subdirectory("cuda_misc")
add_subdirectory("optix_ray_caster")
add_subdirectory("form_factors")
add_subdirectory("import_export")
add_subdirectory("thermal_equation")
add_subdirectory("thermal_solution")
add_subdirectory("test")
add_subdirectory("subject")
add_subdirectory("controller")
add_subdirectory("perfTest")
