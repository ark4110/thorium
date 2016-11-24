cmake_minimum_required(VERSION 3.0.2)
project (Thorium)

set(CMAKE_MODULE_PATH
  "${CMAKE_SOURCE_DIR}/ext/optix/CMake"
  ${CMAKE_MODULE_PATH}
  )

find_package(OptiX REQUIRED)

include_directories("/usr/local/cuda/samples/common/inc")

add_subdirectory("emission")
add_subdirectory("math")
add_subdirectory("ray_caster")
add_subdirectory("cuda_ray_caster")
add_subdirectory("cuda_emission")
add_subdirectory("optix_ray_caster")
add_subdirectory("form_factors")
add_subdirectory("import_export")
add_subdirectory("test")
add_subdirectory("thermal_equation")
add_subdirectory("thermal_solution")
add_subdirectory("subject")
add_subdirectory("controller")
add_subdirectory("perfTest")
