cmake_minimum_required(VERSION 3.0.2)
project(tests)

enable_testing()
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

set(SOURCE_FILES
  adams_solution.cpp
  animation.cpp
  ballistics.cpp
  bounding_volume.cpp
  conductive_solution.cpp
  emission.cpp
  export.cpp
  form_factors.cpp
  heat_source.cpp
  import.cpp
  main.cpp
  math_matrix.cpp
  math_ray.cpp
  mesh_graph.cpp
  radiance_solution.cpp
  ray_casting.cpp
  test_scene.h
  test_scene.cpp
  triangle_math.cpp
    )


configure_file(${CMAKE_SOURCE_DIR}/models/parallel_planes.obj ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/models/parallel_planes.obj)
configure_file(${CMAKE_SOURCE_DIR}/models/unknown_material.obj ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/models/unknown_material.obj)
configure_file(${CMAKE_SOURCE_DIR}/models/test_material.mtl ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/models/test_material.mtl)
configure_file(${CMAKE_SOURCE_DIR}/models/test_material.obj ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/models/test_material.obj)
configure_file(${CMAKE_SOURCE_DIR}/models/zarya.orbit ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/models/zarya.orbit)

configure_file(${CMAKE_SOURCE_DIR}/models/test_hierarchy.obj ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/models/test_hierarchy.obj)
configure_file(${CMAKE_SOURCE_DIR}/models/test_hierarchy.mtl ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/models/test_hierarchy.mtl)
configure_file(${CMAKE_SOURCE_DIR}/models/test_hierarchy.anim ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/models/test_hierarchy.anim)
configure_file(${CMAKE_SOURCE_DIR}/models/test_hierarchy.task ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/models/test_hierarchy.task)

add_executable(tests ${SOURCE_FILES})

target_link_libraries(tests ${GTEST_BOTH_LIBRARIES})
target_link_libraries(tests ballistics ray_caster cuda_ray_caster form_factors import_export subject math thermal_solution thermal_equation emission cuda_emission cuda_misc cuda_equation)
target_link_libraries(tests predict_static)
target_link_libraries(tests ${CUDA_LIBRARIES})
target_link_libraries(tests ${optix_prime_LIBRARY})
