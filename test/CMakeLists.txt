cmake_minimum_required(VERSION 3.0.2)
project(tests)

# Download and unpack googletest at configure time
configure_file(CMakeLists.txt.in googletest-download/CMakeLists.txt)
execute_process(COMMAND "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}" .
        WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/test/googletest-download" )
execute_process(COMMAND "${CMAKE_COMMAND}" --build .
        WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/test/googletest-download" )

# Prevent GoogleTest from overriding our compiler/linker options
# when building with Visual Studio
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

# Add googletest directly to our build. This adds
# the following targets: gtest, gtest_main, gmock
# and gmock_main
add_subdirectory("${CMAKE_BINARY_DIR}/googletest-src"
        "${CMAKE_BINARY_DIR}/googletest-build")


include_directories("${gtest_SOURCE_DIR}/include"
        "${gmock_SOURCE_DIR}/include")

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

configure_file(${CMAKE_SOURCE_DIR}/models/dummy_hierarchy.obj ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/models/dummy_hierarchy.obj)
configure_file(${CMAKE_SOURCE_DIR}/models/dummy_hierarchy.mtl ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/models/dummy_hierarchy.mtl)
configure_file(${CMAKE_SOURCE_DIR}/models/dummy_hierarchy.anim ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/models/dummy_hierarchy.anim)
configure_file(${CMAKE_SOURCE_DIR}/models/dummy_hierarchy.task ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/models/dummy_hierarchy.task)

add_executable(tests ${SOURCE_FILES})

target_compile_options(tests PUBLIC $<$<COMPILE_LANGUAGE:CUDA>:-gencode arch=compute_61,code=sm_61>)

# TODO: FIXME
if(MSVC)
  cmake_policy(SET CMP0053 NEW)
  set(Optix_dll_name "optix_prime.6.0.0.dll")
  message(STATUS "Copy ${Optix_dll_name} to directories with binaries")
  configure_file(${OptiX_INSTALL_DIR}/bin64/${Optix_dll_name} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/Debug/${Optix_dll_name} COPYONLY)
  configure_file(${OptiX_INSTALL_DIR}/bin64/${Optix_dll_name} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/Release/${Optix_dll_name} COPYONLY)
  configure_file(${OptiX_INSTALL_DIR}/bin64/${Optix_dll_name} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/RelWithDebInfo/${Optix_dll_name} COPYONLY)
endif()

target_link_libraries(tests gtest gmock_main)
target_link_libraries(tests ballistics ray_caster cuda_ray_caster form_factors import_export subject math thermal_solution thermal_equation emission cuda_emission cuda_misc cuda_equation)
target_link_libraries(tests predict_static)
target_link_libraries(tests ${CUDA_LIBRARIES})
target_link_libraries(tests ${optix_prime_LIBRARY})
