cmake_minimum_required(VERSION 3.0.2)
project(controller)

set(SOURCE_FILES
    controller.cpp
   )

add_executable(controller ${SOURCE_FILES})

# TODO: FIXME
if(MSVC)
  cmake_policy(SET CMP0053 NEW)
  set(Optix_dll_name "optix_prime.6.0.0.dll")
  configure_file(${OptiX_INSTALL_DIR}/bin64/${Optix_dll_name} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/Debug/${Optix_dll_name} COPYONLY)
  configure_file(${OptiX_INSTALL_DIR}/bin64/${Optix_dll_name} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/Release/${Optix_dll_name} COPYONLY)
  configure_file(${OptiX_INSTALL_DIR}/bin64/${Optix_dll_name} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/RelWithDebInfo/${Optix_dll_name} COPYONLY)
endif()

target_link_libraries(controller ray_caster cuda_ray_caster import_export thermal_equation thermal_solution form_factors subject emission cuda_emission ballistics math cuda_misc cuda_equation)
target_link_libraries(controller predict_static)
target_link_libraries(controller ${CUDA_LIBRARIES})
target_link_libraries(controller ${optix_prime_LIBRARY})
