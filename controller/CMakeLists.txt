cmake_minimum_required(VERSION 3.0.2)
project(controller)

set(SOURCE_FILES
    controller.cpp
   )

add_executable(controller ${SOURCE_FILES})

target_link_libraries(controller ray_caster cuda_ray_caster optix_ray_caster import_export thermal_equation thermal_solution form_factors subject emission cuda_emission ballistics math cuda_misc)
target_link_libraries(controller predict_static)
target_link_libraries(controller ${CUDA_LIBRARIES})
target_link_libraries(controller ${optix_prime_LIBRARY})
