cmake_minimum_required(VERSION 3.0.2)
project(perf_test)

set(SOURCE_FILES
    perfTest.cpp
   )

add_executable(perf_test ${SOURCE_FILES})

target_link_libraries(perf_test import_export thermal_equation thermal_solution emission cuda_emission ray_caster cuda_ray_caster optix_ray_caster form_factors subject math ballistics cuda_misc cuda_equation)
target_link_libraries(perf_test predict_static)
target_link_libraries(perf_test ${CUDA_LIBRARIES})
target_link_libraries(perf_test ${optix_prime_LIBRARY})
