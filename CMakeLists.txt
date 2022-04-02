cmake_minimum_required(VERSION 3.0.2)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
project (Thorium CXX CUDA)

if(MSVC)
	# For the moment, suppress all warnings when building with MSVC on Windows
	# because there are too many warnings that clutter the build output (#106).
	# We should investigate the warnings, fix them, and then enable their
	# emission (e.g. by replacing /W0 with /W3 in the code below).
	if(CMAKE_CXX_FLAGS MATCHES "/W[0-4]")
		string(REGEX REPLACE "/W[0-4]" "/W0" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
	else()
		set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /W0")
	endif()
endif()

find_package(CUDA REQUIRED)

set(CMAKE_CUDA_ARCHITECTURES "86")
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

include_directories(ext/optix/include)
include_directories(ext/libpredict/include)

find_library(optix_prime_LIBRARY
		NAMES optix_prime
		PATHS ext/optix/lib64)
find_path(OptiX_INCLUDE
		NAMES optix.h
		PATHS ext/optix/include
		NO_DEFAULT_PATH
		)

add_subdirectory("ext/libpredict")
set(LIBPREDICT_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/ext/libpredict/include")

set(CMAKE_INCLUDE_CURRENT_DIR ON)
include_directories("${CUDA_INCLUDE_DIRS}")

#set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)
#set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

include(FetchContent)
FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/609281088cfefc76f9d0ce82e1ff6c30cc3591e5.zip
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

add_library(ballistics
		ballistics/attitude.cpp
  		ballistics/attitude.h
  		ballistics/libpredict_tle.cpp
  		ballistics/libpredict_tle.h
  		ballistics/static_prediction.cpp
  		ballistics/static_prediction.h
  		ballistics/system.cpp
  		ballistics/system.h
  		ballistics/tle_orbit.cpp
  		ballistics/tle_orbit.h
		)

add_library(thorium_cuda
		cuda_emission/cuda_emission.h
    	cuda_emission/cuda_emission.cpp
    	cuda_emission/malley_cuda.cuh
    	cuda_emission/malley_cuda.cu
    	cuda_emission/parallel_cuda.cuh
    	cuda_emission/parallel_cuda.cu
		cuda_equation/cuda_equation.h
    	cuda_equation/cuda_equation.cpp
    	cuda_equation/radiance_cuda.h
    	cuda_equation/radiance_cuda.cuh
    	cuda_equation/radiance_cuda.cu
		cuda_misc/cuda_data_util.cuh
    	cuda_misc/cuda_scene_engine.h
    	cuda_misc/cuda_scene_engine.cpp
    	cuda_misc/cuda_scene_engine.cuh
    	cuda_misc/cuda_scene_engine.cu
    	cuda_misc/cuda_util.h
    	cuda_misc/cuda_util.cpp
		cuda_ray_caster/cuda_system.h
    	cuda_ray_caster/cuda_system.cpp
    	cuda_ray_caster/system.cuh
    	cuda_ray_caster/system.cu
    	cuda_ray_caster/task_relocation.h
    	cuda_ray_caster/task_relocation.cpp
    	cuda_ray_caster/optix_system.h
    	cuda_ray_caster/optix_system.cpp
    	cuda_ray_caster/optix_hit_converter.h
    	cuda_ray_caster/optix_hit_converter.cuh
    	cuda_ray_caster/optix_hit_converter.cu
		)
set_target_properties(thorium_cuda PROPERTIES CUDA_STANDARD 20)
target_compile_options(thorium_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
        --use_fast_math
        >)

if (CMAKE_BUILD_TYPE MATCHES Release)
    target_compile_options(thorium_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
            --generate-line-info
            >)
endif()
if (CMAKE_BUILD_TYPE MATCHES Debug)
    target_compile_options(thorium_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
            -G
            >)
endif()

add_library(thorium_lib
		import_export/csv_export.cpp
		import_export/csv_export.h
		import_export/obj_export.cpp
		import_export/obj_export.h
		import_export/obj_import.cpp
		import_export/obj_import.h
		import_export/text_coords.h
		import_export/text_coords.cpp
		form_factors/cpu_system.h
		form_factors/cpu_system.cpp
		form_factors/system.h
		form_factors/system.cpp
		emission/malley_cpu.cpp
		emission/parallel_rays_cpu.cpp
		emission/planet_emission.cpp
		emission/system.cpp
		emission/malley_cpu.h
		emission/parallel_rays_cpu.h
		emission/planet_emission.h
		emission/system.h
		math/mat.cpp
		math/mat.h
		math/Miniball.hpp
		math/operations.cpp
		math/operations.h
		math/ray.cpp
		math/ray.h
		math/three_way_compare.cpp
		math/three_way_compare.h
		math/triangle.cpp
		math/triangle.h
		math/tetrahedron.h
		math/tetrahedron.cpp
		math/rod.h
		math/rod.cpp
		subject/cube_generator.cpp
		subject/generator.h
		subject/mesh_graph.cpp
		subject/objects.h
		subject/objects.cpp
		subject/operations.h
		subject/operations.cpp
		subject/spherial_generator.cpp
		subject/unify_normals.cpp
		subject/system.h
		subject/system.cpp
		thermal_equation/conductive_cpu.h
		thermal_equation/conductive_cpu.cpp
		thermal_equation/factory.cpp
		thermal_equation/form_factors.h
		thermal_equation/form_factors.cpp
		thermal_equation/heat_source_cpu.h
		thermal_equation/heat_source_cpu.cpp
		thermal_equation/radiance_cpu.h
		thermal_equation/radiance_cpu.cpp
		thermal_equation/system.h
		thermal_equation/system.cpp
		thermal_solution/cpu_adams.h
		thermal_solution/cpu_adams.cpp
		thermal_solution/system.h
		thermal_solution/system.cpp
		ray_caster/naive_cpu.h
		ray_caster/naive_cpu.cpp
		ray_caster/system.h
		ray_caster/system.cpp
		)

add_executable(thorium_engine
		controller/controller.cpp
		)
# TODO: FIXME
if(MSVC)
	cmake_policy(SET CMP0053 NEW)
	set(Optix_dll_name "optix_prime.6.0.0.dll")
	configure_file(${OptiX_INSTALL_DIR}/bin64/${Optix_dll_name} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/Debug/${Optix_dll_name} COPYONLY)
	configure_file(${OptiX_INSTALL_DIR}/bin64/${Optix_dll_name} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/Release/${Optix_dll_name} COPYONLY)
	configure_file(${OptiX_INSTALL_DIR}/bin64/${Optix_dll_name} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/RelWithDebInfo/${Optix_dll_name} COPYONLY)
endif()

add_executable(tests
		test/adams_solution.cpp
		test/animation.cpp
		test/ballistics.cpp
		test/bounding_volume.cpp
		test/conductive_solution.cpp
		test/emission.cpp
		test/export.cpp
		test/form_factors.cpp
		test/heat_source.cpp
		test/import.cpp
		test/math_matrix.cpp
		test/math_ray.cpp
		test/mesh_graph.cpp
		test/radiance_solution.cpp
		test/ray_casting.cpp
		test/test_scene.h
		test/test_scene.cpp
		test/triangle_math.cpp
		)
target_link_libraries(ballistics
		predict_static)
target_link_libraries(thorium_cuda
		${optix_prime_LIBRARY}
		cuda
		cudadevrt
		)
target_link_libraries(thorium_lib
		ballistics
		thorium_cuda)
target_link_libraries(thorium_engine
		thorium_lib)
target_link_libraries(tests
		gtest_main
		thorium_lib
		)
add_subdirectory(test)

include(GoogleTest)
gtest_discover_tests(tests)
