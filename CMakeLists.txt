cmake_minimum_required(VERSION 3.0.2)
project (Thorium CXX CUDA)

set(CMAKE_MODULE_PATH
  "${CMAKE_SOURCE_DIR}/ext/optix/CMake"
  ${CMAKE_MODULE_PATH}
  )

set(OptiX_INSTALL_DIR "${CMAKE_SOURCE_DIR}/ext/optix")

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
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR
		CMAKE_CXX_COMPILER_ID STREQUAL "Clang" OR
		CMAKE_CXX_COMPILER_ID STREQUAL "AppleClang")
	# Enable standard warnings.
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")

	# Enable additional warnings that are not included in -Wall and -Wextra.
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wcast-align")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wcast-qual")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wswitch-default")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wuninitialized")
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wold-style-cast")

	# Disable warnings that produce more headaches than use.
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-parameter")
endif()

set(CMAKE_INCLUDE_CURRENT_DIR ON)
include_directories(".")
include_directories("${CMAKE_SOURCE_DIR}/ext/cuda_samples/common/inc")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -march=native -Wall")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Ofast")

#set(CMAKE_BUILD_TYPE Release)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ../lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ../bin)

set(Thorium_CUDA_Compute_Config arch=compute_61,code=sm_61)

#find_package(CUDA REQUIRED)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-lineinfo -gencode arch=compute_61,code=sm_61)
find_package(OptiX REQUIRED)
include_directories("/usr/local/cuda/samples/common/inc")

#find_package(OpenMP)
#if (OPENMP_FOUND)
#    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
#    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
#endif()

add_subdirectory("ext/libpredict")
set(LIBPREDICT_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/ext/libpredict/include")

add_subdirectory("ballistics")
add_subdirectory("emission")
add_subdirectory("math")
add_subdirectory("ray_caster")
add_subdirectory("cuda_ray_caster")
add_subdirectory("cuda_emission")
add_subdirectory("cuda_misc")
add_subdirectory("cuda_equation")
add_subdirectory("form_factors")
add_subdirectory("import_export")
add_subdirectory("thermal_equation")
add_subdirectory("thermal_solution")
add_subdirectory("test")
add_subdirectory("subject")
add_subdirectory("controller")
