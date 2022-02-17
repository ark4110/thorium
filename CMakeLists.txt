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

include_directories(ext/optix/include)
find_library(optix_prime_LIBRARY
		NAMES optix_prime
		PATHS ext/optix/lib64)
find_path(OptiX_INCLUDE
  NAMES optix.h
  PATHS ext/optix/include
  NO_DEFAULT_PATH
  )

set(CMAKE_INCLUDE_CURRENT_DIR ON)
include_directories(".")
include_directories("${CMAKE_SOURCE_DIR}/ext/cuda_samples/common/inc")
include_directories("${CUDA_INCLUDE_DIRS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -march=native -Wall")
#set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -Ofast")

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ../lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ../bin)

set(Thorium_CUDA_Compute_Config "61;70;86")
set(CUDA_ARCHITECTURES ${Thorium_CUDA_Compute_Config})

set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; --generate-line-info)

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
