cmake_minimum_required(VERSION 3.0.2)
project(optix_ray_caster)

set(CMAKE_INCLUDE_PATH "")
set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ../lib)

include_directories(
  "${OptiX_INCLUDE}"
  )

set(SOURCE_FILES
optix_system.h
optix_system.cpp
    )

add_library(optix_ray_caster STATIC ${SOURCE_FILES})
