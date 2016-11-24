cmake_minimum_required(VERSION 3.0.2)
project(emission)

set(SOURCE_FILES
  malley_cpu.cpp
  parallel_rays_cpu.cpp
  planet_emission.cpp
  system.cpp
  malley_cpu.h
  parallel_rays_cpu.h
  planet_emission.h
  system.h
)

add_library(emission STATIC ${SOURCE_FILES})
