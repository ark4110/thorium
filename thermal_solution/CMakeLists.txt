cmake_minimum_required(VERSION 3.0.2)
project(thermal_solution)

set(SOURCE_FILES
  cpu_adams.cpp
  cpu_adams.h
  system.cpp
  system.h
)

add_library(thermal_solution STATIC ${SOURCE_FILES})
