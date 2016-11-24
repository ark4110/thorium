cmake_minimum_required(VERSION 3.0.2)
project(thermal_equation)

set(SOURCE_FILES
  conductive_cpu.cpp
  conductive_cpu.h
  factory.cpp
  form_factors.cpp
  form_factors.h
  heat_source_cpu.cpp
  heat_source_cpu.h
  radiance_cpu.cpp
  radiance_cpu.h
  system.cpp
  system.h
)

add_library(thermal_equation STATIC ${SOURCE_FILES})
