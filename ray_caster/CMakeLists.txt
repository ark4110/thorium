cmake_minimum_required(VERSION 3.0.2)
project(ray_caster)

set(SOURCE_FILES
  naive_cpu.cpp
  naive_cpu.h
  system.cpp
  system.h
  zgrid_cpu.cpp
  zgrid_cpu.h
)

add_library(ray_caster STATIC ${SOURCE_FILES})

