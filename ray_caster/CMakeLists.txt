cmake_minimum_required(VERSION 3.0.2)
project(ray_caster)

set(SOURCE_FILES
  naive_cpu.cpp
  naive_cpu.h
  system.cpp
  system.h
)

add_library(ray_caster STATIC ${SOURCE_FILES})

