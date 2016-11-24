cmake_minimum_required(VERSION 3.0.2)
project(subject)

set(SOURCE_FILES
  generator.h
  objects.cpp
  spherial_generator.cpp
  system.h
  cube_generator.cpp
  mesh_graph.cpp
  objects.h
  system.cpp
  unify_normals.cpp)

add_library(subject STATIC ${SOURCE_FILES})
