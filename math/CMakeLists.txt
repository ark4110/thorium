cmake_minimum_required(VERSION 3.0.2)
project(math)

set(SOURCE_FILES
  mat.cpp
  mat.h
  Miniball.hpp
  operations.cpp
  operations.h
  ray.cpp
  ray.h
  three_way_compare.cpp
  three_way_compare.h
  triangle.cpp
  triangle.h
  tetrahedron.h
  tetrahedron.cpp
  rod.h
  rod.cpp)

add_library(math STATIC ${SOURCE_FILES})
