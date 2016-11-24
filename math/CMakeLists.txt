cmake_minimum_required(VERSION 3.0.2)
project(math)

set(SOURCE_FILES
  grid.cpp
  grid.h
  mat.cpp
  mat.h
  operations.cpp
  operations.h
  ray.cpp
  ray.h
  three_way_compare.cpp
  three_way_compare.h
  triangle.cpp
  triangle.h
)

add_library(math STATIC ${SOURCE_FILES})
