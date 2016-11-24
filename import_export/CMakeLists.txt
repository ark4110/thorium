cmake_minimum_required(VERSION 3.0.2)
project(import_export)

set(SOURCE_FILES
  csv_export.cpp
  csv_export.h
  obj_export.cpp
  obj_export.h
  obj_import.cpp
  obj_import.h
)

add_library(import_export STATIC ${SOURCE_FILES})

