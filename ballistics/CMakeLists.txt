cmake_minimum_required(VERSION 3.0.2)
project(ballistics)

include_directories(${LIBPREDICT_INCLUDE_DIR})

set(SOURCE_FILES
  attitude.h
  libpredict_tle.cpp
  libpredict_tle.h
  system.cpp
  system.h
)

add_library(ballistics STATIC ${SOURCE_FILES})
