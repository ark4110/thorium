cmake_minimum_required(VERSION 3.0.2)
project(form_factors)

set(SOURCE_FILES
    cpu_system.h
    cpu_system.cpp
    system.h
    system.cpp
    )

add_library(form_factors STATIC ${SOURCE_FILES})

