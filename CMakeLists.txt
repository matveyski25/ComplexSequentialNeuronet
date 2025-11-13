cmake_minimum_required(VERSION 3.16)

project(CSN LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory("./SimpleReal1")

add_executable(Main PUBLIC)

target_include_directories(Main PUBLIC "./SimpleReal1")
target_sources(Main PUBLIC "./SimpleReal1/main.cpp")