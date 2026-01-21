cmake_minimum_required(VERSION 3.16)

project(CSN LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CXX_CLANG_TIDY "clang-tidy")
    add_compile_options(-fsanitize=address,undefined)
    add_link_options(-fsanitize=address,undefined)
endif()

add_subdirectory("./Lib")
add_subdirectory("./SimpleReal1")

add_executable(Main)

target_link_libraries(Main PUBLIC Src)

target_include_directories(Main PUBLIC "./SimpleReal1")
target_sources(Main PUBLIC "./SimpleReal1/main.cpp")