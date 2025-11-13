add_library(Lib INTERFACE)

add_subdirectory("./Eigen")

target_link_libraries(Lib INTERFACE Eigen)