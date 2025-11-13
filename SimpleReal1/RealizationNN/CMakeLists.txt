add_library(RealizationNN INTERFACE)

add_subdirectory("./RNN")

target_link_libraries(RealizationNN INTERFACE RNN)