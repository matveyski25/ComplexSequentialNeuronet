add_library(BaseRNN INTERFACE)

target_include_directories(BaseRNN INTERFACE "./")

target_link_libraries(BaseRNN INTERFACE BaseNN)