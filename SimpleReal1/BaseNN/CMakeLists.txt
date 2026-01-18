add_library(BaseNN INTERFACE)

target_include_directories(BaseNN INTERFACE "./" "./../RealizationMatrix")
#target_sources(BaseNN PUBLIC "BaseNN.cpp")

add_subdirectory("./BaseRNN")

target_link_libraries(BaseNN INTERFACE Lib Utils)