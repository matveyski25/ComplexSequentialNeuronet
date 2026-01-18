add_library(BaseNN STATIC)

target_include_directories(BaseNN PUBLIC "./" "./../RealizationMatrix")
target_sources(BaseNN PUBLIC "BaseNN.cpp")

add_subdirectory("./BaseRNN")

target_link_libraries(BaseNN PUBLIC Lib Utils)