add_library(BaseNN PUBLIC)

target_include_directories(BaseNN PUBLIC "./" "./../RealizationMatrix")
target_sources(BaseNN PUBLIC "BaseNN.cpp")

add_subdirectory("./BaseRNN")