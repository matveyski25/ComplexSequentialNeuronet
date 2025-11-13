add_library(BaseRNN PUBLIC)

target_include_directories(BaseRNN PUBLIC "C")
target_sources(BaseRNN PUBLIC "BaseRNN.cpp")

target_link_libraries(BaseRNN PUBLIC BaseNN)