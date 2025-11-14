add_library(BaseRNN STATIC)

#target_include_directories(BaseRNN PUBLIC "./")
target_include_directories(BaseRNN PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_sources(BaseRNN PUBLIC "BaseRNN.cpp")

target_link_libraries(BaseRNN PUBLIC BaseNN)