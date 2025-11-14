add_library(LSTM STATIC)

target_include_directories(LSTM PUBLIC "./")
target_sources(LSTM PUBLIC "LSTM.cpp")

target_link_libraries(LSTM PUBLIC BaseRNN)