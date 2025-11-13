add_library(RNN INTERFACE)

add_subdirectory("./LSTM")

target_link_libraries(RNN INTERFACE LSTM)