add_library(BiLSTM STATIC)

target_include_directories(BiLSTM PUBLIC "./")
target_sources(BiLSTM PUBLIC "BiLSTM.cpp")

target_link_libraries(BiLSTM PUBLIC LSTM)