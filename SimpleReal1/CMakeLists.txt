add_library(Src PUBLIC)

add_subdirectory("./RealizationNN")

target_link_libraries(Src PUBLIC RealizationNN)