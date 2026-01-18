add_library(Src STATIC)

add_subdirectory("./Utils")
add_subdirectory("./BaseNN")
add_subdirectory("./RealizationNN")

target_link_libraries(Src PUBLIC RealizationNN)