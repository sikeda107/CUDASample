#cmake_minimum_required(VERSION 3.13)
cmake_minimum_required(VERSION 2.8)
project(HELLO)

#set(CMAKE_C_STANDARD 11)
set(CMAKE_C_FLAGS "-O2 -Wall")
set(SOURCES c_src/main.c)
#set(HEADERS c_src/main.h)
add_executable(HELLO
        ${SOURCES}
#        ${HEDERS}
        )

add_subdirectory(cuda_src)
target_include_directories(HELLO PUBLIC sumMatrix2D_1)
target_link_libraries(HELLO sumMatrix2D_1)