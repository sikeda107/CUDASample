cmake_minimum_required(VERSION 2.8)
project(MyCudaProject)

# Create Main project
set(SOURCES
    src/main.cpp
)

set(HEADERS
    # src/main.h
)

add_executable(Main
    ${SOURCES}
    ${HEADERS}
)

# Add sub project
add_subdirectory(CudaVecDouble)
target_include_directories(Main PUBLIC CudaVecDouble)
target_link_libraries(Main CudaVecDouble)
