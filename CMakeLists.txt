cmake_minimum_required(VERSION 2.8)
project(body3 C)

find_package(MPI REQUIRED)

#add_definitions("-std=c++14")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${MPI_COMPILE_FLAGS} -g -Wall -Wextra -O0")
set(CMAKE_C_LINK_FLAGS "${CMAKE_C_LINK_FLAGS} ${MPI_LINK_FLAGS} -lm")

set(CMAKE_CXX_COMPILE_FLAGS "${CMAKE_CXX_COMPILE_FLAGS} ${MPI_COMPILE_FLAGS} -g -Wall -Wextra -O0")
set(CMAKE_CXX_LINK_FLAGS ${CMAKE_CXX_LINK_FLAGS} ${MPI_LINK_FLAGS})

include_directories(${MPI_INCLUDE_PATH})
include_directories(src)

add_executable(
    body3
    src/main.c
    src/common.c
    src/compute_accelerations.c
)

target_link_libraries(
    body3
    ${MPI_LIBRARIES}
)
