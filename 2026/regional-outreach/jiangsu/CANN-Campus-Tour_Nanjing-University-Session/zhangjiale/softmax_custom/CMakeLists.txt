# CMakeLists.txt - Top-level build for softmax_ascend Ascend C operator
cmake_minimum_required(VERSION 3.14)
project(softmax_ascend CXX)

set(CMAKE_CXX_STANDARD 17)

find_package(AscendC REQUIRED)

add_subdirectory(op_kernel)
add_subdirectory(op_host)
