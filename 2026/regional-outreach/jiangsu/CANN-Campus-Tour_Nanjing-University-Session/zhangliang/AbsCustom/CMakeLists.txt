# CMakeLists.txt - Top-level build for abs_ascend Ascend C operator
cmake_minimum_required(VERSION 3.14)
project(abs_ascend CXX)

set(CMAKE_CXX_STANDARD 17)

# Find Ascend C package
find_package(AscendC REQUIRED)

# Build kernel
add_subdirectory(op_kernel)

# Build host-side operator registration
add_subdirectory(op_host)
