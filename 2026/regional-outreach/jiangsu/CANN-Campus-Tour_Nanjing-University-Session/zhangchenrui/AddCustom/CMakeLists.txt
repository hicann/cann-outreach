cmake_minimum_required(VERSION 3.16)
project(AddCustom CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 包含自定义 cmake 模块
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)
include(func)

# 查找 Ascend C 编译支持
find_package(AscendC REQUIRED)

# 添加子目录
add_subdirectory(op_host)
add_subdirectory(op_kernel)
add_subdirectory(op_proto)
add_subdirectory(test)