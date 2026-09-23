cmake_minimum_required(VERSION 3.16.0)
project(add_ut CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT DEFINED ENV{ASCEND_HOME_PATH})
    message(FATAL_ERROR "ASCEND_HOME_PATH environment variable is not set!")
endif()

message(STATUS "ASCEND_HOME_PATH: $ENV{ASCEND_HOME_PATH}")

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/BuildGoogleTest.cmake)

# DTYPE 宏定义（根据算子原型和测试用例生成）
add_definitions(
    -DDTYPE_X=float
    -DDTYPE_Y=float
)

# 按实际存在的目录自动适配 host 侧路径：aarch64-linux / x86_64-linux（8.x 及部分 9.x 布局）/ host-libs（9.x 新布局）
if(EXISTS $ENV{ASCEND_HOME_PATH}/aarch64-linux)
    set(OP_HOST_LIB_PATH $ENV{ASCEND_HOME_PATH}/aarch64-linux)
elseif(EXISTS $ENV{ASCEND_HOME_PATH}/x86_64-linux)
    set(OP_HOST_LIB_PATH $ENV{ASCEND_HOME_PATH}/x86_64-linux)
else()
    set(OP_HOST_LIB_PATH $ENV{ASCEND_HOME_PATH}/host-libs)
endif()

set(UT_COMMON_DIR ${CMAKE_CURRENT_SOURCE_DIR}/common)
set(UT_COMMON_SRCS
    ${UT_COMMON_DIR}/infershape_case_executor.cpp
    ${UT_COMMON_DIR}/infershape_context_faker.cpp
    ${UT_COMMON_DIR}/tiling_case_executor.cpp
    ${UT_COMMON_DIR}/tiling_context_faker.cpp
)

set(UT_COMMON_INCLUDE_DIRS
    ${UT_COMMON_DIR}
    $ENV{ASCEND_HOME_PATH}/include
    ${OP_HOST_LIB_PATH}/include
    ${OP_HOST_LIB_PATH}/include/base
    ${OP_HOST_LIB_PATH}/include/base/context_builder
    # CANN 9.x 中 context_builder 头文件位于 CANN 根目录的顶层 include 下
    $ENV{ASCEND_HOME_PATH}/include/base
    $ENV{ASCEND_HOME_PATH}/include/base/context_builder
    ${OP_HOST_LIB_PATH}/asc/include
    ${OP_HOST_LIB_PATH}/asc/include/tiling
    ${OP_HOST_LIB_PATH}/pkg_inc
    ${OP_HOST_LIB_PATH}/pkg_inc/op_common
    ${OP_HOST_LIB_PATH}/pkg_inc/base
    ${OP_HOST_LIB_PATH}/pkg_inc/exe_graph
    ${OP_HOST_LIB_PATH}/pkg_inc/graph
)

set(UT_COMMON_LIB_DIRS
    $ENV{ASCEND_HOME_PATH}/lib64
    ${OP_HOST_LIB_PATH}/lib64
)

add_compile_options(
    -Wall
    -Wno-deprecated-declarations
    -Wno-unused-variable
    -fno-access-control
)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")

add_subdirectory(op_host)

if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/../../op_api)
    add_subdirectory(op_api)
endif()
add_subdirectory(op_kernel)
