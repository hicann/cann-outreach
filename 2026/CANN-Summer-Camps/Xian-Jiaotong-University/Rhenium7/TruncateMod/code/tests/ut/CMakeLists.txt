cmake_minimum_required(VERSION 3.16.0)
project(truncate_mod_ut CXX C)

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
    -DDTYPE_X1=float
    -DDTYPE_X2=float
    -DDTYPE_Y=float
)

set(UT_COMMON_DIR ${CMAKE_CURRENT_SOURCE_DIR}/common)
set(UT_COMMON_SRCS
    ${UT_COMMON_DIR}/infershape_case_executor.cpp
    ${UT_COMMON_DIR}/infershape_context_faker.cpp
    ${UT_COMMON_DIR}/tiling_case_executor.cpp
    ${UT_COMMON_DIR}/tiling_context_faker.cpp
)

# nlohmann/json 依赖（tiling_case_executor.cpp 使用）：CANN opensource 头文件目录
if(EXISTS /home/developer/opensource/json/include)
    set(NLOHMANN_INCLUDE_DIR /home/developer/opensource/json/include)
elseif(EXISTS /opt/home/developer/opensource/json/include)
    set(NLOHMANN_INCLUDE_DIR /opt/home/developer/opensource/json/include)
else()
    message(FATAL_ERROR "nlohmann/json.hpp not found under opensource/json/include")
endif()

set(UT_COMMON_INCLUDE_DIRS
    ${UT_COMMON_DIR}
    ${NLOHMANN_INCLUDE_DIR}
    $ENV{ASCEND_HOME_PATH}/include
    $ENV{ASCEND_HOME_PATH}/aarch64-linux/include
    $ENV{ASCEND_HOME_PATH}/aarch64-linux/include/base
    $ENV{ASCEND_HOME_PATH}/aarch64-linux/include/base/context_builder
    $ENV{ASCEND_HOME_PATH}/aarch64-linux/asc/include
    $ENV{ASCEND_HOME_PATH}/aarch64-linux/asc/include/tiling
    $ENV{ASCEND_HOME_PATH}/aarch64-linux/pkg_inc
    $ENV{ASCEND_HOME_PATH}/aarch64-linux/pkg_inc/op_common
    $ENV{ASCEND_HOME_PATH}/aarch64-linux/pkg_inc/base
    $ENV{ASCEND_HOME_PATH}/aarch64-linux/pkg_inc/exe_graph
    $ENV{ASCEND_HOME_PATH}/aarch64-linux/pkg_inc/graph
)

set(UT_COMMON_LIB_DIRS
    $ENV{ASCEND_HOME_PATH}/lib64
    $ENV{ASCEND_HOME_PATH}/aarch64-linux/lib64
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
