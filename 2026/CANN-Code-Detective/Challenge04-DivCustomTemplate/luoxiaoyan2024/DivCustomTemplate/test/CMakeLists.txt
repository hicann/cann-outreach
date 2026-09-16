# Copyright (c) Huawei Technologies Co., Ltd. 2020. All rights reserved.
cmake_minimum_required(VERSION 3.5.1)
project(acl_execute_div)

add_compile_options(-std=c++11)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "./")

set(INC_PATH $ENV{DDK_PATH})
if (NOT DEFINED ENV{DDK_PATH})
    set(INC_PATH "/usr/local/Ascend/ascend-toolkit/latest")
endif()

set(CUST_PKG_PATH "${INC_PATH}/opp/vendors/customize/op_api")
set(LIB_PATH $ENV{NPU_HOST_LIB})
if (NOT DEFINED ENV{NPU_HOST_LIB})
    set(LIB_PATH "/usr/local/Ascend/ascend-toolkit/latest/${CMAKE_SYSTEM_PROCESSOR}-${CMAKE_SYSTEM_NAME_LOWER}/devlib")
endif()

include_directories(
    ${INC_PATH}/include
    ${CUST_PKG_PATH}/include
)
link_directories(
    ${LIB_PATH}
    ${CUST_PKG_PATH}/lib
)
add_executable(execute_div_op main.cpp)
target_link_libraries(execute_div_op ascendcl cust_opapi acl_op_compiler nnopbase stdc++)
install(TARGETS execute_div_op DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})