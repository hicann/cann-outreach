# Copyright (c) Huawei Technologies Co., Ltd. 2020. All rights reserved.

cmake_minimum_required(VERSION 3.5.1)
project(acl_execute_sub)

add_compile_options(-std=c++11)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "./")

set(INC_PATH $ENV{DDK_PATH})

if (NOT DEFINED ENV{DDK_PATH})
    set(INC_PATH "/usr/local/Ascend/ascend-toolkit/latest")
    message(STATUS "set default INC_PATH: ${INC_PATH}")
else ()
    message(STATUS "env INC_PATH: ${INC_PATH}")
endif()

set(CUST_PKG_PATH "${INC_PATH}/opp/vendors/customize/op_api")

set(LIB_PATH $ENV{NPU_HOST_LIB})

if (NOT DEFINED ENV{NPU_HOST_LIB})
    string(TOLOWER "${CMAKE_SYSTEM_NAME}" SYSTEM_NAME_LOWER)
    set(LIB_PATH "/usr/local/Ascend/ascend-toolkit/latest/${CMAKE_SYSTEM_PROCESSOR}-${SYSTEM_NAME_LOWER}/devlib")
    message(STATUS "set default LIB_PATH: ${LIB_PATH}")
else ()
    message(STATUS "env LIB_PATH: ${LIB_PATH}")
endif()

set(BUILD_OUT_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../build_out")

include_directories(
    ${INC_PATH}/include
    ${CUST_PKG_PATH}/include
    ${BUILD_OUT_PATH}/autogen
    ${BUILD_OUT_PATH}/op_host
)

link_directories(
    ${LIB_PATH}
    ${CUST_PKG_PATH}/lib
    ${BUILD_OUT_PATH}/op_host
)

add_executable(execute_sub_op
    main.cpp
)

target_link_libraries(execute_sub_op
    ascendcl
    cust_opapi
    acl_op_compiler
    nnopbase
    stdc++
)

install(TARGETS execute_sub_op DESTINATION ${CMAKE_RUNTIME_OUTPUT_DIRECTORY})
