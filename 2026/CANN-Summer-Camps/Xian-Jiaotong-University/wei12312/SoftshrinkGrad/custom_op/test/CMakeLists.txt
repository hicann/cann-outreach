cmake_minimum_required(VERSION 3.16)
project(acl_execute_softshrink_grad)

add_compile_options(-std=c++11)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "./")

set(INC_PATH "$ENV{DDK_PATH}")
if(NOT DEFINED ENV{DDK_PATH})
    set(INC_PATH "/usr/local/Ascend/ascend-toolkit/latest")
endif()

if(DEFINED ENV{CUSTOM_OPP_PATH})
    set(CUST_PKG_PATH "$ENV{CUSTOM_OPP_PATH}/op_api")
else()
    set(CUST_PKG_PATH "${INC_PATH}/opp/vendors/customize/op_api")
endif()
set(LIB_PATH "$ENV{NPU_HOST_LIB}")
if(NOT DEFINED ENV{NPU_HOST_LIB})
    string(TOLOWER "${CMAKE_SYSTEM_NAME}" SYSTEM_NAME_LOWER)
    set(LIB_PATH
        "/usr/local/Ascend/ascend-toolkit/latest/${CMAKE_SYSTEM_PROCESSOR}-${SYSTEM_NAME_LOWER}/devlib")
endif()

add_executable(execute_softshrink_grad main.cpp)
target_include_directories(execute_softshrink_grad PRIVATE
    "${INC_PATH}/include"
    "${CUST_PKG_PATH}/include"
)
target_link_directories(execute_softshrink_grad PRIVATE
    "${LIB_PATH}"
    "${CUST_PKG_PATH}/lib"
)
target_link_libraries(execute_softshrink_grad PRIVATE
    ascendcl
    cust_opapi
    acl_op_compiler
    nnopbase
    stdc++
)
