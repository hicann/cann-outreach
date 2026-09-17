# add_custom 算子构建脚本（CPU 调试模式）
cmake_minimum_required(VERSION 3.16)
project(add_custom C CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Ascend CANN 包路径（按实际安装路径修改，或通过 -DASCEND_CANN_PACKAGE_PATH 指定）
if(NOT DEFINED ASCEND_CANN_PACKAGE_PATH)
    set(ASCEND_CANN_PACKAGE_PATH "/usr/local/Ascend/ascend-toolkit/latest")
endif()

# CPU 调试模式需要设置环境变量中的编译器/头文件路径
include_directories(
    ${ASCEND_CANN_PACKAGE_PATH}/include
    ${ASCEND_CANN_PACKAGE_PATH}/include/profiler
    ${CMAKE_CURRENT_SOURCE_DIR}
)

link_directories(
    ${ASCEND_CANN_PACKAGE_PATH}/lib64
    ${ASCEND_CANN_PACKAGE_PATH}/runtime/lib64/stub
)

add_executable(add_custom
    add_custom.cpp
    main.cpp
)

target_compile_definitions(add_custom PRIVATE
    __CCE_KT_TEST__
    __CCE_AICORE__=220
)

target_link_libraries(add_custom PRIVATE
    ascendcl
    tilde
    alog
    c_sec
    runtime
    profapi
    sodium
)

# 输出目录
set_target_properties(add_custom PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/build
)
