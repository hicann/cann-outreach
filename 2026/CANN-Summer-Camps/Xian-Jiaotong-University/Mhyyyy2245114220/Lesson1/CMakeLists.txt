cmake_minimum_required(VERSION 3.10.0)
project(mul_custom LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)

# 依赖 ASCEND_HOME (例如 /usr/local/Ascend/ascend-toolkit/latest)
set(ASCEND_HOME $ENV{ASCEND_HOME})
if(NOT ASCEND_HOME)
    set(ASCEND_HOME "/usr/local/Ascend/ascend-toolkit/latest")
endif()

include_directories(${ASCEND_HOME}/include)
include_directories(${ASCEND_HOME}/include/aclnn)
link_directories(${ASCEND_HOME}/lib64)
link_directories(${ASCEND_HOME}/runtime/lib64)
link_directories(${ASCEND_HOME}/compiler/lib64)
link_directories(${ASCEND_HOME}/cxxlibs/lib64)

add_compile_options(-fPIC)
add_compile_options(-g -O0)

# 使用 asc 编译器编译 .asc 算子
add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/mul_custom.o
    COMMAND asc --cce-aicore-arch=ascend910b
                --cce-aicore-arch-vectorization
                --cce-aicore-disable-ascendc-ascend910b-aicore
                -c ${CMAKE_CURRENT_SOURCE_DIR}/mul_custom.asc
                -o ${CMAKE_CURRENT_BINARY_DIR}/mul_custom.o
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/mul_custom.asc
    COMMENT "Compile mul_custom.asc by asc"
)

add_custom_target(mul_custom_asc ALL DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/mul_custom.o)

add_executable(mul_custom ${CMAKE_CURRENT_BINARY_DIR}/mul_custom.o)
add_dependencies(mul_custom mul_custom_asc)

target_link_libraries(mul_custom
    ascendcl
    acl_op_compiler
    nnopbase
    runtime
    stdc++
    m
)
