cmake_minimum_required(VERSION 3.16)
project(ResNet50Infer)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -Wall -O2")

# 设置 CANN 路径，根据实际安装位置修改
set(ASCEND_CANN_PATH "/usr/local/Ascend/ascend-toolkit/latest" CACHE PATH "CANN installation path")
if(NOT EXISTS ${ASCEND_CANN_PATH})
    set(ASCEND_CANN_PATH "/usr/local/Ascend/ascend-toolkit/latest")
endif()

include_directories(${ASCEND_CANN_PATH}/include)
link_directories(${ASCEND_CANN_PATH}/lib64 ${ASCEND_CANN_PATH}/runtime/lib64)

add_executable(resnet50_infer resnet50_infer.cpp)

target_link_libraries(resnet50_infer
    ascendcl
    pthread
    stdc++fs
)