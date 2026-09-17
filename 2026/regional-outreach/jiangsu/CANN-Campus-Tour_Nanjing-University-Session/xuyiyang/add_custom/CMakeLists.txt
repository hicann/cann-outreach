# add_custom 算子构建脚本 (CANN)
# 用法: 在 CANN 环境下 (source set_env.sh) 执行
#   bash build.sh   或   cmake -B build && cmake --build build

cmake_minimum_required(VERSION 3.16)
project(add_custom C CXX)

set(CMAKE_CXX_STANDARD 14)

# CANN 路径 (按实际环境调整, 或由环境变量 ASCEND_HOME 提供)
if(NOT DEFINED ASCEND_HOME)
    set(ASCEND_HOME /usr/local/Ascend)
endif()
set(CANN_ROOT ${ASCEND_HOME}/ascend-toolkit/latest)
set(AICPU_SOC_VERSION "Ascend910B1" CACHE STRING "SoC 版本, 如 Ascend910B1 / Ascend310B1")

# ---------- device 侧: 使用 CANN 的自定义算子编译目标 ----------
# 将 add_custom.cpp 编译为 .o, 由 op_host / op_kernel 目录结构打包为自定义算子包
add_library(add_custom_kernel OBJECT add_custom.cpp)
target_include_directories(add_custom_kernel PRIVATE
    ${CANN_ROOT}/runtime/include
    ${CANN_ROOT}/include
)

# ---------- host 侧 ----------
add_library(add_custom_host STATIC add_custom_host.cpp)
target_include_directories(add_custom_host PRIVATE
    ${CANN_ROOT}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
)
