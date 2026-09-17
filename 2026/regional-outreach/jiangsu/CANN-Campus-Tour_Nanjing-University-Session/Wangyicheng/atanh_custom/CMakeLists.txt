# atanh_custom 算子工程 CMakeLists
#
# 说明：
#   - 推荐优先使用 msopgen 生成标准算子工程骨架，再把 op_host / op_kernel 代码替换进去。
#   - 若手动构建，需正确设置 ASCEND_CANN_PACKAGE_PATH 环境变量指向 CANN 安装目录。
#   - 不同 CANN 版本的编译工具链路径与函数名略有差异，请按实际环境微调。

cmake_minimum_required(VERSION 3.10)
project(atanh_custom)

# 芯片型号：按实际硬件替换（ascend310p / ascend910 / ascend910b / ascend910c ...）
set(SOC_VERSION "ascend910b" CACHE STRING "system on chip type")

# 调测/运行模式：cpu（CPU 侧调测）/ npu（NPU 侧运行）
set(RUN_MODE "cpu" CACHE STRING "run mode: cpu / npu")

# 依赖 CANN 环境
if(NOT DEFINED ENV{ASCEND_CANN_PACKAGE_PATH})
    message(FATAL_ERROR "请设置环境变量 ASCEND_CANN_PACKAGE_PATH 指向 CANN 安装目录")
endif()
set(ASCEND_CANN_PACKAGE_PATH $ENV{ASCEND_CANN_PACKAGE_PATH})

# 引入 CANN 提供的算子编译工具链（路径随版本变化，请确认实际存在）
include(${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_lib/cmake/AscendC.cmake)

# 编译算子动态库：同时编译 Host 侧与 Kernel 侧源码
ascendc_library(atanh_custom_${RUN_MODE} SHARED
    ${CMAKE_CURRENT_SOURCE_DIR}/op_host/atanh_custom.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/atanh_custom.cpp)

# Kernel 侧需要引用 tiling 头文件（位于 op_host 目录）
target_include_directories(atanh_custom_${RUN_MODE} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/op_host)
