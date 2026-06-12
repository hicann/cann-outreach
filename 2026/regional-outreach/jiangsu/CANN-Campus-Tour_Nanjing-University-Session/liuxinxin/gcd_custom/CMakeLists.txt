# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

# ============================================================================
# CMake 构建配置 - Gcd 算子（FP16 Kernel 直调）
# ============================================================================

cmake_minimum_required(VERSION 3.16)
find_package(ASC REQUIRED)
project(gcd_custom LANGUAGES ASC CXX)
set(CMAKE_CXX_STANDARD 17)

set(ACL_INCLUDE_DIR "$ENV{ASCEND_HOME_PATH}/aarch64-linux/include")
set(ACL_LIB_DIR "$ENV{ASCEND_HOME_PATH}/lib64")

# Target 1: 可执行文件
add_executable(gcd_custom op_host/gcd_custom.asc)
target_link_libraries(gcd_custom PRIVATE tiling_api register platform unified_dlog dl m graph_base)
target_include_directories(gcd_custom PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel ${CMAKE_CURRENT_SOURCE_DIR}/op_host)
target_compile_options(gcd_custom PRIVATE $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-2201>)

# Target 2: TORCH_LIBRARY 模块
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE TORCH_CMAKE_PREFIX_PATH)
find_package(Torch REQUIRED HINTS ${TORCH_CMAKE_PREFIX_PATH})
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import os, torch_npu; print(os.path.dirname(torch_npu.__file__))"
    OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE TORCH_NPU_PATH)
set(TORCH_NPU_INCLUDE_DIRS ${TORCH_NPU_PATH}/include)
set(TORCH_NPU_LIBRARIES ${TORCH_NPU_PATH}/lib)

add_library(gcd_custom_ops SHARED
    op_kernel/gcd_custom_kernel.asc
    op_extension/gcd_custom_torch.cpp
    op_extension/register.cpp)
target_include_directories(gcd_custom_ops PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel ${CMAKE_CURRENT_SOURCE_DIR}/op_extension
    ${TORCH_INCLUDE_DIRS} ${TORCH_NPU_INCLUDE_DIRS} ${ACL_INCLUDE_DIR})
target_link_libraries(gcd_custom_ops PRIVATE torch_npu tiling_api register platform unified_dlog dl m graph_base ascendcl ascendc_runtime)
target_link_directories(gcd_custom_ops PRIVATE ${ACL_LIB_DIR} "$ENV{ASCEND_HOME_PATH}/aarch64-linux/lib64" ${TORCH_NPU_LIBRARIES})
target_compile_options(gcd_custom_ops PRIVATE ${TORCH_CXX_FLAGS} $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-2201>)
message(STATUS "TORCH_LIBRARY module 'libgcd_custom_ops.so' will be built")
