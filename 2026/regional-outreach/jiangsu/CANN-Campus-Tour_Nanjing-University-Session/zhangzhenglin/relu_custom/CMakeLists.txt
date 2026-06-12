# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.16)

find_package(ASC REQUIRED)

project(relu_custom LANGUAGES ASC CXX)

# ============================================================
# Target 1: 可执行文件（直接验证）
# ============================================================
add_executable(relu_custom
    host/relu_custom.asc
    kernel/relu_custom_kernel.asc
)

target_include_directories(relu_custom PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_compile_options(relu_custom PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-3510>
)

target_link_libraries(relu_custom PRIVATE
    ascendcl
    tiling_api
    register
    platform
    m
    dl
)

# ============================================================
# Target 2: PyTorch .so（算子注册库）
# ============================================================
add_library(relu_custom_ops SHARED
    torch_library/relu_custom_torch.cpp
    torch_library/register.cpp
    kernel/relu_custom_kernel.asc
)

target_include_directories(relu_custom_ops PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${TORCH_INCLUDE_DIRS}
)

target_compile_options(relu_custom_ops PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-3510>
)

target_link_libraries(relu_custom_ops PRIVATE
    ascendcl
    tiling_api
    register
    platform
    m
    dl
    ${TORCH_LIBRARIES}
)
