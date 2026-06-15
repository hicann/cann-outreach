# ============================================================================
# CMake 构建配置 — InplaceRsqrt (Ascend C Kernel 直调)
# ============================================================================

cmake_minimum_required(VERSION 3.16)

find_package(ASC REQUIRED)

project(inplace_rsqrt LANGUAGES ASC CXX)

set(CMAKE_CXX_STANDARD 17)

set(ACL_INCLUDE_DIR "$ENV{ASCEND_HOME_PATH}/aarch64-linux/include")
set(ACL_LIB_DIR "$ENV{ASCEND_HOME_PATH}/lib64")

# ============================================================================
# Target: 可执行文件（直调验证）
# ============================================================================

add_executable(inplace_rsqrt op_host/inplace_rsqrt_host.asc)

target_link_libraries(inplace_rsqrt PRIVATE
    tiling_api
    register
    platform
    unified_dlog
    dl
    m
    graph_base
)

target_include_directories(inplace_rsqrt PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel
    ${CMAKE_CURRENT_SOURCE_DIR}/op_host
)

# [CONFIG] 根据目标 NPU 架构调整
# dav-2201: Atlas A2 系列 | dav-3510: Atlas A5 系列
target_compile_options(inplace_rsqrt PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-2201>
)

message(STATUS "Target 'inplace_rsqrt' configured")
