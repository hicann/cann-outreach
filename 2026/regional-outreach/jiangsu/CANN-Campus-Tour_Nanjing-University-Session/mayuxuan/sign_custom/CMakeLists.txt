# ============================================================================
# CMake 构建配置 — Sign 算子 (Ascend C 直调)
# ============================================================================

cmake_minimum_required(VERSION 3.16)

find_package(ASC REQUIRED)

project(sign LANGUAGES ASC CXX)

set(CMAKE_CXX_STANDARD 17)

set(ACL_INCLUDE_DIR "$ENV{ASCEND_HOME_PATH}/aarch64-linux/include")
set(ACL_LIB_DIR "$ENV{ASCEND_HOME_PATH}/lib64")

add_executable(sign op_host/sign_host.asc)

target_link_libraries(sign PRIVATE
    tiling_api
    register
    platform
    unified_dlog
    dl
    m
    graph_base
)

target_include_directories(sign PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel
    ${CMAKE_CURRENT_SOURCE_DIR}/op_host
)

# Ascend910 → dav-2201
target_compile_options(sign PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-2201>
)

message(STATUS "Target 'sign' configured for Ascend910")
