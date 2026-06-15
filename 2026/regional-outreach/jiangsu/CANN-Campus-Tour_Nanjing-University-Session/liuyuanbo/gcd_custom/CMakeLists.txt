# ============================================================================
# CMake 构建配置 - Gcd 算子 Kernel 直调
# ============================================================================

cmake_minimum_required(VERSION 3.16)

find_package(ASC REQUIRED)

project(gcd_custom LANGUAGES ASC CXX)

set(CMAKE_CXX_STANDARD 17)

set(ACL_INCLUDE_DIR "$ENV{ASCEND_HOME_PATH}/aarch64-linux/include")
set(ACL_LIB_DIR "$ENV{ASCEND_HOME_PATH}/lib64")

# ============================================================================
# Target: 可执行文件（直调验证）
# ============================================================================

add_executable(gcd_custom op_host/gcd_custom.asc)

target_link_libraries(gcd_custom PRIVATE
    tiling_api
    register
    platform
    unified_dlog
    dl
    m
    graph_base
)

target_include_directories(gcd_custom PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel
    ${CMAKE_CURRENT_SOURCE_DIR}/op_host
)

target_compile_options(gcd_custom PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-2201>
)
