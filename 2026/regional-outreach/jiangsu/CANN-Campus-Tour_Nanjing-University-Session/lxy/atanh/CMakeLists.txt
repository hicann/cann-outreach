cmake_minimum_required(VERSION 3.16)
project(atanh)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ============================================================
# Atanh 自定义算子构建（AscendC）
#
# 前提：在装有 CANN 的昇腾 Linux 环境执行
#   source /usr/local/Ascend/ascend-toolkit/set_env.sh
#
# 说明：
#   1) ops_info.so / tiling_info.so 为 host 侧产物（g++ 编译）；
#   2) kernel 侧产物使用 CANN 的 ascendc 编译器工具链，
#      推荐直接用 msopgen 生成的工程模板构建（见 README.md），
#      本文件也尝试通过 CANN 提供的 cmake 模块完成 kernel 编译。
# ============================================================

if(NOT DEFINED ENV{ASCEND_CANN_PACKAGE_PATH})
    message(FATAL_ERROR "未找到 CANN 环境，请先执行: source /usr/local/Ascend/ascend-toolkit/set_env.sh")
endif()
set(ASCEND_CANN_PACKAGE_PATH $ENV{ASCEND_CANN_PACKAGE_PATH})

add_compile_options(-fPIC -fstack-protector-all -Werror -Wno-error=deprecated-declarations)
add_compile_options(-fvisibility=hidden)

include_directories(
    ${ASCEND_CANN_PACKAGE_PATH}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/framework
    ${CMAKE_CURRENT_SOURCE_DIR}/op_host
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel
)

# ---------- 1. 算子原型库：libops_info.so ----------
add_library(ops_info SHARED framework/op_define.cc)
set_target_properties(ops_info PROPERTIES OUTPUT_NAME "ops_info")

# ---------- 2. Tiling 库：libtiling_info.so ----------
add_library(tiling_info SHARED op_host/atanh_tiling.cpp)
set_target_properties(tiling_info PROPERTIES OUTPUT_NAME "tiling_info")
target_link_libraries(tiling_info PRIVATE ascendcl)

# ---------- 3. Kernel 库：libatanh_kernel.so ----------
# 优先使用 msopgen 工程自带的 cmake/op_impl.cmake 封装（内置 ascendc 编译宏，
# 会自动从 op_info.json 的 tiling_data 字段注入 -DTILING_DATA_T=AtanhTilingData）。
# 若工程目录下没有该文件，则尝试使用 CANN 安装路径下的封装。
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/cmake/op_impl.cmake)
    include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/op_impl.cmake)
    ascendc_compile_definitions(atanh_kernel op_kernel/atanh_kernel.cpp)
elseif(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/cmake/op_impl.cmake)
    include(${ASCEND_CANN_PACKAGE_PATH}/cmake/op_impl.cmake)
    ascendc_compile_definitions(atanh_kernel op_kernel/atanh_kernel.cpp)
else()
    message(WARNING "未找到 ascendc kernel 编译工具链封装（op_impl.cmake）。\n"
                    "kernel 侧请改用 msopgen 工程构建，host 侧库已编译完成。")
endif()
