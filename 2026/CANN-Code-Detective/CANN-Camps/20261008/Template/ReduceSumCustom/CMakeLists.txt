cmake_minimum_required(VERSION 3.16)

# 本题基于 SIMT 编程范式，仅支持 dav-3510（Ascend 950PR / Ascend 950DT）
set(CMAKE_ASC_ARCHITECTURES "dav-3510" CACHE STRING "NPU ARCH, e.g. dav-3510")
set(CMAKE_ASC_RUN_MODE "npu" CACHE STRING "运行模式: npu(默认) / sim(NPU仿真)")
set(CMAKE_ASC_ENABLE_SIMT ON)

find_package(ASC REQUIRED)
project(kernel_samples LANGUAGES ASC CXX)

add_executable(reduce_sum_custom
    reduce_sum_custom.asc
)

# 显式下发 NPU 架构与 SIMT 模式编译选项。
# 说明：部分 CANN 版本的 ASC cmake 包不会自动把架构与 SIMT 开关传给 bisheng，
# 缺失时会报 "No valid NPU architecture or soc is specified for ASC"。
target_compile_options(reduce_sum_custom PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${CMAKE_ASC_ARCHITECTURES} --enable-simt>
)
