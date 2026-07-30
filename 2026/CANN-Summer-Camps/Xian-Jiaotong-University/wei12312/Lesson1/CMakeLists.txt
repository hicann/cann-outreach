cmake_minimum_required(VERSION 3.16)
find_package(ASC REQUIRED)
project(kernel_samples LANGUAGES ASC CXX)

set(ASC_NPU_ARCH "dav-2201" CACHE STRING "Target Ascend NPU architecture")

add_executable(mul_test
    mul_custom.asc
)
target_compile_options(mul_test PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${ASC_NPU_ARCH}>
)
