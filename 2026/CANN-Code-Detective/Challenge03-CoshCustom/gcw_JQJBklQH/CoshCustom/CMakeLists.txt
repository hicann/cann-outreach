cmake_minimum_required(VERSION 3.16)
find_package(ASC REQUIRED)
project(kernel_samples LANGUAGES ASC CXX)

add_executable(cosh_test
    cosh_custom.asc
)
target_compile_options(cosh_test PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-2201>
)
target_link_libraries(cosh_test m)
