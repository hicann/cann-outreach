# Kernel 侧编译为算子二进制
file(GLOB_RECURSE KERNEL_SRCS *.cpp)
ascendc_compile_kernel(add_custom_kernel ${KERNEL_SRCS})
