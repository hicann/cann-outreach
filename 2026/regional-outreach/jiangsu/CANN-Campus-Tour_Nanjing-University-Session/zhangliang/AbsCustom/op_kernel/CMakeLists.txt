# op_kernel/CMakeLists.txt
# Build the Ascend C kernel library

add_ascendc_kernel(abs_custom_kern)
ascendc_kernel_src(abs_custom abs_custom.cpp)
