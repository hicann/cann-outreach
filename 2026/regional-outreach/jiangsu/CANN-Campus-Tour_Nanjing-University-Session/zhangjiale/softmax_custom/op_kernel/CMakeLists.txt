# op_kernel/CMakeLists.txt
# Build the Ascend C kernel library

add_ascendc_kernel(softmax_custom_kern)
ascendc_kernel_src(softmax_custom softmax_custom.cpp)
