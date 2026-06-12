# op_kernel/CMakeLists.txt
# Build the Ascend C kernel library

add_ascendc_kernel(gcd_custom_kern)
ascendc_kernel_src(gcd_custom gcd_custom.cpp)
