# op_host/CMakeLists.txt
# Build the host-side operator registration library

add_ascendc_ops(gcd_custom_op)
ascendc_op_src(gcd_custom gcd_custom.cpp)
