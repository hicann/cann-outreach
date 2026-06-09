# op_host/CMakeLists.txt
# Build the host-side operator registration library

add_ascendc_ops(abs_custom_op)
ascendc_op_src(abs_custom abs_custom.cpp)
