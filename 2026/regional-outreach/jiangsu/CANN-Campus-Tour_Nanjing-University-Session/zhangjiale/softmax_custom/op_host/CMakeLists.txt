# op_host/CMakeLists.txt
# Build the host-side operator registration library

add_ascendc_ops(softmax_custom_op)
ascendc_op_src(softmax_custom softmax_custom.cpp)
