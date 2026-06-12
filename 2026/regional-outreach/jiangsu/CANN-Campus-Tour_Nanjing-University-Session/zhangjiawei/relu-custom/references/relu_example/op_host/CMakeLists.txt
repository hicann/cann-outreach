npu_op_host(ascendc_host
    OP_TYPE Relu
    OP_FILE relu_def.cpp relu_infershape.cpp
    ARCH_OPTILING_FILES
        arch22 relu_tiling.cpp
)
npu_op_host_library(ascendc_host SRC_BASE ${CMAKE_CURRENT_SOURCE_DIR})
npu_op_package_add(${package_name} LIBRARY ascendc_host OPTILING ascendc_host)
