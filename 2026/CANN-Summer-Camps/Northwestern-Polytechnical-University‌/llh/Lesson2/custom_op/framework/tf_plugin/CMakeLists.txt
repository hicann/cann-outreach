# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# --------------------------------------------------------------------------


aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR} plugin_srcs)
if(NOT plugin_srcs)
    return()
endif()

npu_op_library(cust_tf_parsers TF_PLUGIN
    ${plugin_srcs}
)

npu_op_package_add(${package_name}
    LIBRARY
        cust_tf_parsers
)
