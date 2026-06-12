# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

add_op_host_library(relu_example_host OBJECT
    relu_example_def.cpp
    relu_example_infershape.cpp
)

if(ASCEND_COMPUTE_UNIT MATCHES "ascend910b|ascend910_93")
    add_op_host_library(relu_example_arch22 OBJECT
        arch22/relu_example_tiling.cpp
    )
endif()

if(ASCEND_COMPUTE_UNIT MATCHES "ascend950")
    add_op_host_library(relu_example_arch35 OBJECT
        arch35/relu_example_tiling.cpp
    )
endif()
