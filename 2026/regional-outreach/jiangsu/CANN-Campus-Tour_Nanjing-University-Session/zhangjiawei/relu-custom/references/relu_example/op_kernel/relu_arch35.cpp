/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * ... (License header)
 */

/*!
 * \file relu_arch35.cpp
 * \brief ReLU 算子 Kernel 入口（arch35/Ascend950）
 *
 * 与 arch22 版本共用 kernel 类实现（架构无关的 elementwise 逻辑）。
 * 如果 arch35 需要不同的 TilingKey 或 Kernel 参数，在此文件中替换为 arch35 的头文件。
 */

#include "arch22/relu.h"

template <typename D_T_X, int BUFFER_MODE>
__global__ __aicore__ void relu(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(ReluTilingData);
    GET_TILING_DATA_WITH_STRUCT(ReluTilingData, tilingData, tiling);
    NsRelu::Relu<D_T_X, BUFFER_MODE> op;
    op.Init(x, y, &tilingData);
    op.Process();
}
