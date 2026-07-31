/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __TRUNCATE_MOD_OPTIMIZED_H__
#define __TRUNCATE_MOD_OPTIMIZED_H__

#include "kernel_operator.h"
#include "atvoss/broadcast/broadcast_sch.h"

using namespace AscendC;

// UB空间规划: inQueueX(2份) + inQueueX2(2份) + outQueueY(2份) + workBuf(1份)
// tileElements = (UB_SIZE - overhead) / (sizeof(T) * 7)

// Double Buffer优化: CopyIn -> Compute -> CopyOut 三级流水线并行
template <typename T>
__aicore__ void truncate_mod_optimized(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, 
                                       GM_ADDR workspace, GM_ADDR tiling)
{
    // 获取tiling信息
    int32_t blockIdx = GetBlockIdx();
    int32_t totalLength = GetBlockNum();
    
    // 双缓冲队列初始化
    TQue<VECIN, 2> inQueueX;
    TQue<VECIN, 2> inQueueX2;
    TQue<VECOUT, 2> outQueueY;
    
    // 计算每个core处理的数据量
    int32_t tileElements = (TOTAL_UB_SIZE - OVERHEAD) / (sizeof(T) * 7);
    int32_t offset = blockIdx * tileElements;
    int32_t curTileSize = tileElements;
    
    // 确保128B对齐
    int32_t alignSize = 128 / sizeof(T);
    curTileSize = (curTileSize / alignSize) * alignSize;
    
    // 初始化缓冲区
    LocalTensor<T> x1Local = inQueueX.AllocTensor<T>();
    LocalTensor<T> x2Local = inQueueX2.AllocTensor<T>();
    LocalTensor<T> yLocal = outQueueY.AllocTensor<T>();
    
    // 搬运数据
    DataCopy(x1Local, x1 + offset, curTileSize);
    DataCopy(x2Local, x2 + offset, curTileSize);
    
    // 计算
    // truncate_mod: out = x1 - trunc(x1 / x2) * x2
    LocalTensor<T> temp1 = workspace_local;
    LocalTensor<T> temp2 = temp1 + curTileSize;
    
    // 使用Vector计算
    Div(temp1, x1Local, x2Local, curTileSize);
    Trunc(temp2, temp1, curTileSize);
    Mul(temp1, temp2, x2Local, curTileSize);
    Sub(yLocal, x1Local, temp1, curTileSize);
    
    // 搬出结果
    outQueueY.EnQue(yLocal);
    DataCopy(y + offset, yLocal, curTileSize);
}

// 标量优化路径: other为标量时使用Muls(1/otherScalar)优化除法
template <typename T>
__aicore__ void truncate_mod_scalar_optimized(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, 
                                              GM_ADDR workspace, GM_ADDR tiling)
{
    // 获取标量other的值
    T otherScalar = x2[0];
    T invOther = static_cast<T>(1.0) / otherScalar;
    
    int32_t blockIdx = GetBlockIdx();
    int32_t totalLength = GetBlockNum();
    
    // 双缓冲队列初始化
    TQue<VECIN, 2> inQueueX;
    TQue<VECOUT, 2> outQueueY;
    
    // 计算每个core处理的数据量
    int32_t tileElements = (TOTAL_UB_SIZE - OVERHEAD) / (sizeof(T) * 5);
    int32_t offset = blockIdx * tileElements;
    int32_t curTileSize = tileElements;
    
    // 确保128B对齐
    int32_t alignSize = 128 / sizeof(T);
    curTileSize = (curTileSize / alignSize) * alignSize;
    
    // 初始化缓冲区
    LocalTensor<T> x1Local = inQueueX.AllocTensor<T>();
    LocalTensor<T> yLocal = outQueueY.AllocTensor<T>();
    
    // 搬运数据
    DataCopy(x1Local, x1 + offset, curTileSize);
    
    // 使用Muls优化除法: x1 * (1/other)
    LocalTensor<T> temp1 = workspace_local;
    Muls(temp1, x1Local, invOther, curTileSize);
    
    // 计算truncate
    LocalTensor<T> temp2 = temp1 + curTileSize;
    Trunc(temp2, temp1, curTileSize);
    
    // 计算temp2 * other
    Muls(temp1, temp2, otherScalar, curTileSize);
    
    // 计算最终结果
    Sub(yLocal, x1Local, temp1, curTileSize);
    
    // 搬出结果
    outQueueY.EnQue(yLocal);
    DataCopy(y + offset, yLocal, curTileSize);
}

#endif // __TRUNCATE_MOD_OPTIMIZED_H__