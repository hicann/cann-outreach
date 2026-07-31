/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file truncate_mod.h
 * \brief TruncateMod 算子 kernel 类定义
 */
#ifndef TRUNCATE_MOD_H
#define TRUNCATE_MOD_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "truncate_mod_tiling_data.h"
#include "truncate_mod_tiling_key.h"
#include <limits>
#include <type_traits>

namespace NsTruncateMod {
using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

template <typename T>
class TruncateMod {
public:
    __aicore__ inline TruncateMod() {}

    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, const TruncateModTilingData* tilingData)
    {
        uint32_t blockIdx = GetBlockIdx();
        uint32_t blockOffset = blockIdx * tilingData->smallCoreDataNum;

        this->x1Gm.SetGlobalBuffer((__gm__ T*)x1);
        this->x2Gm.SetGlobalBuffer((__gm__ T*)x2);
        this->yGm.SetGlobalBuffer((__gm__ T*)y + blockOffset);

        this->tileDataNum = tilingData->tmpTileDataNum / 2;
        this->totalDataNum = tilingData->totalDataNum;
        this->x1DataNum = tilingData->x1DataNum;
        this->x2DataNum = tilingData->x2DataNum;
        this->globalOffset = blockOffset;

        uint32_t remainDataNum = this->totalDataNum - blockOffset;
        this->processDataNum = remainDataNum < tilingData->smallCoreDataNum ? remainDataNum :
                                                                              tilingData->smallCoreDataNum;
        this->tileNum = (this->processDataNum + this->tileDataNum - 1U) / this->tileDataNum;
        this->tailDataNum = this->processDataNum - (this->tileNum - 1U) * this->tileDataNum;

        for (uint32_t i = 0; i < TRUNCATE_MOD_MAX_DIMS; ++i) {
            this->outShape[i] = tilingData->outShape[i];
            this->x1Shape[i] = tilingData->x1Shape[i];
            this->x2Shape[i] = tilingData->x2Shape[i];
        }

        uint32_t x1Stride = 1U;
        uint32_t x2Stride = 1U;
        for (int32_t i = static_cast<int32_t>(TRUNCATE_MOD_MAX_DIMS) - 1; i >= 0; --i) {
            this->x1Stride[i] = x1Stride;
            this->x2Stride[i] = x2Stride;
            x1Stride *= this->x1Shape[i];
            x2Stride *= this->x2Shape[i];
        }

        pipe.InitBuffer(x1Queue, BUFFER_NUM, this->tileDataNum * sizeof(T));
        pipe.InitBuffer(x2Queue, BUFFER_NUM, this->tileDataNum * sizeof(T));
        pipe.InitBuffer(yQueue, BUFFER_NUM, this->tileDataNum * sizeof(T));

        // FLOAT32 can execute Fmod directly and INT32 uses exact scalar modulo;
        // neither needs conversion buffers.  Avoiding these buffers also makes
        // the 4096-element fast-path tile possible without UB pressure.
        if constexpr (!std::is_same<T, float>::value && !std::is_same<T, int32_t>::value) {
            pipe.InitBuffer(x1FloatBuf, this->tileDataNum * sizeof(float));
            pipe.InitBuffer(x2FloatBuf, this->tileDataNum * sizeof(float));
            pipe.InitBuffer(divFloatBuf, this->tileDataNum * sizeof(float));
            if constexpr (std::is_same<T, int8_t>::value || std::is_same<T, uint8_t>::value) {
                pipe.InitBuffer(tmpHalfBuf, this->tileDataNum * sizeof(half));
            }
        }
    }

    __aicore__ inline void Process()
    {
        for (uint32_t i = 0; i < this->tileNum; ++i) {
            uint32_t curDataNum = this->tileDataNum;
            if (i == this->tileNum - 1) {
                curDataNum = this->tailDataNum;
            }
            CopyIn(i, curDataNum);
            Compute(curDataNum);
            CopyOut(i, curDataNum);
        }
    }

private:
    __aicore__ inline void CopyBroadcast(LocalTensor<T>& dst, GlobalTensor<T>& src, uint32_t outIndex,
                                         uint32_t curDataNum, const uint32_t* inputShape, const uint32_t* inputStride)
    {
        uint32_t inputIndex = 0U;
        uint32_t coords[TRUNCATE_MOD_MAX_DIMS];
        uint32_t remain = outIndex;

        for (int32_t i = static_cast<int32_t>(TRUNCATE_MOD_MAX_DIMS) - 1; i >= 0; --i) {
            coords[i] = remain % this->outShape[i];
            remain /= this->outShape[i];
            if (inputShape[i] != 1U) {
                inputIndex += coords[i] * inputStride[i];
            }
        }

        for (uint32_t n = 0; n < curDataNum; ++n) {
            dst.SetValue(n, src.GetValue(inputIndex));
            if (n + 1U == curDataNum) {
                break;
            }

            // Advance the output coordinate like an odometer.  This removes
            // eight divisions/modulos per element from the old gather path.
            for (int32_t i = static_cast<int32_t>(TRUNCATE_MOD_MAX_DIMS) - 1; i >= 0; --i) {
                ++coords[i];
                if (coords[i] < this->outShape[i]) {
                    if (inputShape[i] != 1U) {
                        inputIndex += inputStride[i];
                    }
                    break;
                }

                coords[i] = 0U;
                if (inputShape[i] != 1U) {
                    inputIndex -= (this->outShape[i] - 1U) * inputStride[i];
                }
            }
        }
    }

    __aicore__ inline void CopyIn(uint32_t progress, uint32_t curDataNum)
    {
        LocalTensor<T> x1Local = x1Queue.AllocTensor<T>();
        LocalTensor<T> x2Local = x2Queue.AllocTensor<T>();
        uint32_t tileOffset = progress * this->tileDataNum;

        if (this->x1DataNum == this->totalDataNum) {
            DataCopyExtParams copyParams{1U, static_cast<uint32_t>(curDataNum * sizeof(T)), 0U, 0U, 0U};
            DataCopyPadExtParams<T> padParams{false, 0U, 0U, 0};
            DataCopyPad(x1Local, x1Gm[this->globalOffset + tileOffset], copyParams, padParams);
        } else if (this->x1DataNum == 1U) {
            T scalarValue = x1Gm.GetValue(0);
            if constexpr (std::is_same<T, int8_t>::value || std::is_same<T, uint8_t>::value) {
                for (uint32_t i = 0; i < curDataNum; ++i) {
                    x1Local.SetValue(i, scalarValue);
                }
            } else {
                Duplicate(x1Local, scalarValue, curDataNum);
            }
        } else {
            CopyBroadcast(x1Local, x1Gm, this->globalOffset + tileOffset, curDataNum, this->x1Shape, this->x1Stride);
        }

        if (this->x2DataNum == this->totalDataNum) {
            DataCopyExtParams copyParams{1U, static_cast<uint32_t>(curDataNum * sizeof(T)), 0U, 0U, 0U};
            DataCopyPadExtParams<T> padParams{false, 0U, 0U, 0};
            DataCopyPad(x2Local, x2Gm[this->globalOffset + tileOffset], copyParams, padParams);
        } else if (this->x2DataNum == 1U) {
            T scalarValue = x2Gm.GetValue(0);
            if constexpr (std::is_same<T, int8_t>::value || std::is_same<T, uint8_t>::value) {
                for (uint32_t i = 0; i < curDataNum; ++i) {
                    x2Local.SetValue(i, scalarValue);
                }
            } else {
                Duplicate(x2Local, scalarValue, curDataNum);
            }
        } else {
            CopyBroadcast(x2Local, x2Gm, this->globalOffset + tileOffset, curDataNum, this->x2Shape, this->x2Stride);
        }

        x1Queue.EnQue(x1Local);
        x2Queue.EnQue(x2Local);
    }

    __aicore__ inline void CastInputToFloat(LocalTensor<float>& dst, LocalTensor<T>& src, uint32_t curDataNum)
    {
        if constexpr (std::is_same<T, float>::value) {
            Adds(dst, src, 0.0F, curDataNum);
        } else if constexpr (std::is_same<T, int8_t>::value || std::is_same<T, uint8_t>::value) {
            LocalTensor<half> tmpHalf = tmpHalfBuf.Get<half>();
            Cast(tmpHalf, src, RoundMode::CAST_NONE, curDataNum);
            Cast(dst, tmpHalf, RoundMode::CAST_NONE, curDataNum);
        } else {
            Cast(dst, src, RoundMode::CAST_NONE, curDataNum);
        }
    }

    __aicore__ inline void CastFloatToOutput(LocalTensor<T>& dst, LocalTensor<float>& src, uint32_t curDataNum)
    {
        if constexpr (std::is_same<T, float>::value) {
            Adds(dst, src, 0.0F, curDataNum);
        } else if constexpr (std::is_same<T, int8_t>::value || std::is_same<T, uint8_t>::value) {
            LocalTensor<half> tmpHalf = tmpHalfBuf.Get<half>();
            Cast(tmpHalf, src, RoundMode::CAST_TRUNC, curDataNum);
            Cast(dst, tmpHalf, RoundMode::CAST_TRUNC, curDataNum);
        } else if constexpr (std::is_same<T, int32_t>::value) {
            Cast(dst, src, RoundMode::CAST_TRUNC, curDataNum);
        } else if constexpr (std::is_same<T, bfloat16_t>::value) {
            Cast(dst, src, RoundMode::CAST_RINT, curDataNum);
        } else {
            Cast(dst, src, RoundMode::CAST_NONE, curDataNum);
        }
    }

    __aicore__ inline void Compute(uint32_t curDataNum)
    {
        LocalTensor<T> x1Local = x1Queue.DeQue<T>();
        LocalTensor<T> x2Local = x2Queue.DeQue<T>();
        LocalTensor<T> yLocal = yQueue.AllocTensor<T>();

        if constexpr (std::is_same<T, float>::value) {
            // Direct path: the previous implementation copied both inputs to
            // temporary float buffers and copied the result back even though
            // the queue tensors already have the required dtype.
            Fmod<float, false>(yLocal, x1Local, x2Local, curDataNum);
        } else if constexpr (std::is_same<T, int32_t>::value) {
            for (uint32_t i = 0; i < curDataNum; ++i) {
                T dividend = x1Local.GetValue(i);
                T divisor = x2Local.GetValue(i);
                T result = static_cast<T>(0);
                if (divisor != static_cast<T>(0)) {
                    const bool overflowCase = dividend == std::numeric_limits<int32_t>::min() &&
                                              divisor == static_cast<T>(-1);
                    if (!overflowCase) {
                        result = static_cast<T>(dividend % divisor);
                    }
                }
                yLocal.SetValue(i, result);
            }
        } else {
            LocalTensor<float> x1Float = x1FloatBuf.Get<float>();
            LocalTensor<float> x2Float = x2FloatBuf.Get<float>();
            LocalTensor<float> divFloat = divFloatBuf.Get<float>();

            CastInputToFloat(x1Float, x1Local, curDataNum);
            CastInputToFloat(x2Float, x2Local, curDataNum);

            if constexpr (std::is_same<T, int8_t>::value || std::is_same<T, uint8_t>::value) {
                for (uint32_t i = 0; i < curDataNum; ++i) {
                    if (x2Local.GetValue(i) == static_cast<T>(0)) {
                        x2Float.SetValue(i, 1.0F);
                    }
                }
            }

            Fmod<float, false>(divFloat, x1Float, x2Float, curDataNum);

            CastFloatToOutput(yLocal, divFloat, curDataNum);

            if constexpr (std::is_same<T, int8_t>::value || std::is_same<T, uint8_t>::value) {
                for (uint32_t i = 0; i < curDataNum; ++i) {
                    if (x2Local.GetValue(i) == static_cast<T>(0)) {
                        yLocal.SetValue(i, static_cast<T>(0));
                    }
                }
            }
        }

        yQueue.EnQue(yLocal);
        x1Queue.FreeTensor(x1Local);
        x2Queue.FreeTensor(x2Local);
    }

    __aicore__ inline void CopyOut(uint32_t progress, uint32_t curDataNum)
    {
        LocalTensor<T> yLocal = yQueue.DeQue<T>();
        DataCopyExtParams copyParams{1U, static_cast<uint32_t>(curDataNum * sizeof(T)), 0U, 0U, 0U};
        DataCopyPad(yGm[progress * this->tileDataNum], yLocal, copyParams);
        yQueue.FreeTensor(yLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> x1Queue;
    TQue<QuePosition::VECIN, BUFFER_NUM> x2Queue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> yQueue;

    TBuf<QuePosition::VECCALC> x1FloatBuf;
    TBuf<QuePosition::VECCALC> x2FloatBuf;
    TBuf<QuePosition::VECCALC> divFloatBuf;
    TBuf<QuePosition::VECCALC> tmpHalfBuf;

    GlobalTensor<T> x1Gm;
    GlobalTensor<T> x2Gm;
    GlobalTensor<T> yGm;

    uint32_t processDataNum = 0;
    uint32_t tileDataNum = 0;
    uint32_t tileNum = 0;
    uint32_t tailDataNum = 0;
    uint32_t totalDataNum = 0;
    uint32_t x1DataNum = 0;
    uint32_t x2DataNum = 0;
    uint32_t globalOffset = 0;
    uint32_t outShape[TRUNCATE_MOD_MAX_DIMS];
    uint32_t x1Shape[TRUNCATE_MOD_MAX_DIMS];
    uint32_t x2Shape[TRUNCATE_MOD_MAX_DIMS];
    uint32_t x1Stride[TRUNCATE_MOD_MAX_DIMS];
    uint32_t x2Stride[TRUNCATE_MOD_MAX_DIMS];
};

} // namespace NsTruncateMod

#endif // TRUNCATE_MOD_H
