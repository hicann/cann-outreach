/*!
 * \file relu.h
 * \brief Relu 算子 kernel 类定义
 */

#ifndef RELU_H
#define RELU_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "relu_tiling_data.h"
#include "relu_tiling_key.h"

namespace NsRelu {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 1;

template <typename T, int32_t BLOCK_LENGTH>
class Relu {
public:
    __aicore__ inline Relu(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, TPipe* pipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn();
    __aicore__ inline void CopyOut();
    __aicore__ inline void Compute();

private:
    // ReLU 支持输入输出地址完全重叠。绑定 VECIN/VECOUT 后只需一块 UB，
    // 同时由队列负责 MTE2 -> V -> MTE3 的同步。
    TQueBind<QuePosition::VECIN, QuePosition::VECOUT, BUFFER_NUM> dataQueue;

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;
};

template <typename T, int32_t BLOCK_LENGTH>
__aicore__ inline void Relu<T, BLOCK_LENGTH>::Init(GM_ADDR x, GM_ADDR y, TPipe* pipe)
{
    const int32_t blockOffset = BLOCK_LENGTH * GetBlockIdx();

    inputGMX.SetGlobalBuffer((__gm__ T*)x + blockOffset, BLOCK_LENGTH);
    outputGMY.SetGlobalBuffer((__gm__ T*)y + blockOffset, BLOCK_LENGTH);

    pipe->InitBuffer(dataQueue, BUFFER_NUM, BLOCK_LENGTH * sizeof(T));
}

template <typename T, int32_t BLOCK_LENGTH>
__aicore__ inline void Relu<T, BLOCK_LENGTH>::CopyIn()
{
    LocalTensor<T> local = dataQueue.AllocTensor<T>();
    DataCopy(local, inputGMX, BLOCK_LENGTH);
    dataQueue.EnQue<QuePosition::GM, QuePosition::VECIN, T>(local);
}

template <typename T, int32_t BLOCK_LENGTH>
__aicore__ inline void Relu<T, BLOCK_LENGTH>::Compute()
{
    LocalTensor<T> local =
        dataQueue.DeQue<QuePosition::GM, QuePosition::VECIN, T>();

    // 四个特化内核的长度都恰好按 256 Bytes 对齐，直接使用固定
    // mask/repeat 接口，避免通用 calCount 路径生成额外的 Scalar 逻辑。
    constexpr uint64_t MASK = 256 / sizeof(T);
    constexpr uint8_t REPEAT_TIMES = BLOCK_LENGTH / MASK;
    AscendC::Relu<T, true>(local, local, MASK, REPEAT_TIMES, {1, 1, 8, 8});

    dataQueue.EnQue<QuePosition::VECOUT, QuePosition::GM, T>(local);
}

template <typename T, int32_t BLOCK_LENGTH>
__aicore__ inline void Relu<T, BLOCK_LENGTH>::CopyOut()
{
    LocalTensor<T> local =
        dataQueue.DeQue<QuePosition::VECOUT, QuePosition::GM, T>();
    DataCopy(outputGMY, local, BLOCK_LENGTH);
    dataQueue.FreeTensor(local);
}

template <typename T, int32_t BLOCK_LENGTH>
__aicore__ inline void Relu<T, BLOCK_LENGTH>::Process()
{
    CopyIn();
    Compute();
    CopyOut();
}

// 短向量专用静态 Tensor 内核：绕过 TPipe/TQue 的运行时资源管理。
// 本算子只有单次搬入、计算和搬出，因此只需要两组正向同步。
template <typename T, int32_t BLOCK_LENGTH>
class ReluStatic {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y)
    {
        constexpr int32_t BLOCK_BYTES = BLOCK_LENGTH * sizeof(T);
        static_assert((BLOCK_BYTES % 32) == 0, "GM copy must be 32-byte aligned");

        const int32_t blockOffset = BLOCK_LENGTH * GetBlockIdx();
        inputGMX.SetGlobalBuffer((__gm__ T*)x + blockOffset, BLOCK_LENGTH);
        outputGMY.SetGlobalBuffer((__gm__ T*)y + blockOffset, BLOCK_LENGTH);
    }

    __aicore__ inline void Process()
    {
        LocalTensor<T> local(TPosition::VECCALC, 0, BLOCK_LENGTH);

        DataCopy(local, inputGMX, BLOCK_LENGTH);
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);

        constexpr uint64_t MASK = 256 / sizeof(T);
        constexpr uint8_t REPEAT_TIMES = BLOCK_LENGTH / MASK;
        AscendC::Relu<T, true>(local, local, MASK, REPEAT_TIMES, {1, 1, 8, 8});

        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
        DataCopy(outputGMY, local, BLOCK_LENGTH);
    }

private:
    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;
};

} // namespace NsRelu
#endif // RELU_H
