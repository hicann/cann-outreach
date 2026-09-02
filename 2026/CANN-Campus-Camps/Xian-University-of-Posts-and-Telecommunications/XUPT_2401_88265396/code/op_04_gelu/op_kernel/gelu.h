/*!
 * \file gelu.h
 * \brief Gelu 算子 kernel 类定义
 */

#ifndef GELU_H
#define GELU_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "gelu_tiling_data.h"
#include "gelu_tiling_key.h"

namespace NsGelu {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

template <typename T>
class Gelu {
public:
    __aicore__ inline Gelu(){};

    __aicore__ inline void Init(
        GM_ADDR input_x,
        GM_ADDR output,
        const GeluTilingData* tilingData);

    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(
        int64_t progress,
        int64_t currentNum);

    __aicore__ inline void CopyOut(
        int64_t progress,
        int64_t currentNum);

    __aicore__ inline void Compute(
        int64_t currentNum);

private:
    TPipe pipe;

    TQue<
        QuePosition::VECIN,
        BUFFER_NUM>
        inputQueueX;

    TQue<
        QuePosition::VECOUT,
        BUFFER_NUM>
        outputQueueY;

    /*
     * GELU 公式计算中间 Tensor：
     *
     * tmp = x / sqrt(2)
     */
    TBuf<QuePosition::VECCALC>
        tmpBuffer;

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;

    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
};

template <typename T>
__aicore__ inline void Gelu<T>::Init(
    GM_ADDR input_x,
    GM_ADDR output,
    const GeluTilingData* tilingData)
{
    blockLength_ =
        tilingData->blockFactor;

    ubLength_ =
        tilingData->ubFactor;

    /*
     * Kernel UT / 异常 tiling 保护。
     */
    if (blockLength_ <= 0) {
        return;
    }

    if (ubLength_ <= 0 ||
        ubLength_ > blockLength_) {
        ubLength_ = blockLength_;
    }

    /*
     * 进一步限制单 Tile，
     * 给 Erf 内部计算保留足够 UB。
     */
    constexpr int64_t MAX_TILE_NUM = 1024;

    if (ubLength_ > MAX_TILE_NUM) {
        ubLength_ = MAX_TILE_NUM;
    }

    /*
     * 当前 Core 在 GM 中的起始位置。
     */
    const int64_t gmOffset =
        blockLength_ *
        static_cast<int64_t>(
            GetBlockIdx());

    inputGMX.SetGlobalBuffer(
        (__gm__ T*)input_x + gmOffset,
        blockLength_);

    outputGMY.SetGlobalBuffer(
        (__gm__ T*)output + gmOffset,
        blockLength_);

    /*
     * 输入 Double Buffer。
     */
    pipe.InitBuffer(
        inputQueueX,
        BUFFER_NUM,
        static_cast<uint32_t>(
            ubLength_ *
            static_cast<int64_t>(
                sizeof(T))));

    /*
     * 输出 Double Buffer。
     */
    pipe.InitBuffer(
        outputQueueY,
        BUFFER_NUM,
        static_cast<uint32_t>(
            ubLength_ *
            static_cast<int64_t>(
                sizeof(T))));

    /*
     * GELU 中间计算 Buffer。
     */
    pipe.InitBuffer(
        tmpBuffer,
        static_cast<uint32_t>(
            ubLength_ *
            static_cast<int64_t>(
                sizeof(T))));
}

template <typename T>
__aicore__ inline void Gelu<T>::CopyIn(
    int64_t progress,
    int64_t currentNum)
{
    LocalTensor<T> xLocal =
        inputQueueX.AllocTensor<T>();

    AscendC::DataCopy(
        xLocal,
        inputGMX[progress],
        static_cast<uint32_t>(
            currentNum));

    inputQueueX.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void Gelu<T>::Compute(
    int64_t currentNum)
{
    LocalTensor<T> xLocal =
        inputQueueX.DeQue<T>();

    LocalTensor<T> yLocal =
        outputQueueY.AllocTensor<T>();

    LocalTensor<T> tmpLocal =
        tmpBuffer.Get<T>();

    const uint32_t count =
        static_cast<uint32_t>(
            currentNum);

    /*
     * PyTorch 默认 GELU：
     *
     * y = x * 0.5 *
     *     (1 + erf(x / sqrt(2)))
     *
     * 1 / sqrt(2)
     * = 0.7071067811865475244
     */

    /*
     * tmp = x / sqrt(2)
     */
    AscendC::Muls(
        tmpLocal,
        xLocal,
        static_cast<T>(
            0.7071067811865475244f),
        count);

    /*
     * y = erf(tmp)
     *
     * 显式使用 Erf，而不是 AscendC::Gelu，
     * 以严格对应题面的 PyTorch 默认数学定义。
     */
    AscendC::Erf<T, false>(
        yLocal,
        tmpLocal,
        count);

    /*
     * y = 1 + erf(...)
     */
    AscendC::Adds(
        yLocal,
        yLocal,
        static_cast<T>(1.0f),
        count);

    /*
     * y = x * (1 + erf(...))
     */
    AscendC::Mul(
        yLocal,
        xLocal,
        yLocal,
        count);

    /*
     * y = 0.5 * x *
     *     (1 + erf(...))
     */
    AscendC::Muls(
        yLocal,
        yLocal,
        static_cast<T>(0.5f),
        count);

    outputQueueY.EnQue(yLocal);

    inputQueueX.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void Gelu<T>::CopyOut(
    int64_t progress,
    int64_t currentNum)
{
    LocalTensor<T> yLocal =
        outputQueueY.DeQue<T>();

    AscendC::DataCopy(
        outputGMY[progress],
        yLocal,
        static_cast<uint32_t>(
            currentNum));

    outputQueueY.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void Gelu<T>::Process()
{
    if (blockLength_ <= 0 ||
        ubLength_ <= 0) {
        return;
    }

    int64_t progress = 0;

    while (progress < blockLength_) {
        int64_t currentNum =
            blockLength_ - progress;

        if (currentNum > ubLength_) {
            currentNum = ubLength_;
        }

        CopyIn(
            progress,
            currentNum);

        Compute(
            currentNum);

        CopyOut(
            progress,
            currentNum);

        progress += currentNum;
    }
}

} // namespace NsGelu

#endif // GELU_H