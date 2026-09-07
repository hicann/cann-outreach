/*!
 * \file square.h
 * \brief Square 算子 kernel 类定义
 *
 * 实现：y = x·x，元素级平方。
 * - 数据分片：按 32B 对齐单位(alignElem) 均分到各核；每块内按 ubLength_ 分块循环。
 * - 同步：DataCopy(MTE2)/向量计算(V)/DataCopy(MTE3) 之间用 PipeBarrier<PIPE_ALL>
 *   显式同步（910B 异步管线必需；已在 GELU 赛题真机验证过的模式）。
 * - 非 32B 对齐尾部（N 非 32 字节整倍数）：用 DataCopyPad 按字节精确搬运，
 *   GM->UB 补零，对整块(含补零区)算平方，UB->GM 只写有效字节，不越界写。
 */

#ifndef SQUARE_H
#define SQUARE_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "square_tiling_data.h"
#include "square_tiling_key.h"

namespace NsSquare {

using namespace AscendC;

template <typename T>
class Square {
public:
    __aicore__ inline Square(){};

    __aicore__ inline void Init(GM_ADDR input_x, GM_ADDR output, const SquareTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void ProcessChunk(int64_t offElems, int64_t n);
    __aicore__ inline void ProcessTail(int64_t offElems, uint32_t tailBytes, uint16_t rightPad);

private:
    TPipe pipe;
    TBuf<TPosition::VECIN> bufX;
    TBuf<TPosition::VECIN> bufY;

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;

    int64_t totalNum_ = 0;
    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
    int64_t alignElem_ = 8;   // 32B 对齐元素数：fp32 8 / fp16 16
};

template <typename T>
__aicore__ inline void Square<T>::Init(GM_ADDR input_x, GM_ADDR output, const SquareTilingData* tilingData)
{
    totalNum_ = tilingData->totalNum;
    blockLength_ = tilingData->blockFactor;
    ubLength_ = tilingData->ubFactor;
    alignElem_ = 32 / sizeof(T);

    int64_t blockIdx = AscendC::GetBlockIdx();
    int64_t blockStart = blockIdx * blockLength_;
    inputGMX.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(input_x) + blockStart, blockLength_);
    outputGMY.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(output) + blockStart, blockLength_);

    pipe.InitBuffer(bufX, ubLength_ * sizeof(T));
    pipe.InitBuffer(bufY, ubLength_ * sizeof(T));
}

template <typename T>
__aicore__ inline void Square<T>::ProcessChunk(int64_t offElems, int64_t n)
{
    LocalTensor<T> xLocal = bufX.Get<T>();
    LocalTensor<T> yLocal = bufY.Get<T>();

    DataCopy(xLocal, inputGMX[offElems], n);
    PipeBarrier<PIPE_ALL>();           // MTE2 搬入完成后再计算
    Mul(yLocal, xLocal, xLocal, n);    // y = x·x
    PipeBarrier<PIPE_ALL>();           // 计算完成后再搬出
    DataCopy(outputGMY[offElems], yLocal, n);
    PipeBarrier<PIPE_ALL>();           // MTE3 完成，防止下一块 MTE2 重写缓冲竞争
}

template <typename T>
__aicore__ inline void Square<T>::ProcessTail(int64_t offElems, uint32_t tailBytes, uint16_t rightPad)
{
    LocalTensor<T> xLocal = bufX.Get<T>();
    LocalTensor<T> yLocal = bufY.Get<T>();

    // GM -> UB：只搬 tailBytes 有效字节，右侧补零到 32B 对齐宽度
    DataCopyExtParams copyParams{1, tailBytes, 0, 0, 0};
    DataCopyPadExtParams<T> padParams{true, 0, static_cast<uint8_t>(rightPad), static_cast<T>(0)};
    DataCopyPad(xLocal, inputGMX[offElems], copyParams, padParams);
    PipeBarrier<PIPE_ALL>();
    Mul(yLocal, xLocal, xLocal, static_cast<int64_t>(alignElem_));   // 整块(含补零)算平方
    PipeBarrier<PIPE_ALL>();
    DataCopyPad(outputGMY[offElems], yLocal, copyParams);            // 只写有效字节
    PipeBarrier<PIPE_ALL>();
}

template <typename T>
__aicore__ inline void Square<T>::Process()
{
    if (blockLength_ <= 0 || ubLength_ <= 0) {
        return;
    }
    // 本核数据片 [blockStart, blockStart+blockLength_)；alignedTotal 之前全是 32B 对齐
    // 常规块，[alignedTotal, totalNum) 的不足 32B 尾部由包含该边界的核用 pad 拷贝处理。
    int64_t blockIdx = AscendC::GetBlockIdx();
    int64_t blockStart = blockIdx * blockLength_;
    int64_t blockEnd = blockStart + blockLength_;
    int64_t alignedTotal = (totalNum_ / alignElem_) * alignElem_;

    int64_t aEnd = (blockEnd < alignedTotal) ? blockEnd : alignedTotal;
    int64_t off = blockStart;
    for (; off < aEnd; off += ubLength_) {
        int64_t n = (aEnd - off < ubLength_) ? (aEnd - off) : ubLength_;
        ProcessChunk(off - blockStart, n);
    }

    if (alignedTotal < totalNum_ && blockStart <= alignedTotal && alignedTotal < blockEnd) {
        int64_t tailElems = totalNum_ - alignedTotal;
        uint32_t tailBytes = static_cast<uint32_t>(tailElems * sizeof(T));
        uint16_t rightPad = static_cast<uint16_t>(alignElem_ - tailElems);
        ProcessTail(alignedTotal - blockStart, tailBytes, rightPad);
    }
}

} // namespace NsSquare
#endif // SQUARE_H
