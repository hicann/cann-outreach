/*!
 * \file add_custom.h
 * \brief add_custom 算子 Kernel 实现，实现向量逐元素相加操作 z = x + y
 *
 * 编程模型：TPipe + TQue 流水线
 *   - 2 个输入 VECIN 队列（x、y），1 个输出 VECOUT 队列（z）
 *   - BUFFER_NUM = 2 双缓冲：GM->UB 搬入、UB 内向量加法、UB->GM 搬出流水重叠
 *   - 使用 DataCopyPad 保证块首 256 字节对齐、块长 32 字节对齐
 */
#ifndef ADD_CUSTOM_H
#define ADD_CUSTOM_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "add_custom_tiling_data.h"
#include "add_custom_tiling_key.h"

namespace NsAddCustom {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2; // UB 双缓冲

template <typename T>
class AddCustom {
public:
    __aicore__ inline AddCustom(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, const AddCustomTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t progress, int64_t currentNum);
    __aicore__ inline void CopyOut(int64_t progress, int64_t currentNum);
    __aicore__ inline void Compute(int64_t currentNum);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueY;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueZ;

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> inputGMY;
    GlobalTensor<T> outputGMZ;

    int64_t blockLength_ = 0; // 当前 AI Core 负责的元素数
    int64_t ubLength_ = 0;    // 当前 AI Core 内每个 UB 块的元素数
};

template <typename T>
__aicore__ inline void AddCustom<T>::Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, const AddCustomTilingData* tilingData)
{
    // 计算当前 AI Core 的分片长度：
    // 前 (coreNum-1) 个核各处理 blockFactor 个元素，最后一个核处理剩余部分
    int64_t remainderLength = tilingData->totalNum - tilingData->blockFactor * (AscendC::GetBlockIdx() - 1);
    blockLength_ = (remainderLength > tilingData->blockFactor) ? tilingData->blockFactor : remainderLength;
    ubLength_ = tilingData->ubFactor;

    // 设置当前核在 GM 中负责的分片
    inputGMX.SetGlobalBuffer((__gm__ T*)x + tilingData->blockFactor * AscendC::GetBlockIdx(), blockLength_);
    inputGMY.SetGlobalBuffer((__gm__ T*)y + tilingData->blockFactor * AscendC::GetBlockIdx(), blockLength_);
    outputGMZ.SetGlobalBuffer((__gm__ T*)z + tilingData->blockFactor * AscendC::GetBlockIdx(), blockLength_);

    // 为输入/输出队列申请 UB 缓冲（双缓冲）
    pipe.InitBuffer(inputQueueX, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(inputQueueY, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(outputQueueZ, BUFFER_NUM, ubLength_ * sizeof(T));
}

// GM -> UB：搬入当前块的 x、y 数据并入队
template <typename T>
__aicore__ inline void AddCustom<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    AscendC::LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
    AscendC::LocalTensor<T> yLocal = inputQueueY.AllocTensor<T>();
    AscendC::DataCopyParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = currentNum * sizeof(T);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    AscendC::DataCopyPad(xLocal, inputGMX[progress * ubLength_], copyParams, {false, 0, 0, 0});
    AscendC::DataCopyPad(yLocal, inputGMY[progress * ubLength_], copyParams, {false, 0, 0, 0});
    inputQueueX.EnQue(xLocal);
    inputQueueY.EnQue(yLocal);
}

// UB -> GM：出队结果并写回 GM 当前块
template <typename T>
__aicore__ inline void AddCustom<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    AscendC::LocalTensor<T> zLocal = outputQueueZ.DeQue<T>();
    AscendC::DataCopyParams copyParams;
    copyParams.blockCount = 1;
    copyParams.blockLen = currentNum * sizeof(T);
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    AscendC::DataCopyPad(outputGMZ[progress * ubLength_], zLocal, copyParams);
    outputQueueZ.FreeTensor(zLocal);
}

// UB 内向量计算：z = x + y
template <typename T>
__aicore__ inline void AddCustom<T>::Compute(int64_t currentNum)
{
    AscendC::LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    AscendC::LocalTensor<T> yLocal = inputQueueY.DeQue<T>();
    AscendC::LocalTensor<T> zLocal = outputQueueZ.AllocTensor<T>();
    AscendC::Add(zLocal, xLocal, yLocal, currentNum);
    outputQueueZ.EnQue<T>(zLocal);
    inputQueueX.FreeTensor(xLocal);
    inputQueueY.FreeTensor(yLocal);
}

// 主流程：按 UB 块循环执行 搬入 -> 计算 -> 搬出，最后一个块处理剩余元素
template <typename T>
__aicore__ inline void AddCustom<T>::Process()
{
    int64_t loopCount = (blockLength_ + ubLength_ - 1) / ubLength_;
    for (int64_t i = 0; i < loopCount; i++) {
        int64_t currentNum = (i == (loopCount - 1)) ? (blockLength_ - ubLength_ * i) : ubLength_;
        CopyIn(i, currentNum);
        Compute(currentNum);
        CopyOut(i, currentNum);
    }
}

} // namespace NsAddCustom
#endif // ADD_CUSTOM_H
