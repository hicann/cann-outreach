/*!
 * \file relu.h
 * \brief Relu 算子 kernel 类定义（性能优化：单块化处理）
 */

#ifndef RELU_H
#define RELU_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "relu_tiling_data.h"
#include "relu_tiling_key.h"

namespace NsRelu {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;          // 双缓冲
constexpr int64_t TILE_LEN = 128;          // 分块回退时的块长（half/float 均 32 字节对齐）
constexpr int64_t MAX_UB_BYTES = 196608;   // 单核 UB 预算上限（4 缓冲合计 192KB，UB 为 256KB）

template <typename T>
class Relu {
public:
    __aicore__ inline Relu(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const ReluTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t progress, int64_t currentNum);
    __aicore__ inline void CopyOut(int64_t progress, int64_t currentNum);
    __aicore__ inline void Compute(int64_t currentNum);

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueY;

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;

    int64_t blockLength_ = 0; // 本核实际长度
    int64_t alignedLen_ = 0;  // blockLength 对齐到 TILE_LEN 的部分
    int64_t ubLength_ = 0;    // 每次处理的块长（能单块则整块，否则 TILE_LEN）
    int64_t startIdx_ = 0;
};

// Init：每核基础对齐切分（前 N-1 核 alignedBlock，最后一个核吃全部余量）
//   单块化：本核对齐部分若 UB 放得下，则整块一次搬入/搬出（大幅减少循环与同步开销）；
//   放不下则退回 TILE_LEN 分块，保证任意大 shape 正确。
template <typename T>
__aicore__ inline void Relu<T>::Init(GM_ADDR x, GM_ADDR y, const ReluTilingData* tilingData)
{
    int64_t coreNum = AscendC::GetBlockNum();
    int64_t coreId = AscendC::GetBlockIdx();
    int64_t totalLength = tilingData->totalLength;
    int64_t base = totalLength / coreNum;                // 每核基础长度
    int64_t alignedBlock = (base / TILE_LEN) * TILE_LEN; // 对齐到 TILE_LEN 的基础长度
    int64_t startIdx = alignedBlock * coreId;            // 本核起始下标（32 字节对齐）
    int64_t blockLength = alignedBlock;                  // 本核长度
    if (coreId == coreNum - 1) {
        // 最后一个核吃全部余量（含 < TILE_LEN 的非对齐部分）
        blockLength = totalLength - startIdx;
    }
    if (blockLength < 0) {
        blockLength = 0;
    }
    // 单块化决策
    int64_t alignedLen = (blockLength / TILE_LEN) * TILE_LEN;
    int64_t ubLength = alignedLen;
    if (ubLength <= 0 || ubLength * static_cast<int64_t>(sizeof(T)) * (2 * BUFFER_NUM) > MAX_UB_BYTES) {
        ubLength = TILE_LEN;
    }
    this->startIdx_ = startIdx;
    this->blockLength_ = blockLength;
    this->alignedLen_ = alignedLen;
    this->ubLength_ = ubLength;
    inputGMX.SetGlobalBuffer((__gm__ T*)x + this->startIdx_, this->blockLength_);
    outputGMY.SetGlobalBuffer((__gm__ T*)y + this->startIdx_, this->blockLength_);
    pipe.InitBuffer(inputQueueX, BUFFER_NUM, static_cast<uint32_t>(this->ubLength_ * sizeof(T)));
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, static_cast<uint32_t>(this->ubLength_ * sizeof(T)));
}

template <typename T>
__aicore__ inline void Relu<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
    DataCopy(xLocal, inputGMX[static_cast<int32_t>(progress)], static_cast<int32_t>(currentNum));
    inputQueueX.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::Compute(int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();
    // y = max(0, x)
    AscendC::Relu(yLocal, xLocal, static_cast<int32_t>(currentNum));
    outputQueueY.EnQue<T>(yLocal);
    inputQueueX.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> yLocal = outputQueueY.DeQue<T>();
    DataCopy(outputGMY[static_cast<int32_t>(progress)], yLocal, static_cast<int32_t>(currentNum));
    outputQueueY.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::Process()
{
    int64_t mainCount = this->alignedLen_ / this->ubLength_; // 对齐部分的块数
    for (int64_t i = 0; i < mainCount; i++) {
        CopyIn(i * this->ubLength_, this->ubLength_);
        Compute(this->ubLength_);
        CopyOut(i * this->ubLength_, this->ubLength_);
    }
    int64_t tail = this->blockLength_ - this->alignedLen_; // 非对齐残余（< TILE_LEN）
    if (tail > 0) {
        // 尾块：仅最后一个核可能出现。按 TILE_LEN 整块读取/写出，越界落在 GM 对齐填充区，安全。
        CopyIn(this->alignedLen_, TILE_LEN);
        Compute(TILE_LEN);
        CopyOut(this->alignedLen_, TILE_LEN);
    }
}

} // namespace NsRelu
#endif // RELU_H
