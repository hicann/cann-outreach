/*!
 * \file square.h
 * \brief Square 算子 kernel 类定义
 */

#ifndef SQUARE_H
#define SQUARE_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "square_tiling_data.h"
#include "square_tiling_key.h"

namespace NsSquare {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

template <typename T>
class Square {
public:
    __aicore__ inline Square(){};

    __aicore__ inline void Init(GM_ADDR input_x, GM_ADDR output, const SquareTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t progress, int64_t currentNum);
    __aicore__ inline void CopyOut(int64_t progress, int64_t currentNum);
    __aicore__ inline void Compute(int64_t currentNum);
    __aicore__ inline void ProcessSingleTile();

private:
    TPipe pipe;
    TBuf<QuePosition::VECCALC> singleInput;
    TBuf<QuePosition::VECCALC> singleOutput;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueY;

    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;

    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
};

template <typename T>
__aicore__ inline void Square<T>::Init(GM_ADDR input_x, GM_ADDR output, const SquareTilingData* tilingData)
{
    // Distribute 256-byte units evenly, including the remainder units.
    constexpr int64_t alignNum = 256 / sizeof(T);
    const int64_t units = (tilingData->totalNum + alignNum - 1) / alignNum;
    const int64_t coreCount = GetBlockNum();
    const int64_t coreId = GetBlockIdx();
    const int64_t baseUnits = units / coreCount;
    const int64_t extra = units % coreCount;
    const int64_t offset = (coreId * baseUnits + (coreId < extra ? coreId : extra)) * alignNum;
    if (offset >= tilingData->totalNum) {
        return;
    }
    const int64_t assigned = (baseUnits + (coreId < extra ? 1 : 0)) * alignNum;
    blockLength_ = tilingData->totalNum - offset;
    blockLength_ = blockLength_ < assigned ? blockLength_ : assigned;
    ubLength_ = tilingData->ubFactor;
    inputGMX.SetGlobalBuffer((__gm__ T*)input_x + offset, blockLength_);
    outputGMY.SetGlobalBuffer((__gm__ T*)output + offset, blockLength_);
    if (blockLength_ <= ubLength_) {
        pipe.InitBuffer(singleInput, ubLength_ * sizeof(T));
        pipe.InitBuffer(singleOutput, ubLength_ * sizeof(T));
    } else {
        pipe.InitBuffer(inputQueueX, BUFFER_NUM, ubLength_ * sizeof(T));
        pipe.InitBuffer(outputQueueY, BUFFER_NUM, ubLength_ * sizeof(T));
    }
}

template <typename T>
__aicore__ inline void Square<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> inputLocal = inputQueueX.AllocTensor<T>();
    if (currentNum % (32 / sizeof(T)) == 0) {
        DataCopy(inputLocal, inputGMX[progress], static_cast<uint32_t>(currentNum));
    } else {
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(currentNum * sizeof(T)), 0, 0, 0};
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyPad(inputLocal, inputGMX[progress], copyParams, padParams);
    }
    inputQueueX.EnQue(inputLocal);
}

template <typename T>
__aicore__ inline void Square<T>::Compute(int64_t currentNum)
{
    LocalTensor<T> inputLocal = inputQueueX.DeQue<T>();
    LocalTensor<T> outputLocal = outputQueueY.AllocTensor<T>();
    Mul(outputLocal, inputLocal, inputLocal, static_cast<int32_t>(currentNum));
    outputQueueY.EnQue(outputLocal);
    inputQueueX.FreeTensor(inputLocal);
}

template <typename T>
__aicore__ inline void Square<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> outputLocal = outputQueueY.DeQue<T>();
    if (currentNum % (32 / sizeof(T)) == 0) {
        DataCopy(outputGMY[progress], outputLocal, static_cast<uint32_t>(currentNum));
    } else {
        DataCopyExtParams copyParams{1, static_cast<uint32_t>(currentNum * sizeof(T)), 0, 0, 0};
        DataCopyPad(outputGMY[progress], outputLocal, copyParams);
    }
    outputQueueY.FreeTensor(outputLocal);
}

template <typename T>
__aicore__ inline void Square<T>::ProcessSingleTile()
{
    LocalTensor<T> inputLocal = singleInput.Get<T>();
    LocalTensor<T> outputLocal = singleOutput.Get<T>();
    const uint32_t count = static_cast<uint32_t>(blockLength_);
    const bool aligned = count % (32 / sizeof(T)) == 0;
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
    if (aligned) {
        DataCopy(inputLocal, inputGMX, count);
    } else {
        DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
        DataCopyPad(inputLocal, inputGMX, copyParams, padParams);
    }
    const event_t inputReady = static_cast<event_t>(pipe.FetchEventID(HardEvent::MTE2_V));
    SetFlag<HardEvent::MTE2_V>(inputReady);
    WaitFlag<HardEvent::MTE2_V>(inputReady);
    Mul(outputLocal, inputLocal, inputLocal, static_cast<int32_t>(count));
    const event_t outputReady = static_cast<event_t>(pipe.FetchEventID(HardEvent::V_MTE3));
    SetFlag<HardEvent::V_MTE3>(outputReady);
    WaitFlag<HardEvent::V_MTE3>(outputReady);
    if (aligned) {
        DataCopy(outputGMY, outputLocal, count);
    } else {
        DataCopyPad(outputGMY, outputLocal, copyParams);
    }
}

template <typename T>
__aicore__ inline void Square<T>::Process()
{
    if (blockLength_ == 0) {
        return;
    }
    if (blockLength_ <= ubLength_) {
        ProcessSingleTile();
        return;
    }
    CopyIn(0, ubLength_);
    for (int64_t progress = 0; progress < blockLength_; progress += ubLength_) {
        const int64_t remaining = blockLength_ - progress;
        const int64_t currentNum = remaining < ubLength_ ? remaining : ubLength_;
        const int64_t next = progress + ubLength_;
        if (next < blockLength_) {
            const int64_t nextRemaining = blockLength_ - next;
            CopyIn(next, nextRemaining < ubLength_ ? nextRemaining : ubLength_);
        }
        Compute(currentNum);
        CopyOut(progress, currentNum);
    }
}

} // namespace NsSquare
#endif // SQUARE_H
