/*!
 * \file relu.h
 * \brief Relu Ascend C kernel implementation
 */

#ifndef RELU_H
#define RELU_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "relu_tiling_data.h"
#include "relu_tiling_key.h"

namespace NsRelu {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

template <typename DT_X, int32_t BUFFER_COUNT = BUFFER_NUM>
class Relu {
public:
    __aicore__ inline Relu() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const ReluTilingData* tilingData)
    {
        this->totalNum = tilingData->totalNum;
        this->blockFactor = tilingData->blockFactor;
        this->tileLength = tilingData->ubFactor;

        const uint64_t blockOffset = this->blockFactor * AscendC::GetBlockIdx();
        this->blockLength = blockOffset < this->totalNum
            ? Min(this->blockFactor, this->totalNum - blockOffset)
            : 0;
        if (this->blockLength == 0) {
            return;
        }

        inputGMX.SetGlobalBuffer(
            reinterpret_cast<__gm__ DT_X*>(x) + blockOffset, this->blockLength);
        outputGMY.SetGlobalBuffer(
            reinterpret_cast<__gm__ DT_X*>(y) + blockOffset, this->blockLength);

        pipe.InitBuffer(inputQueueX, BUFFER_COUNT, this->tileLength * sizeof(DT_X));
        pipe.InitBuffer(outputQueueY, BUFFER_COUNT, this->tileLength * sizeof(DT_X));
    }

    __aicore__ inline void Process()
    {
        if (this->blockLength == 0) {
            return;
        }
        // The entry selects this specialization only when a whole core fits in one tile.
        if constexpr (BUFFER_COUNT == 1) {
            CopyIn(0, static_cast<uint32_t>(this->blockLength));
            Compute();
            CopyOut(0);
        } else {
            uint64_t offset = 0;
            uint64_t remaining = this->blockLength;
            while (remaining > this->tileLength) {
                CopyIn(offset, static_cast<uint32_t>(this->tileLength));
                Compute();
                CopyOut(offset);
                offset += this->tileLength;
                remaining -= this->tileLength;
            }
            CopyIn(offset, static_cast<uint32_t>(remaining));
            Compute();
            CopyOut(offset);
        }
    }

private:
    __aicore__ inline void CopyIn(uint64_t offset, uint32_t count)
    {
        this->currentTileLength = count;
        this->currentTileAligned = count % (32 / sizeof(DT_X)) == 0;

        LocalTensor<DT_X> xLocal = inputQueueX.template AllocTensor<DT_X>();
        if (this->currentTileAligned) {
            DataCopy(xLocal, inputGMX[offset], count);
        } else {
            // Copy only valid GM bytes; never round a tail copy up into the next core.
            DataCopyExtParams copyParams {
                1, static_cast<uint32_t>(count * sizeof(DT_X)), 0, 0, 0};
            DataCopyPadExtParams<DT_X> padParams {false, 0, 0, static_cast<DT_X>(0)};
            DataCopyPad(xLocal, inputGMX[offset], copyParams, padParams);
        }
        inputQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute()
    {
        LocalTensor<DT_X> xLocal = inputQueueX.template DeQue<DT_X>();
        LocalTensor<DT_X> yLocal = outputQueueY.template AllocTensor<DT_X>();
        AscendC::Relu(yLocal, xLocal, static_cast<int32_t>(this->currentTileLength));
        outputQueueY.EnQue(yLocal);
        inputQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(uint64_t offset)
    {
        LocalTensor<DT_X> yLocal = outputQueueY.template DeQue<DT_X>();
        if (this->currentTileAligned) {
            DataCopy(outputGMY[offset], yLocal, this->currentTileLength);
        } else {
            DataCopyExtParams copyParams {
                1, static_cast<uint32_t>(this->currentTileLength * sizeof(DT_X)), 0, 0, 0};
            DataCopyPad(outputGMY[offset], yLocal, copyParams);
        }
        outputQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline uint64_t Min(uint64_t lhs, uint64_t rhs) const
    {
        return lhs < rhs ? lhs : rhs;
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_COUNT> inputQueueX;
    TQue<QuePosition::VECOUT, BUFFER_COUNT> outputQueueY;

    GlobalTensor<DT_X> inputGMX;
    GlobalTensor<DT_X> outputGMY;

    uint64_t totalNum;
    uint64_t blockFactor;
    uint64_t blockLength;
    uint64_t tileLength;
    uint32_t currentTileLength;
    bool currentTileAligned;
};

// A one-shot tile has no buffer reuse or inter-tile overlap. Use one UB tensor
// and explicit producer/consumer events instead of input/output queue bookkeeping.
template <typename DT_X>
class Relu<DT_X, 1> {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const ReluTilingData* tilingData)
    {
        const uint64_t offset = tilingData->blockFactor * AscendC::GetBlockIdx();
        count = 0;
        if (offset >= tilingData->totalNum) {
            return;
        }
        const uint64_t remaining = tilingData->totalNum - offset;
        count = static_cast<uint32_t>(remaining < tilingData->blockFactor
            ? remaining : tilingData->blockFactor);
        inputGM.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X*>(x) + offset, count);
        outputGM.SetGlobalBuffer(reinterpret_cast<__gm__ DT_X*>(y) + offset, count);
        const uint32_t bufferBytes = (count * sizeof(DT_X) + 31) / 32 * 32;
        pipe.InitBuffer(buffer, bufferBytes);
    }

    __aicore__ inline void Process()
    {
        if (count == 0) {
            return;
        }
        LocalTensor<DT_X> local = buffer.Get<DT_X>();
        const bool aligned = count % (32 / sizeof(DT_X)) == 0;
        if (aligned) {
            DataCopy(local, inputGM, count);
        } else {
            DataCopyExtParams params {1, static_cast<uint32_t>(count * sizeof(DT_X)), 0, 0, 0};
            DataCopyPadExtParams<DT_X> padding {false, 0, 0, static_cast<DT_X>(0)};
            DataCopyPad(local, inputGM, params, padding);
        }

        const event_t inputReady = static_cast<event_t>(pipe.FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(inputReady);
        WaitFlag<HardEvent::MTE2_V>(inputReady);
        // Fully overlapping operands: each element reads and writes its own position.
        AscendC::Relu(local, local, static_cast<int32_t>(count));
        const event_t outputReady = static_cast<event_t>(pipe.FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(outputReady);
        WaitFlag<HardEvent::V_MTE3>(outputReady);

        if (aligned) {
            DataCopy(outputGM, local, count);
        } else {
            DataCopyExtParams params {1, static_cast<uint32_t>(count * sizeof(DT_X)), 0, 0, 0};
            DataCopyPad(outputGM, local, params);
        }
        // Complete the GM write before leaving the lifetime of this buffer.
        const event_t outputDone = static_cast<event_t>(pipe.FetchEventID(HardEvent::MTE3_S));
        SetFlag<HardEvent::MTE3_S>(outputDone);
        WaitFlag<HardEvent::MTE3_S>(outputDone);
    }

private:
    TPipe pipe;
    TBuf<TPosition::VECIN> buffer;
    GlobalTensor<DT_X> inputGM;
    GlobalTensor<DT_X> outputGM;
    uint32_t count;
};

} // namespace NsRelu
#endif // RELU_H
