#include "kernel_operator.h"

#include "gelu_tiling.h"
#include "tiling_key_gelu.h"

using namespace AscendC;

constexpr uint32_t BUFFER_NUM = 1;
constexpr uint32_t TILE_LENGTH = 6144;

template <typename T>
class KernelGelu {
public:
    __aicore__ inline KernelGelu() {}

    __aicore__ inline void Init(GM_ADDR inputX, GM_ADDR output,
                                uint32_t totalLength,
                                uint32_t smallBlockLength,
                                uint32_t bigCoreCount)
    {
        constexpr uint32_t ALIGN_NUM = 32 / sizeof(T);
        const uint32_t blockIdx = GetBlockIdx();
        const uint32_t precedingBigCores =
            blockIdx < bigCoreCount ? blockIdx : bigCoreCount;
        const uint32_t blockStart =
            blockIdx * smallBlockLength + precedingBigCores * ALIGN_NUM;
        const uint32_t scheduledLength =
            smallBlockLength + (blockIdx < bigCoreCount ? ALIGN_NUM : 0);

        if (blockStart < totalLength) {
            const uint32_t remaining = totalLength - blockStart;
            blockLength_ = remaining < scheduledLength ? remaining : scheduledLength;
        } else {
            blockLength_ = 0;
        }

        inputGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ T *>(inputX) + blockStart, blockLength_);
        outputGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ T *>(output) + blockStart, blockLength_);

        pipe_.InitBuffer(inputQueue_, BUFFER_NUM, TILE_LENGTH * sizeof(T));
        pipe_.InitBuffer(outputQueue_, BUFFER_NUM, TILE_LENGTH * sizeof(T));
        pipe_.InitBuffer(calcBuffer_, TILE_LENGTH * sizeof(T));
    }

    __aicore__ inline void Process()
    {
        for (uint32_t offset = 0; offset < blockLength_; offset += TILE_LENGTH) {
            const uint32_t remaining = blockLength_ - offset;
            const uint32_t count = remaining < TILE_LENGTH ? remaining : TILE_LENGTH;
            CopyIn(offset, count);
            Compute(count);
            CopyOut(offset, count);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t offset, uint32_t count)
    {
        LocalTensor<T> inputLocal = inputQueue_.AllocTensor<T>();
        if ((count * sizeof(T)) % 32 == 0) {
            DataCopy(inputLocal, inputGm_[offset], count);
        } else {
            DataCopyExtParams copyParams{
                1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
            DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
            DataCopyPad(inputLocal, inputGm_[offset], copyParams, padParams);
        }
        inputQueue_.EnQue(inputLocal);
    }

    __aicore__ inline void Compute(uint32_t count)
    {
        LocalTensor<T> inputLocal = inputQueue_.DeQue<T>();
        LocalTensor<T> outputLocal = outputQueue_.AllocTensor<T>();
        LocalTensor<T> erfLocal = calcBuffer_.Get<T>();

        Muls(outputLocal, inputLocal,
             static_cast<T>(0.7071067811865475244), count);
        Erf(erfLocal, outputLocal, count);
        Adds(erfLocal, erfLocal, static_cast<T>(1.0), count);
        Mul(outputLocal, erfLocal, inputLocal, count);
        Muls(outputLocal, outputLocal, static_cast<T>(0.5), count);

        outputQueue_.EnQue(outputLocal);
        inputQueue_.FreeTensor(inputLocal);
    }

    __aicore__ inline void CopyOut(uint32_t offset, uint32_t count)
    {
        LocalTensor<T> outputLocal = outputQueue_.DeQue<T>();
        if ((count * sizeof(T)) % 32 == 0) {
            DataCopy(outputGm_[offset], outputLocal, count);
        } else {
            DataCopyExtParams copyParams{
                1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
            DataCopyPad(outputGm_[offset], outputLocal, copyParams);
        }
        outputQueue_.FreeTensor(outputLocal);
    }

private:
    TPipe pipe_;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueue_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueue_;
    TBuf<QuePosition::VECCALC> calcBuffer_;
    GlobalTensor<T> inputGm_;
    GlobalTensor<T> outputGm_;
    uint32_t blockLength_ = 0;
};

template <typename DT_INPUT_X>
__global__ __aicore__ void gelu(GM_ADDR input_x, GM_ADDR output,
                                GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(GeluTilingData);
    GET_TILING_DATA_WITH_STRUCT(GeluTilingData, tilingData, tiling);

    KernelGelu<DT_INPUT_X> op;
    op.Init(input_x, output, tilingData.length,
            tilingData.smallBlockLength, tilingData.bigCoreCount);
    op.Process();
}