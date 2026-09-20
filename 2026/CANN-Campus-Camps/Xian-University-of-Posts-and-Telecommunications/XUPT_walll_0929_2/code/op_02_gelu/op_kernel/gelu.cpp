// Kernel侧核函数实现
#include "kernel_operator.h"

#include "gelu_tiling.h"
#include "tiling_key_gelu.h"

constexpr uint32_t BUFFER_NUM = 2;

template <class DT_X>
class KernelGelu {
public:
    __aicore__ inline KernelGelu() {}

    __aicore__ inline void Init(GM_ADDR input_x, GM_ADDR output,
                               uint32_t totalLength, uint32_t tileLength)
    {
        this->tileLength = tileLength;
        const uint32_t coreNum = AscendC::GetBlockNum();
        const uint32_t coreIdx = AscendC::GetBlockIdx();
        const uint32_t baseLength = totalLength / coreNum;
        const uint32_t remainder = totalLength % coreNum;
        blockLength = baseLength + (coreIdx < remainder ? 1 : 0);
        const uint32_t offset = coreIdx * baseLength +
            (coreIdx < remainder ? coreIdx : remainder);
        xGm.SetGlobalBuffer((__gm__ DT_X*)input_x + offset, blockLength);
        yGm.SetGlobalBuffer((__gm__ DT_X*)output + offset, blockLength);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, tileLength * sizeof(DT_X));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, tileLength * sizeof(DT_X));
        pipe.InitBuffer(xFloatBuf, tileLength * sizeof(float));
        pipe.InitBuffer(argBuf, tileLength * sizeof(float));
        pipe.InitBuffer(erfBuf, tileLength * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        for (uint32_t offset = 0; offset < blockLength;) {
            const uint32_t remaining = blockLength - offset;
            currentNum = remaining < tileLength ? remaining : tileLength;
            CopyIn(offset);
            Compute();
            CopyOut(offset);
            offset += currentNum;
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t offset)
    {
        AscendC::LocalTensor<DT_X> xLocal = inQueueX.AllocTensor<DT_X>();
        AscendC::DataCopyExtParams copyParams = {
            1, static_cast<uint32_t>(currentNum * sizeof(DT_X)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<DT_X> padParams = {false, 0, 0, 0};
        AscendC::DataCopyPad(xLocal, xGm[offset], copyParams, padParams);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute()
    {
        AscendC::LocalTensor<DT_X> xLocal = inQueueX.DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> yLocal = outQueueY.AllocTensor<DT_X>();
        AscendC::LocalTensor<float> xFloat = xFloatBuf.Get<float>();
        AscendC::LocalTensor<float> arg = argBuf.Get<float>();
        AscendC::LocalTensor<float> erfValue = erfBuf.Get<float>();

        if constexpr (sizeof(DT_X) == sizeof(half)) {
            AscendC::Cast(xFloat, xLocal, AscendC::RoundMode::CAST_NONE, currentNum);
        } else {
            AscendC::Adds(xFloat, xLocal, 0.0f, currentNum);
        }

        // Follow the repository's erf reference; retain FP32 intermediates for FP16 input.
        AscendC::Muls(arg, xFloat, 0.7071067811865476f, currentNum);
        AscendC::Erf(erfValue, arg, currentNum);
        AscendC::Adds(arg, erfValue, 1.0f, currentNum);
        AscendC::Muls(erfValue, xFloat, 0.5f, currentNum);
        AscendC::Mul(arg, erfValue, arg, currentNum);

        if constexpr (sizeof(DT_X) == sizeof(half)) {
            AscendC::Cast(yLocal, arg, AscendC::RoundMode::CAST_RINT, currentNum);
        } else {
            AscendC::Adds(yLocal, arg, 0.0f, currentNum);
        }
        outQueueY.EnQue(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(uint32_t offset)
    {
        AscendC::LocalTensor<DT_X> yLocal = outQueueY.DeQue<DT_X>();
        AscendC::DataCopyExtParams copyParams = {
            1, static_cast<uint32_t>(currentNum * sizeof(DT_X)), 0, 0, 0};
        AscendC::DataCopyPad(yGm[offset], yLocal, copyParams);
        outQueueY.FreeTensor(yLocal);
    }

    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> xFloatBuf;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> argBuf;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> erfBuf;
    AscendC::GlobalTensor<DT_X> xGm, yGm;
    uint32_t blockLength;
    uint32_t tileLength;
    uint32_t currentNum;
};

template <typename DT_X>
 __global__ __aicore__ void gelu(GM_ADDR input_x, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(GeluTilingData);
    GET_TILING_DATA_WITH_STRUCT(GeluTilingData, tiling_data, tiling);
    KernelGelu<DT_X> op;
    op.Init(input_x, output, tiling_data.totalLength, tiling_data.tileLength);
    op.Process();
}
