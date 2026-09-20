#include "kernel_operator.h"

#include "gelu_tiling.h"
#include "tiling_key_gelu.h"

namespace NsGelu {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 1;

template <typename T>
class KernelGelu {
public:
    __aicore__ inline KernelGelu() {}

    __aicore__ inline void Init(GM_ADDR input_x, GM_ADDR output, const GeluTilingData *tiling) {
        int64_t blockOffset = static_cast<int64_t>(tiling->blockFactor) * GetBlockIdx();
        int64_t remainderLength = static_cast<int64_t>(tiling->length) - blockOffset;
        blockLength_ = (remainderLength > static_cast<int64_t>(tiling->blockFactor))
                           ? static_cast<int64_t>(tiling->blockFactor)
                           : remainderLength;
        if (blockLength_ < 0) {
            blockLength_ = 0;
        }
        ubLength_ = (tiling->ubFactor > 0) ? static_cast<int64_t>(tiling->ubFactor) : blockLength_;
        if (ubLength_ <= 0) {
            ubLength_ = 1;
        }

        inputGMX.SetGlobalBuffer((__gm__ T *)input_x + blockOffset, blockLength_);
        outputGMY.SetGlobalBuffer((__gm__ T *)output + blockOffset, blockLength_);

        // 1 输入 + 1 输出 + 2 个临时 buffer，共 4 份，均在 host 侧按 UB/4 预留
        pipe.InitBuffer(inputQueueX, BUFFER_NUM, ubLength_ * sizeof(T));
        pipe.InitBuffer(outputQueueY, BUFFER_NUM, ubLength_ * sizeof(T));
        pipe.InitBuffer(tmpBufA, ubLength_ * sizeof(T));
    }

    __aicore__ inline void Process() {
        if (blockLength_ <= 0) {
            return;
        }
        int64_t loopCount = (blockLength_ + ubLength_ - 1) / ubLength_;
        for (int64_t progress = 0; progress < loopCount; ++progress) {
            int64_t currentNum = blockLength_ - progress * ubLength_;
            if (currentNum > ubLength_) {
                currentNum = ubLength_;
            }
            CopyIn(progress, currentNum);
            Compute(currentNum);
            CopyOut(progress, currentNum);
        }
    }

private:
    __aicore__ inline void CopyIn(int64_t progress, int64_t currentNum) {
        LocalTensor<T> xLocal = inputQueueX.template AllocTensor<T>();
        if ((currentNum * sizeof(T)) % 32 == 0) {
            AscendC::DataCopy(xLocal, inputGMX[progress * ubLength_], static_cast<uint32_t>(currentNum));
        } else {
            DataCopyParams copyParams{1, static_cast<uint16_t>(currentNum * sizeof(T)), 0, 0};
            DataCopyPad(xLocal, inputGMX[progress * ubLength_], copyParams, {false, 0, 0, 0});
        }
        inputQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(int64_t currentNum) {
        uint32_t count = static_cast<uint32_t>(currentNum);
        LocalTensor<T> xLocal = inputQueueX.template DeQue<T>();
        LocalTensor<T> yLocal = outputQueueY.template AllocTensor<T>();
        LocalTensor<T> tmpA = tmpBufA.template Get<T>();

        // gelu(x) = 0.5 * x * (1 + erf(x / sqrt(2)))  （与 torch 默认 erf 精度一致）
        AscendC::Muls(tmpA, xLocal, static_cast<T>(0.70710678118654752440), count); // x / sqrt(2)
        AscendC::Erf(tmpA, tmpA, count);
        AscendC::Adds(tmpA, tmpA, static_cast<T>(1), count);
        AscendC::Muls(tmpA, tmpA, static_cast<T>(0.5), count);
        AscendC::Mul(yLocal, xLocal, tmpA, count);

        outputQueueY.template EnQue<T>(yLocal);
        inputQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(int64_t progress, int64_t currentNum) {
        LocalTensor<T> yLocal = outputQueueY.template DeQue<T>();
        if ((currentNum * sizeof(T)) % 32 == 0) {
            AscendC::DataCopy(outputGMY[progress * ubLength_], yLocal, static_cast<uint32_t>(currentNum));
        } else {
            DataCopyParams copyParams{1, static_cast<uint16_t>(currentNum * sizeof(T)), 0, 0};
            DataCopyPad(outputGMY[progress * ubLength_], yLocal, copyParams);
        }
        outputQueueY.FreeTensor(yLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueY;
    TBuf<QuePosition::VECCALC> tmpBufA;
    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;
    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
};

}  // namespace NsGelu

template <typename DT_INPUT_X>
__global__ __aicore__ void gelu(GM_ADDR input_x, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(GeluTilingData);
    GET_TILING_DATA_WITH_STRUCT(GeluTilingData, tiling_data, tiling);
    NsGelu::KernelGelu<DT_INPUT_X> op;
    op.Init(input_x, output, &tiling_data);
    op.Process();
}


