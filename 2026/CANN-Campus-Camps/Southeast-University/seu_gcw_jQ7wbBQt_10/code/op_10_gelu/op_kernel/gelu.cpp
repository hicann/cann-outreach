// Kernel侧核函数实现
#include "kernel_operator.h"

#include "gelu_tiling.h"
#include "tiling_key_gelu.h"

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;
constexpr float INV_SQRT2 = 0.70710678118f;

template <class T>
class KernelGelu {
public:
    __aicore__ inline KernelGelu() {}
    __aicore__ inline void Init(GM_ADDR src_gm, GM_ADDR dst_gm,
                                uint32_t ALIGN_NUM, uint32_t blockSize,
                                uint32_t coreSize, uint32_t extraCores,
                                uint32_t lastRemainder) {
        uint32_t blockIdx = static_cast<uint32_t>(GetBlockIdx());
        uint32_t blockNum = static_cast<uint32_t>(GetBlockNum());
        ASSERT(blockNum != 0 && "block dim can not be zero");

        // 当前核负责的元素个数: 前extraCores个核多分ALIGN_NUM, 最后一个核多分lastRemainder
        this->tileLength = blockSize;
        uint32_t coreLen = coreSize + (blockIdx < extraCores ? ALIGN_NUM : 0) +
                           (blockIdx + 1 == blockNum ? lastRemainder : 0);
        this->coreLength = coreLen;
        // 向上对齐到ALIGN_NUM, 保证DataCopy按32字节对齐
        this->blockLength = coreLen + (coreLen % ALIGN_NUM ? ALIGN_NUM - coreLen % ALIGN_NUM : 0);
        this->tileNum = this->blockLength / this->tileLength +
                        (this->blockLength % this->tileLength > 0);

        // 前extraCores个核各多拿了ALIGN_NUM, 计算本核的起始偏移
        uint32_t startPointer = coreSize * blockIdx +
                                ALIGN_NUM * (blockIdx < extraCores ? blockIdx : extraCores);
        src_global.SetGlobalBuffer((__gm__ T *)src_gm + startPointer, this->blockLength);
        dst_global.SetGlobalBuffer((__gm__ T *)dst_gm + startPointer, this->blockLength);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(T));
        pipe.InitBuffer(outQueue, BUFFER_NUM, this->tileLength * sizeof(T));
        if constexpr (sizeof(T) == 2) {
            pipe.InitBuffer(calcBuf, this->tileLength * sizeof(float));
        }
    }

    __aicore__ inline void Process() {
        if (this->tileNum == 0) {
            return;
        }
        uint32_t loopCount = this->tileNum;
        if (loopCount == 1) {
            CopyIn(0, this->coreLength, this->blockLength);
            Compute(0, this->blockLength);
            CopyOut(0, this->coreLength, this->blockLength);
            return;
        }

        uint32_t last = loopCount - 1;
        uint32_t done = this->tileLength * last;
        uint32_t lastCalcLength = this->blockLength - done;
        uint32_t lastValidLength = this->coreLength - done;

        // 提前搬入下一块数据，让下一次DMA与当前向量计算重叠。
        CopyIn(0, this->tileLength, this->tileLength);
        for (uint32_t i = 0; i < last; i++) {
            if (i + 1 == last) {
                CopyIn(i + 1, lastValidLength, lastCalcLength);
            } else {
                CopyIn(i + 1, this->tileLength, this->tileLength);
            }
            Compute(i, this->tileLength);
            CopyOut(i, this->tileLength, this->tileLength);
        }
        Compute(last, lastCalcLength);
        CopyOut(last, lastValidLength, lastCalcLength);
    }

private:
    __aicore__ inline void CopyIn(uint32_t process, uint32_t validLength, uint32_t calcLength) {
        LocalTensor<T> srcLocal = inQueueX.AllocTensor<T>();
        if (validLength == calcLength) {
            DataCopy(srcLocal, src_global[process * this->tileLength], validLength);
        } else {
            DataCopyExtParams copyParams{1,
                static_cast<uint32_t>(validLength * sizeof(T)), 0, 0, 0};
            DataCopyPadExtParams<T> padParams{true, 0, 0, 0};
            DataCopyPad(srcLocal, src_global[process * this->tileLength], copyParams, padParams);
        }
        inQueueX.EnQue(srcLocal);
    }

    __aicore__ inline void Compute(uint32_t process, uint32_t length) {
        LocalTensor<T> dst = outQueue.AllocTensor<T>();
        LocalTensor<T> src = inQueueX.DeQue<T>();
        if constexpr (sizeof(T) == 2) {
            // FP16输入先提升到FP32计算erf部分，再将cdf转回FP16，保证双千分之一精度。
            LocalTensor<float> work = calcBuf.Get<float>();
            Cast(work, src, RoundMode::CAST_NONE, length);
            Muls(work, work, INV_SQRT2, length);
            Erf(work, work, length);
            Adds(work, work, 1.0f, length);
            Muls(work, work, 0.5f, length);
            Cast(dst, work, RoundMode::CAST_NONE, length);
            Mul(dst, dst, src, length);
        } else {
            // GELU(x) = x * 0.5 * (1 + erf(x / sqrt(2))), 直接在输出缓冲上计算
            Muls(dst, src, (T)INV_SQRT2, length);
            Erf(dst, dst, length);
            Adds(dst, dst, (T)1, length);
            Muls(dst, dst, (T)0.5, length);
            Mul(dst, dst, src, length);
        }
        outQueue.EnQue<T>(dst);
        inQueueX.FreeTensor(src);
    }

    __aicore__ inline void CopyOut(uint32_t process, uint32_t validLength, uint32_t calcLength) {
        LocalTensor<T> dstLocal = outQueue.DeQue<T>();
        if (validLength == calcLength) {
            DataCopy(dst_global[process * this->tileLength], dstLocal, validLength);
        } else {
            DataCopyExtParams copyParams{1,
                static_cast<uint32_t>(validLength * sizeof(T)), 0, 0, 0};
            DataCopyPad(dst_global[process * this->tileLength], dstLocal, copyParams);
        }
        outQueue.FreeTensor(dstLocal);
    }

private:
    GlobalTensor<T> src_global;
    GlobalTensor<T> dst_global;
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue;
    TBuf<QuePosition::VECCALC> calcBuf;
    uint32_t coreLength;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
};

template <typename DT_INPUT_X>
__global__ __aicore__ void gelu(GM_ADDR input_x, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(GeluTilingData);
    GET_TILING_DATA_WITH_STRUCT(GeluTilingData, tiling_data, tiling);
    KernelGelu<DT_INPUT_X> op;
    op.Init(input_x, output,
            tiling_data.ALIGN_NUM,
            tiling_data.blockSize,
            tiling_data.coreSize,
            tiling_data.extraCores,
            tiling_data.lastRemainder);
    op.Process();
}
