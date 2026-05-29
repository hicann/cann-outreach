#include "kernel_operator.h"
#include "tanh_custom_tiling.h"

using DTYPE_X = half;
using DTYPE_Y = half;

constexpr int32_t BUFFER_NUM = 2;

class KernelTanh {
public:
    __aicore__ inline KernelTanh() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y,
                                uint32_t totalLength, uint32_t tileNum,
                                uint32_t coreOffset)   // 增加核偏移参数
    {
        this->tileNum = tileNum;
        this->coreLength = totalLength;

        uint32_t totalBlocks = tileNum * BUFFER_NUM;
        this->blockLength = (totalLength + totalBlocks - 1) / totalBlocks;
        this->tileLength = this->blockLength;

        // 通过偏移设置本核起始地址（Init内部允许类型转换）
        xGm.SetGlobalBuffer((__gm__ DTYPE_X*)x + coreOffset, totalLength);
        yGm.SetGlobalBuffer((__gm__ DTYPE_Y*)y + coreOffset, totalLength);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->blockLength * sizeof(DTYPE_X));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->blockLength * sizeof(DTYPE_Y));
        pipe.InitBuffer(tmpBuf0, this->blockLength * sizeof(DTYPE_X));
        pipe.InitBuffer(tmpBuf1, this->blockLength * sizeof(DTYPE_X));
        pipe.InitBuffer(tmpBuf2, this->blockLength * sizeof(DTYPE_X));
    }

    __aicore__ inline void Process()
    {
        int32_t loopCount = this->tileNum * BUFFER_NUM;
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        uint32_t offset = progress * this->blockLength;
        uint32_t length = this->blockLength;
        if (offset + length > this->coreLength) {
            length = this->coreLength - offset;
        }

        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.AllocTensor<DTYPE_X>();
        AscendC::DataCopy(xLocal, xGm[offset], length);
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(int32_t progress)
    {
        uint32_t offset = progress * this->blockLength;
        uint32_t length = this->blockLength;
        if (offset + length > this->coreLength) {
            length = this->coreLength - offset;
        }

        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.DeQue<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_X> tmp0 = tmpBuf0.Get<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_X> tmp1 = tmpBuf1.Get<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_X> tmp2 = tmpBuf2.Get<DTYPE_X>();

        AscendC::Exp(tmp0, xLocal, length);
        AscendC::Muls(tmp1, xLocal, (DTYPE_X)(-1.0f), length);
        AscendC::Exp(tmp1, tmp1, length);
        AscendC::Add(tmp2, tmp0, tmp1, length);
        AscendC::Sub(tmp0, tmp0, tmp1, length);

        AscendC::LocalTensor<DTYPE_Y> yLocal = outQueueY.AllocTensor<DTYPE_Y>();
        AscendC::Div(yLocal, tmp0, tmp2, length);
        outQueueY.EnQue<DTYPE_Y>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress)
    {
        uint32_t offset = progress * this->blockLength;
        uint32_t length = this->blockLength;
        if (offset + length > this->coreLength) {
            length = this->coreLength - offset;
        }

        AscendC::LocalTensor<DTYPE_Y> yLocal = outQueueY.DeQue<DTYPE_Y>();
        AscendC::DataCopy(yGm[offset], yLocal, length);
        outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmpBuf0, tmpBuf1, tmpBuf2;
    AscendC::GlobalTensor<DTYPE_X> xGm;
    AscendC::GlobalTensor<DTYPE_Y> yGm;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
    uint32_t coreLength;
};

extern "C" __global__ __aicore__ void tanh_custom(GM_ADDR x, GM_ADDR y,
                                                  GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(TanhCustomTilingData);
    GET_TILING_DATA(tilingData, tiling);

    uint32_t totalLength = tilingData.totalLength;
    uint32_t tileNum = tilingData.tileNum;

    uint32_t blockNum = (uint32_t)AscendC::GetBlockNum();
    uint32_t blockIdx = (uint32_t)AscendC::GetBlockIdx();
    uint32_t coreLength = (totalLength + blockNum - 1) / blockNum;
    uint32_t offset = blockIdx * coreLength;
    if (offset + coreLength > totalLength) {
        coreLength = totalLength - offset;
    }

    if (coreLength > 0) {
        KernelTanh op;
        // 传递原始GM_ADDR和偏移量，避免显式类型转换
        op.Init(x, y, coreLength, tileNum, offset);
        op.Process();
    }
}