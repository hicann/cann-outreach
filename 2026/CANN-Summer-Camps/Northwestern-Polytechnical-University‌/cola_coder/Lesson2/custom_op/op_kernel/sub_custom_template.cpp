#include "kernel_operator.h"
#include "sub_custom_template_tiling.h"

constexpr uint32_t BUFFER_NUM = 2;

class KernelSubCustomTemplate {
public:
    __aicore__ inline KernelSubCustomTemplate() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t totalLength)
    {
        const uint32_t coreNum = AscendC::GetBlockNum();
        const uint32_t coreIdx = AscendC::GetBlockIdx();

        const uint32_t baseLen = totalLength / coreNum;
        const uint32_t extra = totalLength % coreNum;
        this->blockLen = baseLen + (coreIdx < extra ? 1U : 0U);
        this->blockOffset = coreIdx * baseLen + (coreIdx < extra ? coreIdx : extra);

        this->tileLen = this->blockLen == 0 ? 1 : (this->blockLen + BUFFER_NUM - 1) / BUFFER_NUM;
        this->loopCount = this->blockLen == 0 ? 0 : (this->blockLen + this->tileLen - 1) / this->tileLen;

        xGm.SetGlobalBuffer((__gm__ DTYPE_X *)x + this->blockOffset, this->blockLen);
        yGm.SetGlobalBuffer((__gm__ DTYPE_Y *)y + this->blockOffset, this->blockLen);
        zGm.SetGlobalBuffer((__gm__ DTYPE_Z *)z + this->blockOffset, this->blockLen);

        pipe.InitBuffer(xQueue, BUFFER_NUM, this->tileLen * sizeof(DTYPE_X));
        pipe.InitBuffer(yQueue, BUFFER_NUM, this->tileLen * sizeof(DTYPE_Y));
        pipe.InitBuffer(zQueue, BUFFER_NUM, this->tileLen * sizeof(DTYPE_Z));
    }

    __aicore__ inline void Process()
    {
        for (uint32_t i = 0; i < this->loopCount; ++i) {
            const uint32_t len = GetCurrentLen(i);
            CopyIn(i, len);
            Compute(len);
            CopyOut(i, len);
        }
    }

private:
    __aicore__ inline uint32_t GetCurrentLen(uint32_t tileIdx) const
    {
        const uint32_t offset = tileIdx * this->tileLen;
        const uint32_t remain = this->blockLen - offset;
        return remain < this->tileLen ? remain : this->tileLen;
    }

    __aicore__ inline void CopyIn(uint32_t tileIdx, uint32_t len)
    {
        AscendC::LocalTensor<DTYPE_X> xLocal = xQueue.AllocTensor<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_Y> yLocal = yQueue.AllocTensor<DTYPE_Y>();
        const uint32_t offset = tileIdx * this->tileLen;

        AscendC::DataCopy(xLocal, xGm[offset], len);
        AscendC::DataCopy(yLocal, yGm[offset], len);

        xQueue.EnQue(xLocal);
        yQueue.EnQue(yLocal);
    }

    __aicore__ inline void Compute(uint32_t len)
    {
        AscendC::LocalTensor<DTYPE_X> xLocal = xQueue.DeQue<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_Y> yLocal = yQueue.DeQue<DTYPE_Y>();
        AscendC::LocalTensor<DTYPE_Z> zLocal = zQueue.AllocTensor<DTYPE_Z>();

        AscendC::Sub(zLocal, xLocal, yLocal, len);

        zQueue.EnQue(zLocal);
        xQueue.FreeTensor(xLocal);
        yQueue.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOut(uint32_t tileIdx, uint32_t len)
    {
        AscendC::LocalTensor<DTYPE_Z> zLocal = zQueue.DeQue<DTYPE_Z>();
        AscendC::DataCopy(zGm[tileIdx * this->tileLen], zLocal, len);
        zQueue.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> xQueue;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> yQueue;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> zQueue;
    AscendC::GlobalTensor<DTYPE_X> xGm;
    AscendC::GlobalTensor<DTYPE_Y> yGm;
    AscendC::GlobalTensor<DTYPE_Z> zGm;
    uint32_t blockLen = 0;
    uint32_t blockOffset = 0;
    uint32_t tileLen = 0;
    uint32_t loopCount = 0;
};

extern "C" __global__ __aicore__ void sub_custom_template(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(SubCustomTemplateTilingData);
    GET_TILING_DATA(tilingData, tiling);
    (void)workspace;

    KernelSubCustomTemplate op;
    op.Init(x, y, z, tilingData.size);
    op.Process();
}
