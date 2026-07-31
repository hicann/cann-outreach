#include "kernel_operator.h"
#include "sub_custom_template_tiling.h"

#ifndef DTYPE_X
#define DTYPE_X half
#endif
#ifndef DTYPE_Y
#define DTYPE_Y half
#endif
#ifndef DTYPE_Z
#define DTYPE_Z half
#endif

constexpr int32_t BUFFER_NUM = 2; // tensor num for each queue (double buffer)

template <typename Tx, typename Ty, typename Tz>
class KernelSub {
public:
    __aicore__ inline KernelSub() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t totalLength, uint32_t tileNum)
    {
        // 每个核处理的数据量 = 总长度 / 核数
        this->blockLength = totalLength / AscendC::GetBlockNum();
        this->tileNum = tileNum;
        // 每个 tile 的长度:再按 tileNum 和 double buffer 切分
        this->tileLength = this->blockLength / tileNum / BUFFER_NUM;

        // 当前核在整个数据上的偏移
        uint32_t blockOffset = this->blockLength * AscendC::GetBlockIdx();

        xGm.SetGlobalBuffer((__gm__ Tx *)x + blockOffset, this->blockLength);
        yGm.SetGlobalBuffer((__gm__ Ty *)y + blockOffset, this->blockLength);
        zGm.SetGlobalBuffer((__gm__ Tz *)z + blockOffset, this->blockLength);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(Tx));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, this->tileLength * sizeof(Ty));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileLength * sizeof(Tz));
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
        AscendC::LocalTensor<Tx> xLocal = inQueueX.AllocTensor<Tx>();
        AscendC::LocalTensor<Ty> yLocal = inQueueY.AllocTensor<Ty>();

        AscendC::DataCopy(xLocal, xGm[progress * this->tileLength], this->tileLength);
        AscendC::DataCopy(yLocal, yGm[progress * this->tileLength], this->tileLength);

        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }

    __aicore__ inline void Compute(int32_t progress)
    {
        AscendC::LocalTensor<Tx> xLocal = inQueueX.DeQue<Tx>();
        AscendC::LocalTensor<Ty> yLocal = inQueueY.DeQue<Ty>();
        AscendC::LocalTensor<Tz> zLocal = outQueueZ.AllocTensor<Tz>();

        // z = x - y
        AscendC::Sub(zLocal, xLocal, yLocal, this->tileLength);

        outQueueZ.EnQue<Tz>(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<Tz> zLocal = outQueueZ.DeQue<Tz>();
        AscendC::DataCopy(zGm[progress * this->tileLength], zLocal, this->tileLength);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueY;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueZ;

    AscendC::GlobalTensor<Tx> xGm;
    AscendC::GlobalTensor<Ty> yGm;
    AscendC::GlobalTensor<Tz> zGm;

    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
};

extern "C" __global__ __aicore__ void sub_custom_template(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(SubCustomTemplateTilingData);
    GET_TILING_DATA(tilingData, tiling);

    KernelSub<DTYPE_X, DTYPE_Y, DTYPE_Z> op;
    constexpr uint32_t tileNum = 8;
    op.Init(x, y, z, tilingData.size, tileNum);
    op.Process();
}
