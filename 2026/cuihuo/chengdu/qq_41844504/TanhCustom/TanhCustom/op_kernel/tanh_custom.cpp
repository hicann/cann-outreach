#include "kernel_operator.h"
#include "tanh_custom_tiling.h"

// 【修改点1】定义数据类型别名，解决编译报错
using DTYPE_X = half;
using DTYPE_Y = half;

constexpr int32_t BUFFER_NUM = 2; // tensor num for each queue

class KernelTanh {
public:
    __aicore__ inline KernelTanh() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength, uint32_t tileNum)
    {
        this->blockLength = totalLength;
        this->tileNum = tileNum;
        // 【修改点2】向上取整，保证所有数据被分块处理
        this->tileLength = (totalLength + tileNum * BUFFER_NUM - 1) / (tileNum * BUFFER_NUM);

        xGm.SetGlobalBuffer((__gm__ DTYPE_X *)x + AscendC::GetBlockIdx() * this->blockLength,
                             this->blockLength);
        yGm.SetGlobalBuffer((__gm__ DTYPE_Y *)y + AscendC::GetBlockIdx() * this->blockLength,
                             this->blockLength);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(DTYPE_X));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileLength * sizeof(DTYPE_Y));
        pipe.InitBuffer(tmpBuf0, this->tileLength * sizeof(DTYPE_X));
        pipe.InitBuffer(tmpBuf1, this->tileLength * sizeof(DTYPE_X));
        pipe.InitBuffer(tmpBuf2, this->tileLength * sizeof(DTYPE_X));
    }
    __aicore__ inline void Process()
    {
        int32_t loopCount = this->tileNum * BUFFER_NUM;
        uint32_t processedCount = 0;  // 【新增】记录已处理数据量
        for (int32_t i = 0; i < loopCount; i++) {
            // 【修改点3】计算本次实际处理长度，处理余数
            uint32_t currentTileLength = this->tileLength;
            uint32_t remainData = this->blockLength - processedCount;
            if (remainData < currentTileLength) {
                currentTileLength = remainData;
            }
            if (currentTileLength == 0) {
                break;
            }
            CopyIn(i, currentTileLength);
            Compute(i, currentTileLength);
            CopyOut(i, currentTileLength);
            processedCount += currentTileLength;
        }
    }

private:
    // 【修改点4】CopyIn/Compute/CopyOut增加len参数，支持动态长度
    __aicore__ inline void CopyIn(int32_t progress, uint32_t len)
    {
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.AllocTensor<DTYPE_X>();
        AscendC::DataCopy(xLocal, xGm[progress * tileLength], len);
        inQueueX.EnQue<DTYPE_X>(xLocal);
    }
    __aicore__ inline void Compute(int32_t progress, uint32_t len)
    {
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.DeQue<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_Y> yLocal = outQueueY.AllocTensor<DTYPE_Y>();

        AscendC::LocalTensor<DTYPE_X> expX = tmpBuf0.Get<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_X> expNX = tmpBuf1.Get<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_X> tmp = tmpBuf2.Get<DTYPE_X>();

        AscendC::Muls(tmp, xLocal, (DTYPE_X)(-1.0f), len);
        AscendC::Exp(expX, xLocal, len);
        AscendC::Exp(expNX, tmp, len);
        AscendC::Sub(tmp, expX, expNX, len);
        AscendC::Add(expX, expX, expNX, len);
        AscendC::Div(yLocal, tmp, expX, len);

        outQueueY.EnQue<DTYPE_Y>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress, uint32_t len)
    {
        AscendC::LocalTensor<DTYPE_Y> yLocal = outQueueY.DeQue<DTYPE_Y>();
        AscendC::DataCopy(yGm[progress * tileLength], yLocal, len);
        outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmpBuf0,tmpBuf1,tmpBuf2;
    AscendC::GlobalTensor<DTYPE_X> xGm;
    AscendC::GlobalTensor<DTYPE_Y> yGm;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
};

extern "C" __global__ __aicore__ void tanh_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(TanhCustomTilingData);
    GET_TILING_DATA(tilingData, tiling);
    KernelTanh op;
    op.Init(x, y, tilingData.blockLength, tilingData.tileNum);
    op.Process();
}