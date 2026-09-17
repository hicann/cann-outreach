/*
 * add_custom 算子 - Ascend C 实现
 * 功能：对两个输入张量 x, y 执行逐元素加法，输出 z = x + y
 * 数据类型：float16 (half)
 * 数据格式：ND，二维 [N2, N1]
 */
#include "kernel_operator.h"

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2; // 双缓冲队列深度

class KernelAddCustom {
public:
    __aicore__ inline KernelAddCustom() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z,
                                uint32_t totalLength, uint32_t tileLength)
    {
        this->blockLength = totalLength / GetBlockNum();
        this->tileNum = blockLength / tileLength;
        this->tileLength = tileLength;
        this->lastTileLength = blockLength % tileLength; // 不足以整除时的尾块长度

        xGm.SetGlobalBuffer((__gm__ half *)x + GetBlockIdx() * this->blockLength, this->blockLength);
        yGm.SetGlobalBuffer((__gm__ half *)y + GetBlockIdx() * this->blockLength, this->blockLength);
        zGm.SetGlobalBuffer((__gm__ half *)z + GetBlockIdx() * this->blockLength, this->blockLength);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, tileLength * sizeof(half));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, tileLength * sizeof(half));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, tileLength * sizeof(half));
    }

    __aicore__ inline void Process()
    {
        int32_t loopCount = this->tileNum;
        // 处理整块
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i, this->tileLength);
            Compute(i, this->tileLength);
            CopyOut(i, this->tileLength);
        }
        // 处理尾块
        if (this->lastTileLength > 0) {
            CopyIn(loopCount, this->lastTileLength);
            Compute(loopCount, this->lastTileLength);
            CopyOut(loopCount, this->lastTileLength);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress, uint32_t length)
    {
        LocalTensor<half> xLocal = inQueueX.AllocTensor<half>();
        LocalTensor<half> yLocal = inQueueY.AllocTensor<half>();
        DataCopy(xLocal, xGm[progress * this->tileLength], length);
        DataCopy(yLocal, yGm[progress * this->tileLength], length);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }

    __aicore__ inline void Compute(int32_t progress, uint32_t length)
    {
        LocalTensor<half> xLocal = inQueueX.DeQue<half>();
        LocalTensor<half> yLocal = inQueueY.DeQue<half>();
        LocalTensor<half> zLocal = outQueueZ.AllocTensor<half>();

        Add(zLocal, xLocal, yLocal, length);

        outQueueZ.EnQue<half>(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress, uint32_t length)
    {
        LocalTensor<half> zLocal = outQueueZ.DeQue<half>();
        DataCopy(zGm[progress * this->tileLength], zLocal, length);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX, inQueueY;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueZ;
    GlobalTensor<half> xGm, yGm, zGm;

    uint32_t blockLength;    // 每个核处理的元素数
    uint32_t tileNum;        // 每核分块数
    uint32_t tileLength;     // 每块元素数
    uint32_t lastTileLength; // 尾块元素数
};

// 核函数入口：z = x + y
extern "C" __global__ __aicore__ void add_custom(GM_ADDR x, GM_ADDR y, GM_ADDR z,
                                                 GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);
    KernelAddCustom op;
    op.Init(x, y, z, tilingData.totalLength, tilingData.tileLength);
    op.Process();
}
