/*
 * add_custom 算子 — Ascend C 实现
 * 功能: z = x + y (逐元素加法)
 * 数据类型: float16 (half)
 * 数据格式: ND, 2 维 shape [N2, N1]
 *
 * 实现方式: 按一块固定长度 TILE_LENGTH 分块搬运到 Unified Buffer,
 * 在 vector 单元上完成 Add, 再拷回 GlobalMemory。
 */
#include "kernel_operator.h"

using namespace AscendC;

constexpr int32_t TILE_LENGTH = 2048; // 每次处理的元素个数(需为 16 的倍数, fp16 对齐)

class KernelAdd {
public:
    __aicore__ inline KernelAdd() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t totalLength)
    {
        this->totalLength = totalLength;
        // 2 维 shape [N2, N1] 在 ND 格式下按行主序平铺成一维处理
        xGm.SetGlobalBuffer((__gm__ half *)x, totalLength);
        yGm.SetGlobalBuffer((__gm__ half *)y, totalLength);
        zGm.SetGlobalBuffer((__gm__ half *)z, totalLength);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, TILE_LENGTH * sizeof(half));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, TILE_LENGTH * sizeof(half));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, TILE_LENGTH * sizeof(half));
    }

    __aicore__ inline void Process()
    {
        // 总块数向上取整; 最后一块做尾部对齐处理
        int32_t loopCount = totalLength / TILE_LENGTH;
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
        int32_t remain = totalLength % TILE_LENGTH;
        if (remain > 0) {
            CopyInTail(loopCount, remain);
            Compute(loopCount);
            CopyOutTail(loopCount, remain);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        LocalTensor<half> xLocal = inQueueX.AllocTensor<half>();
        LocalTensor<half> yLocal = inQueueY.AllocTensor<half>();
        DataCopy(xLocal, xGm[progress * TILE_LENGTH], TILE_LENGTH);
        DataCopy(yLocal, yGm[progress * TILE_LENGTH], TILE_LENGTH);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }

    __aicore__ inline void CopyInTail(int32_t progress, int32_t remain)
    {
        // 尾块按 32B 对齐搬运, 避免非对齐访问
        int32_t alignedRemain = (remain + FP16_ALIGN_NUM - 1) / FP16_ALIGN_NUM * FP16_ALIGN_NUM;
        LocalTensor<half> xLocal = inQueueX.AllocTensor<half>();
        LocalTensor<half> yLocal = inQueueY.AllocTensor<half>();
        DataCopy(xLocal, xGm[progress * TILE_LENGTH], alignedRemain);
        DataCopy(yLocal, yGm[progress * TILE_LENGTH], alignedRemain);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }

    __aicore__ inline void Compute(int32_t progress)
    {
        LocalTensor<half> xLocal = inQueueX.DeQue<half>();
        LocalTensor<half> yLocal = inQueueY.DeQue<half>();
        LocalTensor<half> zLocal = outQueueZ.AllocTensor<half>();

        // z = x + y
        Add(zLocal, xLocal, yLocal, TILE_LENGTH);

        outQueueZ.EnQue<half>(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress)
    {
        LocalTensor<half> zLocal = outQueueZ.DeQue<half>();
        DataCopy(zGm[progress * TILE_LENGTH], zLocal, TILE_LENGTH);
        outQueueZ.FreeTensor(zLocal);
    }

    __aicore__ inline void CopyOutTail(int32_t progress, int32_t remain)
    {
        int32_t alignedRemain = (remain + FP16_ALIGN_NUM - 1) / FP16_ALIGN_NUM * FP16_ALIGN_NUM;
        LocalTensor<half> zLocal = outQueueZ.DeQue<half>();
        DataCopy(zGm[progress * TILE_LENGTH], zLocal, alignedRemain);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    static constexpr int32_t FP16_ALIGN_NUM = 16; // fp16 32B 对齐所需元素数
    static constexpr int32_t BUFFER_NUM = 2;      // double buffer

    TPipe pipe;
    GlobalTensor<half> xGm, yGm, zGm;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX, inQueueY;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueZ;
    uint32_t totalLength = 0;
};

/*
 * 核函数入口
 * totalLength = N2 * N1, 由 host 侧传入
 */
extern "C" __global__ __aicore__ void add_custom(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t totalLength)
{
    KernelAdd op;
    op.Init(x, y, z, totalLength);
    if (ASCEND_IS_AICORE0) {
        op.Process();
    } else {
        // 单核处理全部数据; 如需多核可按 block 切分 totalLength
        op.Process();
    }
}
