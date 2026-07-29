#include "kernel_operator.h"
// 不再包含 tiling 头文件，结构体在文件内定义以避免冲突

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;      // 双缓冲深度
constexpr int32_t TILE_LENGTH = 512;   // 单 tile 元素个数（按 half 计数）

// Tiling 数据结构（与测试端一致）
struct SubCustomTemplateTilingData {
    uint32_t totalLength;  // 总元素个数
    uint32_t tileNum;      // tile 数量
};

class KernelSub {
public:
    __aicore__ inline KernelSub() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z,
                                uint32_t totalLength, uint32_t tileNum)
    {
        // 多核任务划分：每个核处理一段连续数据
        uint32_t cores = GetBlockNum();
        uint32_t lengthPerCore = (totalLength + cores - 1) / cores;
        uint32_t offset = GetBlockIdx() * lengthPerCore;
        if (offset + lengthPerCore > totalLength) {
            lengthPerCore = totalLength - offset;
        }

        blockLength = lengthPerCore;
        tileLength = TILE_LENGTH;
        this->tileNum = (blockLength + tileLength - 1) / tileLength;

        // 将全局内存映射到本核负责的 half 数据区间
        xGm.SetGlobalBuffer((__gm__ half*)x + offset, blockLength);
        yGm.SetGlobalBuffer((__gm__ half*)y + offset, blockLength);
        zGm.SetGlobalBuffer((__gm__ half*)z + offset, blockLength);

        // 初始化双缓冲队列（缓冲区大小以字节为单位）
        pipe.InitBuffer(inQueueX, BUFFER_NUM, tileLength * sizeof(half));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, tileLength * sizeof(half));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, tileLength * sizeof(half));
    }

    __aicore__ inline void Process()
    {
        for (int32_t i = 0; i < tileNum; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        LocalTensor<half> xLocal = inQueueX.AllocTensor<half>();
        LocalTensor<half> yLocal = inQueueY.AllocTensor<half>();

        DataCopy(xLocal, xGm[progress * tileLength], tileLength);
        DataCopy(yLocal, yGm[progress * tileLength], tileLength);

        inQueueX.EnQue<half>(xLocal);
        inQueueY.EnQue<half>(yLocal);
    }

    __aicore__ inline void Compute(int32_t progress)
    {
        LocalTensor<half> xLocal = inQueueX.DeQue<half>();
        LocalTensor<half> yLocal = inQueueY.DeQue<half>();
        LocalTensor<half> zLocal = outQueueZ.AllocTensor<half>();

        // 向量减法：z = x - y （支持 half 类型）
        Sub(zLocal, xLocal, yLocal, tileLength);

        outQueueZ.EnQue<half>(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress)
    {
        LocalTensor<half> zLocal = outQueueZ.DeQue<half>();
        DataCopy(zGm[progress * tileLength], zLocal, tileLength);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX, inQueueY;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueZ;

    GlobalTensor<half> xGm, yGm, zGm;
    uint32_t blockLength = 0;
    uint32_t tileNum = 0;
    uint32_t tileLength = 0;
};

extern "C" __global__ __aicore__ void sub_custom_template(
    GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(SubCustomTemplateTilingData);
    GET_TILING_DATA(tilingData, tiling);

    KernelSub op;
    op.Init(x, y, z, tilingData.totalLength, tilingData.tileNum);
    op.Process();
}