#include "kernel_operator.h"
#include "sub_custom_template_tiling.h"

// 若编译框架未定义输入数据类型宏，则默认使用 half（float16）
#ifndef DTYPE_X
#define DTYPE_X half
#endif

constexpr int32_t BUFFER_NUM = 2; // 双缓冲队列，实现搬入/计算/搬出流水并行

template <typename T>
class KernelSub {
public:
    __aicore__ inline KernelSub() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t totalLength, uint32_t tileNum)
    {
        uint32_t blockNum = AscendC::GetBlockNum();
        uint32_t blockIdx = AscendC::GetBlockIdx();
        // 每个核的基础数据长度；余数全部并入最后一个核，避免 totalLength 非整除时尾部数据被静默丢弃
        uint32_t baseBlockLength = totalLength / blockNum;
        uint32_t blockRemainder = totalLength % blockNum;
        this->blockLength = baseBlockLength + (blockIdx == blockNum - 1 ? blockRemainder : 0);
        // 各核起始偏移按基础长度计算，保证核间数据不重叠、不留空洞
        uint32_t blockOffset = baseBlockLength * blockIdx;

        // 单核内切分为 tileNum 块；核内余数并入最后一个 tile，避免核内尾块数据被丢弃
        this->tileNum = tileNum;
        this->tileLength = this->blockLength / tileNum;
        this->tailTileLength = this->blockLength - this->tileLength * (tileNum - 1);

        // 设置每个核的 Global Memory 起始地址
        xGm.SetGlobalBuffer((__gm__ T *)x + blockOffset, this->blockLength);
        yGm.SetGlobalBuffer((__gm__ T *)y + blockOffset, this->blockLength);
        zGm.SetGlobalBuffer((__gm__ T *)z + blockOffset, this->blockLength);

        // 队列按最大 tile 长度（tailTileLength >= tileLength）分配，确保尾块也放得下
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tailTileLength * sizeof(T));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, this->tailTileLength * sizeof(T));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tailTileLength * sizeof(T));
    }

    __aicore__ inline void Process()
    {
        // 流水线并行：依次处理每个数据块
        for (int32_t i = 0; i < this->tileNum; i++) {
            // 最后一个 tile 处理并入核内余数后的实际长度
            uint32_t curLength = (i == this->tileNum - 1) ? this->tailTileLength : this->tileLength;
            CopyIn(i, curLength);
            Compute(i, curLength);
            CopyOut(i, curLength);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress, uint32_t length)
    {
        // 为 LocalTensor 分配内存
        AscendC::LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
        AscendC::LocalTensor<T> yLocal = inQueueY.AllocTensor<T>();
        // 从 Global Memory 搬入当前数据块到 Local Memory
        AscendC::DataCopy(xLocal, xGm[progress * this->tileLength], length);
        AscendC::DataCopy(yLocal, yGm[progress * this->tileLength], length);
        // 入队，通知 Compute
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }

    __aicore__ inline void Compute(int32_t progress, uint32_t length)
    {
        // 从输入队列取数据
        AscendC::LocalTensor<T> xLocal = inQueueX.DeQue<T>();
        AscendC::LocalTensor<T> yLocal = inQueueY.DeQue<T>();
        AscendC::LocalTensor<T> zLocal = outQueueZ.AllocTensor<T>();
        // 执行矢量减法 z = x - y
        AscendC::Sub(zLocal, xLocal, yLocal, length);
        // 结果入队，释放输入
        outQueueZ.EnQue<T>(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress, uint32_t length)
    {
        // 从输出队列取结果并搬出到 Global Memory
        AscendC::LocalTensor<T> zLocal = outQueueZ.DeQue<T>();
        AscendC::DataCopy(zGm[progress * this->tileLength], zLocal, length);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX, inQueueY;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueZ;
    AscendC::GlobalTensor<T> xGm;
    AscendC::GlobalTensor<T> yGm;
    AscendC::GlobalTensor<T> zGm;
    uint32_t blockLength; // 每个核的计算数据长度
    uint32_t tileNum;     // 每个核需要计算的数据块个数
    uint32_t tileLength;  // 每个核内每个数据块的长度
    uint32_t tailTileLength; // 每个核内最后一个数据块的长度（含核内余数）
};

extern "C" __global__ __aicore__ void sub_custom_template(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(SubCustomTemplateTilingData);
    GET_TILING_DATA(tilingData, tiling);
    // 实例化算子（数据类型由编译框架的 DTYPE_X 宏决定，支持 float16/float32）
    KernelSub<DTYPE_X> op;
    uint32_t tileNum = 8;
    op.Init(x, y, z, tilingData.size, tileNum);
    op.Process();
}
