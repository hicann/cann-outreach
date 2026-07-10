#include "kernel_operator.h"
#include "sub_custom_template_tiling.h"
 
using namespace AscendC;
 
constexpr int32_t BUFFER_NUM = 2;     // double buffer，提升流水并行度
constexpr int32_t TILE_LENGTH = 512;  // 每个 tile 处理的元素个数
 
class KernelSub {
public:
    __aicore__ inline KernelSub() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t totalSize)
    {
        ASSERT(GetBlockNum() != 0 && totalSize != 0);
        // 每个 block 负责处理的元素个数
        this->blockLength = totalSize / GetBlockNum();
        // 每个 block 内再分 tile 处理
        this->tileLength = TILE_LENGTH;
        this->tileNum = this->blockLength / this->tileLength;
 
        // 按当前 block 的偏移设置 GlobalTensor
        xGm.SetGlobalBuffer((__gm__ half *)x + this->blockLength * GetBlockIdx(), this->blockLength);
        yGm.SetGlobalBuffer((__gm__ half *)y + this->blockLength * GetBlockIdx(), this->blockLength);
        zGm.SetGlobalBuffer((__gm__ half *)z + this->blockLength * GetBlockIdx(), this->blockLength);
 
        // 初始化 UB 上的输入输出队列
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(half));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, this->tileLength * sizeof(half));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileLength * sizeof(half));
    }
 
    __aicore__ inline void Process()
    {
        for (int32_t i = 0; i < this->tileNum; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }
 
private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        // 从 GM 搬运数据到 UB
        LocalTensor<half> xLocal = inQueueX.AllocTensor<half>();
        LocalTensor<half> yLocal = inQueueY.AllocTensor<half>();
        DataCopy(xLocal, xGm[progress * this->tileLength], this->tileLength);
        DataCopy(yLocal, yGm[progress * this->tileLength], this->tileLength);
        inQueueX.EnQue<half>(xLocal);
        inQueueY.EnQue<half>(yLocal);
    }
 
    __aicore__ inline void Compute(int32_t progress)
    {
        // 从队列取出数据，执行减法计算 z = x - y
        LocalTensor<half> xLocal = inQueueX.DeQue<half>();
        LocalTensor<half> yLocal = inQueueY.DeQue<half>();
        LocalTensor<half> zLocal = outQueueZ.AllocTensor<half>();
        Sub(zLocal, xLocal, yLocal, this->tileLength);
        outQueueZ.EnQue<half>(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }
 
    __aicore__ inline void CopyOut(int32_t progress)
    {
        // 计算结果从 UB 搬回 GM
        LocalTensor<half> zLocal = outQueueZ.DeQue<half>();
        DataCopy(zGm[progress * this->tileLength], zLocal, this->tileLength);
        outQueueZ.FreeTensor(zLocal);
    }
 
private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX, inQueueY;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueZ;
    GlobalTensor<half> xGm;
    GlobalTensor<half> yGm;
    GlobalTensor<half> zGm;
    uint32_t blockLength = 0;
    uint32_t tileNum = 0;
    uint32_t tileLength = 0;
};
 
extern "C" __global__ __aicore__ void sub_custom_template(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(SubCustomTemplateTilingData);
    GET_TILING_DATA(tilingData, tiling);
    // TODO: user kernel impl
    KernelSub op;
    op.Init(x, y, z, tilingData.size);
    op.Process();
}