#include "kernel_operator.h"
#include "div_custom_template_tiling.h"

#ifndef DTYPE_X
#define DTYPE_X half
#endif

#ifndef DTYPE_Y
#define DTYPE_Y half
#endif

#ifndef DTYPE_Z
#define DTYPE_Z half
#endif

constexpr uint32_t BUFFER_NUM = 2;  // 每个队列的双缓冲数

class KernelDivCustomTemplate {
public:
    __aicore__ inline KernelDivCustomTemplate() {}
    // 初始化：设置GM地址（按核号分片）、计算每核tile长度、申请UB内存
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t totalLength, uint32_t tileNum)
    {
        // 每核处理数据长度
        this->blockLength = totalLength / AscendC::GetBlockNum();
        this->tileNum = tileNum;
        // 每tile元素数
        this->tileLength = this->blockLength / tileNum / BUFFER_NUM;

        // 按核号偏移各GM段的起始地址与长度，实现多核数据切分
        const uint32_t blockOffset = AscendC::GetBlockIdx() * this->blockLength;
        xGm.SetGlobalBuffer((__gm__ DTYPE_X *)x + blockOffset, this->blockLength);
        yGm.SetGlobalBuffer((__gm__ DTYPE_Y *)y + blockOffset, this->blockLength);
        zGm.SetGlobalBuffer((__gm__ DTYPE_Z *)z + blockOffset, this->blockLength);

        // 为输入输出队列分配UB内存
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(DTYPE_X));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, this->tileLength * sizeof(DTYPE_Y));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileLength * sizeof(DTYPE_Z));
    }

    // 按tile迭代执行 CopyIn -> Compute -> CopyOut
    __aicore__ inline void Process()
    {
        uint32_t loopCount = this->tileNum * BUFFER_NUM;
        for (uint32_t i = 0; i < loopCount; ++i) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    // 数据搬入：从GM搬运x、y的一个tile到UB，并入队
    __aicore__ inline void CopyIn(uint32_t progress)
    {
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.AllocTensor<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_Y> yLocal = inQueueY.AllocTensor<DTYPE_Y>();
        AscendC::DataCopy(xLocal, xGm[progress * this->tileLength], this->tileLength);
        AscendC::DataCopy(yLocal, yGm[progress * this->tileLength], this->tileLength);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }
    // zLocal[i] = xLocal[i] / yLocal[i]
    __aicore__ inline void Compute(uint32_t progress)
    {
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.DeQue<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_Y> yLocal = inQueueY.DeQue<DTYPE_Y>();
        AscendC::LocalTensor<DTYPE_Z> zLocal = outQueueZ.AllocTensor<DTYPE_Z>();
        AscendC::Div(zLocal, xLocal, yLocal, this->tileLength);
        outQueueZ.EnQue<DTYPE_Z>(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }
    // 数据搬出：出队结果，从UB搬回GM
    __aicore__ inline void CopyOut(uint32_t progress)
    {
        AscendC::LocalTensor<DTYPE_Z> zLocal = outQueueZ.DeQue<DTYPE_Z>();
        AscendC::DataCopy(zGm[progress * this->tileLength], zLocal, this->tileLength);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe pipe;                                              // UB内存管理对象
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX;    // 输入x队列（VECIN）
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueY;    // 输入y队列（VECIN）
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueZ;  // 输出z队列（VECOUT）
    AscendC::GlobalTensor<DTYPE_X> xGm;
    AscendC::GlobalTensor<DTYPE_Y> yGm;
    AscendC::GlobalTensor<DTYPE_Z> zGm;
    uint32_t blockLength;  // 每核元素数
    uint32_t tileNum;      // 每核tile数
    uint32_t tileLength;   // 每tile元素数
};

// 从GM取回tiling数据后初始化并执行
extern "C" __global__ __aicore__ void div_custom_template(GM_ADDR x, GM_ADDR y, GM_ADDR z,
    GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(DivCustomTemplateTilingData);
    GET_TILING_DATA(tilingData, tiling);

    KernelDivCustomTemplate op;
    op.Init(x, y, z, tilingData.totalLength, tilingData.tileNum);
    op.Process();
}
