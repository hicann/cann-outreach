#include "kernel_operator.h"
#include "add_custom_template_tiling.h"

constexpr int32_t BUFFER_NUM = 2;  // 双缓冲: BUFFER_NUM 1→2 启用搬入/计算/搬出流水并行
constexpr int32_t ALIGN_SIZE = 16;

template <class dtypeX, class dtypeY, class dtypeZ>
class KernelAdd {
public:
    __aicore__ inline KernelAdd() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t totalLength, uint32_t tileNum)
    {
        this->blockLength = totalLength / AscendC::GetBlockNum();
        this->tileNum = tileNum;
        // ALIGN_SIZE 对齐优化内存访问 (ALIGN_DOWN), 剩余尾部单独处理
        this->tileLength = (this->blockLength / tileNum / ALIGN_SIZE) * ALIGN_SIZE;
        if (this->tileLength < ALIGN_SIZE) {
            this->tileLength = (this->blockLength < ALIGN_SIZE) ? this->blockLength : ALIGN_SIZE;
        }
        this->tailLength = this->blockLength - this->tileNum * this->tileLength;

        uint32_t offset = this->blockLength * AscendC::GetBlockIdx();
        xGm.SetGlobalBuffer((__gm__ dtypeX *)x + offset, this->blockLength);
        yGm.SetGlobalBuffer((__gm__ dtypeY *)y + offset, this->blockLength);
        zGm.SetGlobalBuffer((__gm__ dtypeZ *)z + offset, this->blockLength);

        // BUFFER_NUM=2 双缓冲: 搬入和计算可流水并行
        uint32_t bufSize = this->tileLength * sizeof(dtypeX);
        if (this->tailLength > 0 && this->tailLength * sizeof(dtypeX) > bufSize) {
            bufSize = this->tailLength * sizeof(dtypeX);
        }
        pipe.InitBuffer(inQueueX, BUFFER_NUM, bufSize);
        pipe.InitBuffer(inQueueY, BUFFER_NUM, bufSize);
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, bufSize);
    }

    __aicore__ inline void Process()
    {
        // 双缓冲流水线: CopyIn(i+1) 与 Compute(i)/CopyOut(i) 重叠执行
        int32_t loopCount = this->tileNum;
        if (loopCount > 0) {
            CopyIn(0, this->tileLength);
            for (int32_t i = 0; i < loopCount - 1; i++) {
                CopyIn(i + 1, this->tileLength);
                Compute(i, this->tileLength);
                CopyOut(i, this->tileLength);
            }
            Compute(loopCount - 1, this->tileLength);
            CopyOut(loopCount - 1, this->tileLength);
        }
        // 处理对齐截断遗留的尾部元素
        if (this->tailLength > 0) {
            TailCopyIn();
            TailCompute();
            TailCopyOut();
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress, uint32_t length)
    {
        AscendC::LocalTensor<dtypeX> xLocal = inQueueX.AllocTensor<dtypeX>();
        AscendC::LocalTensor<dtypeY> yLocal = inQueueY.AllocTensor<dtypeY>();
        AscendC::DataCopy(xLocal, xGm[progress * this->tileLength], length);
        AscendC::DataCopy(yLocal, yGm[progress * this->tileLength], length);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }

    __aicore__ inline void Compute(int32_t progress, uint32_t length)
    {
        AscendC::LocalTensor<dtypeX> xLocal = inQueueX.DeQue<dtypeX>();
        AscendC::LocalTensor<dtypeY> yLocal = inQueueY.DeQue<dtypeY>();
        AscendC::LocalTensor<dtypeZ> zLocal = outQueueZ.AllocTensor<dtypeZ>();
        AscendC::Add(zLocal, xLocal, yLocal, length);
        outQueueZ.EnQue<dtypeZ>(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress, uint32_t length)
    {
        AscendC::LocalTensor<dtypeZ> zLocal = outQueueZ.DeQue<dtypeZ>();
        AscendC::DataCopy(zGm[progress * this->tileLength], zLocal, length);
        outQueueZ.FreeTensor(zLocal);
    }

    __aicore__ inline void TailCopyIn()
    {
        uint32_t offset = this->tileNum * this->tileLength;
        AscendC::LocalTensor<dtypeX> xLocal = inQueueX.AllocTensor<dtypeX>();
        AscendC::LocalTensor<dtypeY> yLocal = inQueueY.AllocTensor<dtypeY>();
        AscendC::DataCopy(xLocal, xGm[offset], this->tailLength);
        AscendC::DataCopy(yLocal, yGm[offset], this->tailLength);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }

    __aicore__ inline void TailCompute()
    {
        AscendC::LocalTensor<dtypeX> xLocal = inQueueX.DeQue<dtypeX>();
        AscendC::LocalTensor<dtypeY> yLocal = inQueueY.DeQue<dtypeY>();
        AscendC::LocalTensor<dtypeZ> zLocal = outQueueZ.AllocTensor<dtypeZ>();
        AscendC::Add(zLocal, xLocal, yLocal, this->tailLength);
        outQueueZ.EnQue<dtypeZ>(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void TailCopyOut()
    {
        uint32_t offset = this->tileNum * this->tileLength;
        AscendC::LocalTensor<dtypeZ> zLocal = outQueueZ.DeQue<dtypeZ>();
        AscendC::DataCopy(zGm[offset], zLocal, this->tailLength);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueY;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueZ;
    AscendC::GlobalTensor<dtypeX> xGm;
    AscendC::GlobalTensor<dtypeY> yGm;
    AscendC::GlobalTensor<dtypeZ> zGm;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
    uint32_t tailLength;
};

extern "C" __global__ __aicore__ void add_custom_template(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(AddCustomTemplateTilingData);
    GET_TILING_DATA_WITH_STRUCT(AddCustomTemplateTilingData, tiling_data, tiling);
    KernelAdd<DTYPE_X, DTYPE_Y, DTYPE_Z> op;
    op.Init(x, y, z, tiling_data.totalLength, tiling_data.tileNum);
    op.Process();
}
