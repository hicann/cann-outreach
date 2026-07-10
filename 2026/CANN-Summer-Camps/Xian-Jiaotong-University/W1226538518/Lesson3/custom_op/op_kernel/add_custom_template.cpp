#include "kernel_operator.h"
#include "add_custom_template_tiling.h"

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;  // 双Buffer乒乓，掩蔽DataCopy延迟

template <class dtypeX, class dtypeY, class dtypeZ>
class KernelAdd {
public:
    __aicore__ inline KernelAdd() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t totalLength, uint32_t tileNum)
    {
        this->blockLength = totalLength / AscendC::GetBlockNum();
        this->tileNum = tileNum;
        this->tileLength = this->blockLength / tileNum;

        xGm.SetGlobalBuffer((__gm__ dtypeX *)x + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        yGm.SetGlobalBuffer((__gm__ dtypeY *)y + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        zGm.SetGlobalBuffer((__gm__ dtypeZ *)z + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);

        // 双Buffer：BUFFER_NUM份tileLength的空间，用于乒乓操作
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(dtypeX));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, this->tileLength * sizeof(dtypeY));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileLength * sizeof(dtypeZ));
    }

    __aicore__ inline void Process()
    {
        // 总循环次数 = tileNum + BUFFER_NUM - 1，流水线填充+稳定+排空
        int32_t loopCount = this->tileNum + BUFFER_NUM - 1;
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        if (progress < this->tileNum) {
            LocalTensor<dtypeX> xLocal = inQueueX.AllocTensor<dtypeX>();
            LocalTensor<dtypeY> yLocal = inQueueY.AllocTensor<dtypeY>();
            DataCopy(xLocal, xGm[progress * this->tileLength], this->tileLength);
            DataCopy(yLocal, yGm[progress * this->tileLength], this->tileLength);
            inQueueX.EnQue(xLocal);
            inQueueY.EnQue(yLocal);
        }
    }

    __aicore__ inline void Compute(int32_t progress)
    {
        if (progress >= 1 && progress < this->tileNum + 1) {
            LocalTensor<dtypeX> xLocal = inQueueX.DeQue<dtypeX>();
            LocalTensor<dtypeY> yLocal = inQueueY.DeQue<dtypeY>();
            LocalTensor<dtypeZ> zLocal = outQueueZ.AllocTensor<dtypeZ>();
            Add(zLocal, xLocal, yLocal, this->tileLength);
            outQueueZ.EnQue<dtypeZ>(zLocal);
            inQueueX.FreeTensor(xLocal);
            inQueueY.FreeTensor(yLocal);
        }
    }

    __aicore__ inline void CopyOut(int32_t progress)
    {
        if (progress >= 2 && progress < this->tileNum + 2) {
            LocalTensor<dtypeZ> zLocal = outQueueZ.DeQue<dtypeZ>();
            DataCopy(zGm[(progress - 2) * this->tileLength], zLocal, this->tileLength);
            outQueueZ.FreeTensor(zLocal);
        }
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
};

__global__ __aicore__ void add_custom_template(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(AddCustomTemplateTilingData);
    GET_TILING_DATA_WITH_STRUCT(AddCustomTemplateTilingData, tiling_data, tiling);
    KernelAdd<DTYPE_X, DTYPE_Y, DTYPE_Z> op;
    op.Init(x, y, z, tiling_data.totalLength, tiling_data.tileNum);
    op.Process();
}
