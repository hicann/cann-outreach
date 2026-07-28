/**
 * @file    add_custom_template.cpp  (op_kernel)
 * @brief   AddCustomTemplate 算子 Ascend C 核函数   z = x + y
 *
 * 输入: x, y  shape=[45, 20480], dtype=float
 * 输出: z     shape=[45, 20480], dtype=float
 *
 * 优化: 双缓冲 + 大 tile (tileNum=8) 流水线
 */

#include "kernel_operator.h"
#include "add_custom_template_tiling.h"

constexpr int32_t BUFFER_NUM = 2;  // 双缓冲

template <class dtypeX, class dtypeY, class dtypeZ>
class KernelAdd {
public:
    __aicore__ inline KernelAdd() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z,
                                uint32_t totalLength, uint32_t tileNum)
    {
        this->blockLength = totalLength / AscendC::GetBlockNum();
        this->tileNum     = tileNum;
        this->tileLength  = this->blockLength / tileNum;

        uint32_t offset = this->blockLength * AscendC::GetBlockIdx();

        xGm.SetGlobalBuffer((__gm__ dtypeX *)x + offset, this->blockLength);
        yGm.SetGlobalBuffer((__gm__ dtypeY *)y + offset, this->blockLength);
        zGm.SetGlobalBuffer((__gm__ dtypeZ *)z + offset, this->blockLength);

        // 双缓冲: 每个队列 2 个 buffer
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(dtypeX));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, this->tileLength * sizeof(dtypeY));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileLength * sizeof(dtypeZ));
    }

    // ============================================================
    //  双缓冲流水线:
    //    Prologue:  CopyIn(0)
    //    Body:      Compute(i), CopyOut(i)  ||  CopyIn(i+1)
    //    Epilogue:  Compute(last), CopyOut(last)
    // ============================================================
    __aicore__ inline void Process()
    {
        CopyIn(0);  // prologue

        for (int32_t i = 0; i < static_cast<int32_t>(tileNum) - 1; i++) {
            Compute(i);
            CopyOut(i);
            CopyIn(i + 1);   // 与上面的 Compute/CopyOut 流水重叠
        }

        Compute(tileNum - 1);   // epilogue
        CopyOut(tileNum - 1);
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        AscendC::LocalTensor<dtypeX> xLocal = inQueueX.AllocTensor<dtypeX>();
        AscendC::LocalTensor<dtypeY> yLocal = inQueueY.AllocTensor<dtypeY>();
        AscendC::DataCopy(xLocal, xGm[progress * tileLength], tileLength);
        AscendC::DataCopy(yLocal, yGm[progress * tileLength], tileLength);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }

    __aicore__ inline void Compute(int32_t progress)
    {
        AscendC::LocalTensor<dtypeX> xLocal = inQueueX.DeQue<dtypeX>();
        AscendC::LocalTensor<dtypeY> yLocal = inQueueY.DeQue<dtypeY>();
        AscendC::LocalTensor<dtypeZ> zLocal = outQueueZ.AllocTensor<dtypeZ>();
        AscendC::Add(zLocal, xLocal, yLocal, tileLength);
        outQueueZ.EnQue<dtypeZ>(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<dtypeZ> zLocal = outQueueZ.DeQue<dtypeZ>();
        AscendC::DataCopy(zGm[progress * tileLength], zLocal, tileLength);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX, inQueueY;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueZ;
    AscendC::GlobalTensor<dtypeX> xGm;
    AscendC::GlobalTensor<dtypeY> yGm;
    AscendC::GlobalTensor<dtypeZ> zGm;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
};

extern "C" __global__ __aicore__ void add_custom_template(
    GM_ADDR x, GM_ADDR y, GM_ADDR z,
    GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(AddCustomTemplateTilingData);
    GET_TILING_DATA_WITH_STRUCT(AddCustomTemplateTilingData, tiling_data, tiling);
    KernelAdd<DTYPE_X, DTYPE_Y, DTYPE_Z> op;
    op.Init(x, y, z, tiling_data.totalLength, tiling_data.tileNum);
    op.Process();
}
