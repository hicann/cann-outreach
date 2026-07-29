#include "kernel_operator.h"
#include "add_custom_template_tiling.h"

// 双缓冲，提升流水线并行度
constexpr int32_t BUFFER_NUM = 2;

template <class dtypeX, class dtypeY, class dtypeZ>
class KernelAdd {
public:
    __aicore__ inline KernelAdd() : blockLength(0), tileNum(0), tileLength(0) {}

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR y,
        GM_ADDR z,
        uint32_t totalLength,
        uint32_t tileNum)
    {
        // 每个 AI Core 负责的数据长度（最后一个 Core 处理余数）
        this->blockLength = totalLength / AscendC::GetBlockNum();
        uint32_t remainder = totalLength % AscendC::GetBlockNum();
        if (AscendC::GetBlockIdx() == AscendC::GetBlockNum() - 1) {
            this->blockLength += remainder;
        }

        // 每个 AI Core 的基本切片数量
        this->tileNum = tileNum;

        // 双缓冲后每个 tile 的长度
        this->tileLength =
            this->blockLength / this->tileNum / BUFFER_NUM;

        // 定位当前 AI Core 负责的输入和输出数据
        xGm.SetGlobalBuffer(
            (__gm__ dtypeX *)x +
                this->blockLength * AscendC::GetBlockIdx(),
            this->blockLength);

        yGm.SetGlobalBuffer(
            (__gm__ dtypeY *)y +
                this->blockLength * AscendC::GetBlockIdx(),
            this->blockLength);

        zGm.SetGlobalBuffer(
            (__gm__ dtypeZ *)z +
                this->blockLength * AscendC::GetBlockIdx(),
            this->blockLength);

        // 为队列申请双缓冲空间
        pipe.InitBuffer(
            inQueueX,
            BUFFER_NUM,
            this->tileLength * sizeof(dtypeX));

        pipe.InitBuffer(
            inQueueY,
            BUFFER_NUM,
            this->tileLength * sizeof(dtypeY));

        pipe.InitBuffer(
            outQueueZ,
            BUFFER_NUM,
            this->tileLength * sizeof(dtypeZ));
    }

    __aicore__ inline void Process()
    {
        // 双缓冲后，实际处理次数为 tileNum × 2
        int32_t loopCount = this->tileNum * BUFFER_NUM;

        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        AscendC::LocalTensor<dtypeX> xLocal =
            inQueueX.AllocTensor<dtypeX>();

        AscendC::LocalTensor<dtypeY> yLocal =
            inQueueY.AllocTensor<dtypeY>();

        AscendC::DataCopy(
            xLocal,
            xGm[progress * this->tileLength],
            this->tileLength);

        AscendC::DataCopy(
            yLocal,
            yGm[progress * this->tileLength],
            this->tileLength);

        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }

    __aicore__ inline void Compute(int32_t progress)
    {
        AscendC::LocalTensor<dtypeX> xLocal =
            inQueueX.DeQue<dtypeX>();

        AscendC::LocalTensor<dtypeY> yLocal =
            inQueueY.DeQue<dtypeY>();

        AscendC::LocalTensor<dtypeZ> zLocal =
            outQueueZ.AllocTensor<dtypeZ>();

        // 执行向量加法 z = x + y
        AscendC::Add(
            zLocal,
            xLocal,
            yLocal,
            this->tileLength);

        outQueueZ.EnQue(zLocal);

        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<dtypeZ> zLocal =
            outQueueZ.DeQue<dtypeZ>();

        AscendC::DataCopy(
            zGm[progress * this->tileLength],
            zLocal,
            this->tileLength);

        outQueueZ.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe pipe;

    AscendC::TQue<
        AscendC::TPosition::VECIN,
        BUFFER_NUM> inQueueX;

    AscendC::TQue<
        AscendC::TPosition::VECIN,
        BUFFER_NUM> inQueueY;

    AscendC::TQue<
        AscendC::TPosition::VECOUT,
        BUFFER_NUM> outQueueZ;

    AscendC::GlobalTensor<dtypeX> xGm;
    AscendC::GlobalTensor<dtypeY> yGm;
    AscendC::GlobalTensor<dtypeZ> zGm;

    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
};

__global__ __aicore__ void add_custom_template(
    GM_ADDR x,
    GM_ADDR y,
    GM_ADDR z,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(AddCustomTemplateTilingData);

    GET_TILING_DATA_WITH_STRUCT(
        AddCustomTemplateTilingData,
        tiling_data,
        tiling);

    KernelAdd<DTYPE_X, DTYPE_Y, DTYPE_Z> op;

    op.Init(
        x,
        y,
        z,
        tiling_data.totalLength,
        tiling_data.tileNum);

    op.Process();
}
