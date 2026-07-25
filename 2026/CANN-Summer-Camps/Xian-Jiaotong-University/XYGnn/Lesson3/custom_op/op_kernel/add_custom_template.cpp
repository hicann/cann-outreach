#include "kernel_operator.h"
#include "add_custom_template_tiling.h"

constexpr uint32_t BUFFER_NUM = 2;

template <class dtypeX, class dtypeY, class dtypeZ>
class KernelAdd {
public:
    __aicore__ inline KernelAdd()
    {
    }

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR y,
        GM_ADDR z,
        uint32_t totalLength,
        uint32_t tileNum)
    {
        const uint32_t blockNum = AscendC::GetBlockNum();
        const uint32_t blockIdx = AscendC::GetBlockIdx();

        const uint32_t baseBlockLength = totalLength / blockNum;
        const uint32_t remainder = totalLength % blockNum;

        this->blockLength =
            baseBlockLength + (blockIdx < remainder ? 1U : 0U);

        this->blockOffset =
            blockIdx * baseBlockLength +
            (blockIdx < remainder ? blockIdx : remainder);

        this->tileNum = tileNum;

        const uint32_t maxLoopCount = this->tileNum * BUFFER_NUM;
        const uint32_t maxBlockLength =
            (totalLength + blockNum - 1U) / blockNum;

        const uint32_t rawTileLength =
            (maxBlockLength + maxLoopCount - 1U) / maxLoopCount;

        constexpr uint32_t ALIGN_NUM = 32U / sizeof(dtypeX);

        this->tileLength =
            ((rawTileLength + ALIGN_NUM - 1U) / ALIGN_NUM) * ALIGN_NUM;

        xGm.SetGlobalBuffer(
            (__gm__ dtypeX *)x + this->blockOffset,
            this->blockLength);

        yGm.SetGlobalBuffer(
            (__gm__ dtypeY *)y + this->blockOffset,
            this->blockLength);

        zGm.SetGlobalBuffer(
            (__gm__ dtypeZ *)z + this->blockOffset,
            this->blockLength);

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
        const uint32_t loopCount =
            (this->blockLength + this->tileLength - 1U) /
            this->tileLength;

        for (uint32_t i = 0; i < loopCount; ++i) {
            const uint32_t offset = i * this->tileLength;
            const uint32_t remain = this->blockLength - offset;

            const uint32_t currentLength =
                remain < this->tileLength
                    ? remain
                    : this->tileLength;

            CopyIn(offset, currentLength);
            Compute(currentLength);
            CopyOut(offset, currentLength);
        }
    }

private:
    __aicore__ inline void CopyIn(
        uint32_t offset,
        uint32_t currentLength)
    {
        AscendC::LocalTensor<dtypeX> xLocal =
            inQueueX.AllocTensor<dtypeX>();

        AscendC::LocalTensor<dtypeY> yLocal =
            inQueueY.AllocTensor<dtypeY>();

        constexpr uint32_t ALIGN_NUM_X = 32U / sizeof(dtypeX);
        constexpr uint32_t ALIGN_NUM_Y = 32U / sizeof(dtypeY);

        if (currentLength % ALIGN_NUM_X == 0U) {
            AscendC::DataCopy(
                xLocal,
                xGm[offset],
                currentLength);
        } else {
            const uint32_t copyBytesX =
                static_cast<uint32_t>(
                    currentLength * sizeof(dtypeX));

            AscendC::DataCopyExtParams copyParamsX{
                1,
                copyBytesX,
                0,
                0,
                0
            };

            AscendC::DataCopyPadExtParams<dtypeX> padParamsX{
                true,
                0,
                0,
                static_cast<dtypeX>(0)
            };

            AscendC::DataCopyPad(
                xLocal,
                xGm[offset],
                copyParamsX,
                padParamsX);
        }

        if (currentLength % ALIGN_NUM_Y == 0U) {
            AscendC::DataCopy(
                yLocal,
                yGm[offset],
                currentLength);
        } else {
            const uint32_t copyBytesY =
                static_cast<uint32_t>(
                    currentLength * sizeof(dtypeY));

            AscendC::DataCopyExtParams copyParamsY{
                1,
                copyBytesY,
                0,
                0,
                0
            };

            AscendC::DataCopyPadExtParams<dtypeY> padParamsY{
                true,
                0,
                0,
                static_cast<dtypeY>(0)
            };

            AscendC::DataCopyPad(
                yLocal,
                yGm[offset],
                copyParamsY,
                padParamsY);
        }

        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }

    __aicore__ inline void Compute(
        uint32_t currentLength)
    {
        AscendC::LocalTensor<dtypeX> xLocal =
            inQueueX.DeQue<dtypeX>();

        AscendC::LocalTensor<dtypeY> yLocal =
            inQueueY.DeQue<dtypeY>();

        AscendC::LocalTensor<dtypeZ> zLocal =
            outQueueZ.AllocTensor<dtypeZ>();

        AscendC::Add(
            zLocal,
            xLocal,
            yLocal,
            currentLength);

        outQueueZ.EnQue(zLocal);

        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOut(
        uint32_t offset,
        uint32_t currentLength)
    {
        AscendC::LocalTensor<dtypeZ> zLocal =
            outQueueZ.DeQue<dtypeZ>();

        constexpr uint32_t ALIGN_NUM_Z = 32U / sizeof(dtypeZ);

        if (currentLength % ALIGN_NUM_Z == 0U) {
            AscendC::DataCopy(
                zGm[offset],
                zLocal,
                currentLength);
        } else {
            const uint32_t copyBytesZ =
                static_cast<uint32_t>(
                    currentLength * sizeof(dtypeZ));

            AscendC::DataCopyExtParams copyParamsZ{
                1,
                copyBytesZ,
                0,
                0,
                0
            };

            AscendC::DataCopyPad(
                zGm[offset],
                zLocal,
                copyParamsZ);
        }

        outQueueZ.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe pipe;

    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM>
        inQueueX;

    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM>
        inQueueY;

    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM>
        outQueueZ;

    AscendC::GlobalTensor<dtypeX> xGm;
    AscendC::GlobalTensor<dtypeY> yGm;
    AscendC::GlobalTensor<dtypeZ> zGm;

    uint32_t blockLength = 0;
    uint32_t blockOffset = 0;
    uint32_t tileNum = 0;
    uint32_t tileLength = 0;
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