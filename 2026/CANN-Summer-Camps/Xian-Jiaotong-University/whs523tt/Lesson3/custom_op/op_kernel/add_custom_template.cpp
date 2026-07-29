#include "kernel_operator.h"
#include "add_custom_template_tiling.h"

constexpr int32_t BUFFER_NUM = 2;

template <
    class dtypeX,
    class dtypeY,
    class dtypeZ>
class KernelAdd {
public:
    __aicore__ inline KernelAdd()
        : blockLength(0),
          tileNum(0),
          tileLength(0)
    {
    }

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR y,
        GM_ADDR z,
        uint32_t totalLength,
        uint32_t tileNum)
    {
        const uint32_t blockNum =
            AscendC::GetBlockNum();

        const uint32_t blockIdx =
            AscendC::GetBlockIdx();

        /*
         * Host 侧保证 totalLength 可以被 blockNum 整除。
         */
        this->blockLength =
            totalLength / blockNum;

        this->tileNum = tileNum;

        /*
         * tileLength 与 BUFFER_NUM 无关。
         *
         * BUFFER_NUM 仅表示 Queue 中有几块 Buffer，
         * 不表示数据还要再切 BUFFER_NUM 份。
         */
        this->tileLength =
            this->blockLength / this->tileNum;

        const uint32_t blockOffset =
            this->blockLength * blockIdx;

        xGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ dtypeX*>(x) +
                blockOffset,
            this->blockLength);

        yGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ dtypeY*>(y) +
                blockOffset,
            this->blockLength);

        zGm.SetGlobalBuffer(
            reinterpret_cast<__gm__ dtypeZ*>(z) +
                blockOffset,
            this->blockLength);

        /*
         * BUFFER_NUM=2 启用 Double Buffer。
         *
         * 三个 Queue 的 UB 占用为：
         *
         * 2 × tileLength ×
         * (sizeof(dtypeX) +
         *  sizeof(dtypeY) +
         *  sizeof(dtypeZ))
         */

        pipe.InitBuffer(
            inQueueX,
            BUFFER_NUM,
            static_cast<uint64_t>(this->tileLength) * sizeof(dtypeX));

        pipe.InitBuffer(
            inQueueY,
            BUFFER_NUM,
            static_cast<uint64_t>(this->tileLength) * sizeof(dtypeY));

        pipe.InitBuffer(
            outQueueZ,
            BUFFER_NUM,
            static_cast<uint64_t>(this->tileLength) * sizeof(dtypeZ));
    }

    __aicore__ inline void Process()
    {
        /*
         * tileNum 是实际 tile 数量。
         *
         * 不再使用：
         *
         *     tileNum * BUFFER_NUM
         *
         * Double Buffer 由 Queue 深度控制，
         * 不应改变总工作量。
         */
        for (uint32_t i = 0;
             i < this->tileNum;
             ++i) {
            CopyIn(i);
            Compute();
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(
        uint32_t progress)
    {
        AscendC::LocalTensor<dtypeX> xLocal =
            inQueueX.AllocTensor<dtypeX>();

        AscendC::LocalTensor<dtypeY> yLocal =
            inQueueY.AllocTensor<dtypeY>();

        const uint32_t offset =
            progress * this->tileLength;

        AscendC::DataCopy(
            xLocal,
            xGm[offset],
            this->tileLength);

        AscendC::DataCopy(
            yLocal,
            yGm[offset],
            this->tileLength);

        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }

    __aicore__ inline void Compute()
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
            this->tileLength);

        outQueueZ.EnQue(zLocal);

        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOut(
        uint32_t progress)
    {
        AscendC::LocalTensor<dtypeZ> zLocal =
            outQueueZ.DeQue<dtypeZ>();

        const uint32_t offset =
            progress * this->tileLength;

        AscendC::DataCopy(
            zGm[offset],
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

extern "C" __global__ __aicore__
void add_custom_template(
    GM_ADDR x,
    GM_ADDR y,
    GM_ADDR z,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(
        AddCustomTemplateTilingData);

    GET_TILING_DATA_WITH_STRUCT(
        AddCustomTemplateTilingData,
        tilingData,
        tiling);

    KernelAdd<
        DTYPE_X,
        DTYPE_Y,
        DTYPE_Z> op;

    op.Init(
        x,
        y,
        z,
        tilingData.totalLength,
        tilingData.tileNum);

    op.Process();
}