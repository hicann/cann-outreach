#include "kernel_operator.h"
#include "add_custom_template_tiling.h"

constexpr int32_t BUFFER_NUM = 1;

template <class dtypeX, class dtypeY, class dtypeZ>
class KernelAdd {
public:
    __aicore__ inline KernelAdd() {}

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR y,
        GM_ADDR z,
        uint64_t totalLength,
        uint64_t tileNum)
    {
        this->totalLength = totalLength;
        this->blockNum = AscendC::GetBlockNum();
        this->blockIdx = AscendC::GetBlockIdx();

        // ceil division，避免尾部数据丢失
        this->blockLength =
            (totalLength + blockNum - 1) / blockNum;

        uint64_t offset =
            static_cast<uint64_t>(blockIdx) * blockLength;

        // 当前block实际处理长度
        if (offset >= totalLength) {
            this->blockLength = 0;
            return;
        }

        if (offset + blockLength > totalLength) {
            this->blockLength = totalLength - offset;
        }

        this->tileNum = tileNum;

        this->tileLength =
            (blockLength + tileNum - 1) / tileNum;


        xGm.SetGlobalBuffer(
            (__gm__ dtypeX *)x + offset,
            blockLength);

        yGm.SetGlobalBuffer(
            (__gm__ dtypeY *)y + offset,
            blockLength);

        zGm.SetGlobalBuffer(
            (__gm__ dtypeZ *)z + offset,
            blockLength);


        pipe.InitBuffer(
            inQueueX,
            BUFFER_NUM,
            tileLength * sizeof(dtypeX));

        pipe.InitBuffer(
            inQueueY,
            BUFFER_NUM,
            tileLength * sizeof(dtypeY));

        pipe.InitBuffer(
            outQueueZ,
            BUFFER_NUM,
            tileLength * sizeof(dtypeZ));
    }


    __aicore__ inline void Process()
    {
        if (blockLength == 0) {
            return;
        }


        for (uint64_t i = 0; i < tileNum; i++) {

            uint64_t offset =
                i * tileLength;


            if (offset >= blockLength) {
                break;
            }


            uint64_t currentLength =
                tileLength;


            if (offset + currentLength > blockLength) {
                currentLength =
                    blockLength - offset;
            }


            CopyIn(i, currentLength);

            Compute(currentLength);

            CopyOut(i, currentLength);
        }
    }


private:

    __aicore__ inline void CopyIn(
        uint64_t progress,
        uint64_t length)
    {

        AscendC::LocalTensor<dtypeX> xLocal =
            inQueueX.AllocTensor<dtypeX>();

        AscendC::LocalTensor<dtypeY> yLocal =
            inQueueY.AllocTensor<dtypeY>();


        AscendC::DataCopy(
            xLocal,
            xGm[progress * tileLength],
            length);


        AscendC::DataCopy(
            yLocal,
            yGm[progress * tileLength],
            length);


        inQueueX.EnQue(xLocal);

        inQueueY.EnQue(yLocal);
    }


    __aicore__ inline void Compute(
        uint64_t length)
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
            length);


        outQueueZ.EnQue(zLocal);


        inQueueX.FreeTensor(xLocal);

        inQueueY.FreeTensor(yLocal);
    }


    __aicore__ inline void CopyOut(
        uint64_t progress,
        uint64_t length)
    {

        AscendC::LocalTensor<dtypeZ> zLocal =
            outQueueZ.DeQue<dtypeZ>();


        AscendC::DataCopy(
            zGm[progress * tileLength],
            zLocal,
            length);


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



    uint64_t totalLength;

    uint32_t blockNum;

    uint32_t blockIdx;


    uint64_t blockLength;

    uint64_t tileNum;

    uint64_t tileLength;
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