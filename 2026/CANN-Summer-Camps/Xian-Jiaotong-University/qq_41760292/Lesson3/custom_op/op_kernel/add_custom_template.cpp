#include "kernel_operator.h"
#include "add_custom_template_tiling.h"

namespace {

    constexpr uint32_t BUFFER_NUM = 2;
    constexpr uint32_t TILE_LENGTH = 4096;

    __aicore__ inline uint32_t CeilDiv(
        uint32_t x,
        uint32_t y)
    {
        return (x + y - 1U) / y;
    }

}


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
        uint32_t totalLength)
    {

        const uint32_t blockNum =
            AscendC::GetBlockNum();

        const uint32_t blockIdx =
            AscendC::GetBlockIdx();


        /*
         * 均衡划分Block
         */
        const uint32_t baseLength =
            totalLength / blockNum;

        const uint32_t remain =
            totalLength % blockNum;


        const uint32_t extra =
            (blockIdx < remain) ? 1U : 0U;


        blockLength_ =
            baseLength + extra;


        if (blockLength_ == 0U) {

            tileNum_ = 0U;
            lastTileLength_ = 0U;
            return;
        }


        /*
         * 当前Block起始位置
         */
        const uint32_t blockOffset =
            blockIdx * baseLength +
            ((blockIdx < remain)
                ? blockIdx
                : remain);



        tileNum_ =
            CeilDiv(
                blockLength_,
                TILE_LENGTH);



        /*
         * 最后一个Tile长度
         */
        lastTileLength_ =
            blockLength_
            -
            (tileNum_ - 1U)
            * TILE_LENGTH;


        if (lastTileLength_ == 0U) {

            lastTileLength_ =
                TILE_LENGTH;
        }



        xGm_.SetGlobalBuffer(
            (__gm__ dtypeX*)x + blockOffset,
            blockLength_);


        yGm_.SetGlobalBuffer(
            (__gm__ dtypeY*)y + blockOffset,
            blockLength_);


        zGm_.SetGlobalBuffer(
            (__gm__ dtypeZ*)z + blockOffset,
            blockLength_);



        pipe_.InitBuffer(
            xQueue_,
            BUFFER_NUM,
            TILE_LENGTH * sizeof(dtypeX));


        pipe_.InitBuffer(
            yQueue_,
            BUFFER_NUM,
            TILE_LENGTH * sizeof(dtypeY));


        pipe_.InitBuffer(
            zQueue_,
            BUFFER_NUM,
            TILE_LENGTH * sizeof(dtypeZ));
    }



    __aicore__ inline void Process()
    {

        if (blockLength_ == 0U) {
            return;
        }


        uint32_t offset = 0U;


        for (uint32_t tileIdx = 0;
            tileIdx < tileNum_;
            ++tileIdx)
        {


            uint32_t currentLength =
                (tileIdx == tileNum_ - 1U)
                ?
                lastTileLength_
                :
                TILE_LENGTH;



            CopyIn(
                offset,
                currentLength);


            Compute(
                currentLength);


            CopyOut(
                offset,
                currentLength);



            offset += TILE_LENGTH;
        }

    }



private:


    __aicore__ inline void CopyIn(
        uint32_t offset,
        uint32_t length)
    {


        AscendC::LocalTensor<dtypeX>
            xLocal =
            xQueue_.AllocTensor<dtypeX>();


        AscendC::LocalTensor<dtypeY>
            yLocal =
            yQueue_.AllocTensor<dtypeY>();


        AscendC::DataCopy(
            xLocal,
            xGm_[offset],
            length);


        AscendC::DataCopy(
            yLocal,
            yGm_[offset],
            length);



        xQueue_.EnQue(xLocal);

        yQueue_.EnQue(yLocal);

    }




    __aicore__ inline void Compute(
        uint32_t length)
    {


        AscendC::LocalTensor<dtypeX>
            xLocal =
            xQueue_.DeQue<dtypeX>();


        AscendC::LocalTensor<dtypeY>
            yLocal =
            yQueue_.DeQue<dtypeY>();


        AscendC::LocalTensor<dtypeZ>
            zLocal =
            zQueue_.AllocTensor<dtypeZ>();



        AscendC::Add(
            zLocal,
            xLocal,
            yLocal,
            length);



        zQueue_.EnQue<dtypeZ>(
            zLocal);



        xQueue_.FreeTensor(
            xLocal);


        yQueue_.FreeTensor(
            yLocal);

    }




    __aicore__ inline void CopyOut(
        uint32_t offset,
        uint32_t length)
    {


        AscendC::LocalTensor<dtypeZ>
            zLocal =
            zQueue_.DeQue<dtypeZ>();


        AscendC::DataCopy(
            zGm_[offset],
            zLocal,
            length);



        zQueue_.FreeTensor(
            zLocal);

    }



private:


    AscendC::TPipe pipe_;


    AscendC::TQue<
        AscendC::TPosition::VECIN,
        BUFFER_NUM> xQueue_;


    AscendC::TQue<
        AscendC::TPosition::VECIN,
        BUFFER_NUM> yQueue_;


    AscendC::TQue<
        AscendC::TPosition::VECOUT,
        BUFFER_NUM> zQueue_;



    AscendC::GlobalTensor<dtypeX> xGm_;

    AscendC::GlobalTensor<dtypeY> yGm_;

    AscendC::GlobalTensor<dtypeZ> zGm_;



    uint32_t blockLength_ = 0U;

    uint32_t tileNum_ = 0U;

    uint32_t lastTileLength_ = 0U;

};




extern "C"
__global__
__aicore__
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
        tiling_data,
        tiling);



    KernelAdd<
        DTYPE_X,
        DTYPE_Y,
        DTYPE_Z> op;



    op.Init(
        x,
        y,
        z,
        tiling_data.totalLength);



    op.Process();

}