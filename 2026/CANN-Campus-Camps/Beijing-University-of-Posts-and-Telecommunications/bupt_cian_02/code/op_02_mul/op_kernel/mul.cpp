// Kernel侧核函数实现
#include "kernel_operator.h"

#include "mul_tiling.h"
#include "tiling_key_mul.h"

constexpr uint32_t BUFFER_NUM = 1;
constexpr uint32_t BLOCK_SIZE = 32;

template <class DT_X>
class KernelMul {
public:
    __aicore__ inline KernelMul()
    {
    }

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR y,
        GM_ADDR z,
        uint32_t totalLength,
        uint32_t blockLength,
        uint32_t tileLength)
    {
        this->totalLength = totalLength;
        this->blockLength = blockLength;
        this->tileLength = tileLength;

        uint32_t blockIdx =
            AscendC::GetBlockIdx();

        // 当前核负责的数据在整个Tensor中的起点
        this->blockOffset =
            blockIdx * blockLength;

        // 计算当前核真正需要处理的数据量
        // 最后一个核的数据量可能小于blockLength
        if (this->blockOffset >= totalLength) {
            this->currentBlockLength = 0;
        } else {
            uint32_t remainLength =
                totalLength - this->blockOffset;

            this->currentBlockLength =
                (remainLength < blockLength)
                    ? remainLength
                    : blockLength;
        }

        // 设置当前核对应的GM地址
        xGm.SetGlobalBuffer(
            (__gm__ DT_X *)x + this->blockOffset,
            this->currentBlockLength);

        yGm.SetGlobalBuffer(
            (__gm__ DT_X *)y + this->blockOffset,
            this->currentBlockLength);

        zGm.SetGlobalBuffer(
            (__gm__ DT_X *)z + this->blockOffset,
            this->currentBlockLength);

        // =====================================================
        // UB Buffer需要按照32 Byte对齐
        // =====================================================

        uint32_t typeSize =
            static_cast<uint32_t>(sizeof(DT_X));

        uint32_t alignNum =
            BLOCK_SIZE / typeSize;

        uint32_t tileBufferLength =
            ((this->tileLength + alignNum - 1U)
                / alignNum)
            * alignNum;

        uint32_t tileBufferBytes =
            tileBufferLength * typeSize;

        pipe.InitBuffer(
            inQueueX,
            BUFFER_NUM,
            tileBufferBytes);

        pipe.InitBuffer(
            inQueueY,
            BUFFER_NUM,
            tileBufferBytes);

        pipe.InitBuffer(
            outQueueZ,
            BUFFER_NUM,
            tileBufferBytes);
    }


    __aicore__ inline void Process()
    {
        if (this->currentBlockLength == 0) {
            return;
        }

        // 当前核内部按照tileLength继续切分
        for (uint32_t offset = 0;
             offset < this->currentBlockLength;
             offset += this->tileLength)
        {
            uint32_t remainLength =
                this->currentBlockLength - offset;

            uint32_t validLength =
                (remainLength < this->tileLength)
                    ? remainLength
                    : this->tileLength;

            CopyIn(offset, validLength);
            Compute(validLength);
            CopyOut(offset, validLength);
        }
    }


private:
    __aicore__ inline void CopyIn(
        uint32_t offset,
        uint32_t validLength)
    {
        AscendC::LocalTensor<DT_X> xLocal =
            inQueueX.AllocTensor<DT_X>();

        AscendC::LocalTensor<DT_X> yLocal =
            inQueueY.AllocTensor<DT_X>();

        // =====================================================
        // 注意：
        // sizeof(DT_X)返回size_t，
        // 必须先显式转换成uint32_t，
        // 否则DataCopyExtParams聚合初始化会触发narrowing。
        // =====================================================

        uint32_t typeSize =
            static_cast<uint32_t>(sizeof(DT_X));

        uint32_t copyBytes =
            validLength * typeSize;

        // 不使用聚合初始化，逐字段赋值，
        // 彻底避免C++11 narrowing问题。
        AscendC::DataCopyExtParams copyParams;

        copyParams.blockCount = 1;
        copyParams.blockLen = copyBytes;
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        copyParams.rsv = 0;

        // 默认：
        // isPad = false
        // leftPadding = 0
        // rightPadding = 0
        // paddingValue = 0
        AscendC::DataCopyPadExtParams<DT_X>
            padParams;

        // GM -> UB
        AscendC::DataCopyPad(
            xLocal,
            xGm[offset],
            copyParams,
            padParams);

        AscendC::DataCopyPad(
            yLocal,
            yGm[offset],
            copyParams,
            padParams);

        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }


    __aicore__ inline void Compute(
        uint32_t validLength)
    {
        AscendC::LocalTensor<DT_X> xLocal =
            inQueueX.DeQue<DT_X>();

        AscendC::LocalTensor<DT_X> yLocal =
            inQueueY.DeQue<DT_X>();

        AscendC::LocalTensor<DT_X> zLocal =
            outQueueZ.AllocTensor<DT_X>();

        // =====================================================
        // z = x * y
        // =====================================================

        AscendC::Mul(
            zLocal,
            xLocal,
            yLocal,
            validLength);

        outQueueZ.EnQue(zLocal);

        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }


    __aicore__ inline void CopyOut(
        uint32_t offset,
        uint32_t validLength)
    {
        AscendC::LocalTensor<DT_X> zLocal =
            outQueueZ.DeQue<DT_X>();

        uint32_t typeSize =
            static_cast<uint32_t>(sizeof(DT_X));

        uint32_t copyBytes =
            validLength * typeSize;

        // 同样逐字段赋值，避免narrowing
        AscendC::DataCopyExtParams copyParams;

        copyParams.blockCount = 1;
        copyParams.blockLen = copyBytes;
        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        copyParams.rsv = 0;

        // UB -> GM
        AscendC::DataCopyPad(
            zGm[offset],
            zLocal,
            copyParams);

        outQueueZ.FreeTensor(zLocal);
    }


private:
    AscendC::TPipe pipe;

    AscendC::TQue<
        AscendC::TPosition::VECIN,
        BUFFER_NUM>
        inQueueX;

    AscendC::TQue<
        AscendC::TPosition::VECIN,
        BUFFER_NUM>
        inQueueY;

    AscendC::TQue<
        AscendC::TPosition::VECOUT,
        BUFFER_NUM>
        outQueueZ;

    AscendC::GlobalTensor<DT_X> xGm;
    AscendC::GlobalTensor<DT_X> yGm;
    AscendC::GlobalTensor<DT_X> zGm;

    uint32_t totalLength;
    uint32_t blockLength;
    uint32_t currentBlockLength;
    uint32_t blockOffset;
    uint32_t tileLength;
};


template <typename DT_X>
__global__ __aicore__ void mul(
    GM_ADDR x,
    GM_ADDR y,
    GM_ADDR z,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(MulTilingData);

    GET_TILING_DATA_WITH_STRUCT(
        MulTilingData,
        tilingData,
        tiling);

    KernelMul<DT_X> op;

    op.Init(
        x,
        y,
        z,
        tilingData.totalLength,
        tilingData.blockLength,
        tilingData.tileLength);

    op.Process();
}