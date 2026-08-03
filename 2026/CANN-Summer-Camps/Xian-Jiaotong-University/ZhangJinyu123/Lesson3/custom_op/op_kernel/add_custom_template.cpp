#include "kernel_operator.h"
#include "add_custom_template_tiling.h"

namespace {
constexpr uint32_t DOUBLE_BUFFER = 2U;
}

template <typename TX, typename TY, typename TZ>
class AddCustomTemplateKernel {
public:
    __aicore__ inline AddCustomTemplateKernel() = default;

    __aicore__ inline void Init(GM_ADDR xAddr, GM_ADDR yAddr, GM_ADDR zAddr,
                                uint32_t totalLength, uint32_t tileNum)
    {
        const uint32_t blockCount = AscendC::GetBlockNum();
        const uint32_t blockIndex = AscendC::GetBlockIdx();
        const uint32_t averageLength = totalLength / blockCount;
        const uint32_t remainder = totalLength % blockCount;

        blockLength_ = averageLength + (blockIndex < remainder ? 1U : 0U);
        blockOffset_ = blockIndex * averageLength +
                       (blockIndex < remainder ? blockIndex : remainder);

        tileCount_ = tileNum * DOUBLE_BUFFER;
        if (tileCount_ == 0U) {
            tileCount_ = DOUBLE_BUFFER;
        }

        tileLength_ = (blockLength_ + tileCount_ - 1U) / tileCount_;

        xGm_.SetGlobalBuffer(
            (__gm__ TX *)xAddr + blockOffset_, blockLength_);
        yGm_.SetGlobalBuffer(
            (__gm__ TY *)yAddr + blockOffset_, blockLength_);
        zGm_.SetGlobalBuffer(
            (__gm__ TZ *)zAddr + blockOffset_, blockLength_);

        pipe_.InitBuffer(xQueue_, DOUBLE_BUFFER, tileLength_ * sizeof(TX));
        pipe_.InitBuffer(yQueue_, DOUBLE_BUFFER, tileLength_ * sizeof(TY));
        pipe_.InitBuffer(zQueue_, DOUBLE_BUFFER, tileLength_ * sizeof(TZ));
    }

    __aicore__ inline void Process()
    {
        for (uint32_t tileIndex = 0U; tileIndex < tileCount_; ++tileIndex) {
            const uint32_t offset = tileIndex * tileLength_;
            if (offset >= blockLength_) {
                break;
            }
            const uint32_t remain = blockLength_ - offset;
            const uint32_t validLength =
                remain < tileLength_ ? remain : tileLength_;

            CopyIn(offset, validLength);
            Compute(validLength);
            CopyOut(offset, validLength);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t offset, uint32_t length)
    {
        AscendC::LocalTensor<TX> xLocal = xQueue_.AllocTensor<TX>();
        AscendC::LocalTensor<TY> yLocal = yQueue_.AllocTensor<TY>();
        AscendC::DataCopy(xLocal, xGm_[offset], length);
        AscendC::DataCopy(yLocal, yGm_[offset], length);
        xQueue_.EnQue(xLocal);
        yQueue_.EnQue(yLocal);
    }

    __aicore__ inline void Compute(uint32_t length)
    {
        AscendC::LocalTensor<TX> xLocal = xQueue_.DeQue<TX>();
        AscendC::LocalTensor<TY> yLocal = yQueue_.DeQue<TY>();
        AscendC::LocalTensor<TZ> zLocal = zQueue_.AllocTensor<TZ>();
        AscendC::Add(zLocal, xLocal, yLocal, length);
        zQueue_.EnQue(zLocal);
        xQueue_.FreeTensor(xLocal);
        yQueue_.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOut(uint32_t offset, uint32_t length)
    {
        AscendC::LocalTensor<TZ> zLocal = zQueue_.DeQue<TZ>();
        AscendC::DataCopy(zGm_[offset], zLocal, length);
        zQueue_.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe pipe_;
    AscendC::TQue<AscendC::TPosition::VECIN, DOUBLE_BUFFER> xQueue_;
    AscendC::TQue<AscendC::TPosition::VECIN, DOUBLE_BUFFER> yQueue_;
    AscendC::TQue<AscendC::TPosition::VECOUT, DOUBLE_BUFFER> zQueue_;
    AscendC::GlobalTensor<TX> xGm_;
    AscendC::GlobalTensor<TY> yGm_;
    AscendC::GlobalTensor<TZ> zGm_;
    uint32_t blockLength_ = 0U;
    uint32_t blockOffset_ = 0U;
    uint32_t tileCount_ = 0U;
    uint32_t tileLength_ = 0U;
};

extern "C" __global__ __aicore__ void add_custom_template(
    GM_ADDR x, GM_ADDR y, GM_ADDR z,
    GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(AddCustomTemplateTilingData);
    GET_TILING_DATA_WITH_STRUCT(AddCustomTemplateTilingData, tilingData, tiling);
    AddCustomTemplateKernel<DTYPE_X, DTYPE_Y, DTYPE_Z> kernel;
    kernel.Init(x, y, z, tilingData.totalLength, tilingData.tileNum);
    kernel.Process();
}
