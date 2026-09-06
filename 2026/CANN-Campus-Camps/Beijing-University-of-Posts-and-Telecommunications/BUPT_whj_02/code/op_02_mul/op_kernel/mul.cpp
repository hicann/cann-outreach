#include "kernel_operator.h"

#include "mul_tiling.h"
#include "tiling_key_mul.h"

template <class DT_X>
class KernelMul {
public:
    __aicore__ inline KernelMul() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t length) {
        // 按 tile 均匀分核，前 extra 个核各多处理一个 tile。
        const uint32_t totalTiles = length / TILE_LENGTH + (length % TILE_LENGTH != 0);
        const uint32_t coreNum = AscendC::GetBlockNum();
        const uint32_t coreId = AscendC::GetBlockIdx();
        const uint32_t base = totalTiles / coreNum;
        const uint32_t extra = totalTiles % coreNum;
        const uint32_t coreTiles = base + (coreId < extra);
        const uint32_t firstTile = coreId * base + (coreId < extra ? coreId : extra);
        const uint64_t offset = static_cast<uint64_t>(firstTile) * TILE_LENGTH;
        this->length = 0;
        if (offset >= length) {
            return;
        }
        const uint64_t coreLength = static_cast<uint64_t>(coreTiles) * TILE_LENGTH;
        const uint64_t remaining = length - offset;
        this->length = static_cast<uint32_t>(remaining < coreLength ? remaining : coreLength);
        xGm.SetGlobalBuffer((__gm__ DT_X *)x + offset, this->length);
        yGm.SetGlobalBuffer((__gm__ DT_X *)y + offset, this->length);
        zGm.SetGlobalBuffer((__gm__ DT_X *)z + offset, this->length);

        // 双缓冲：FP32 时每核三个队列合计 24 KiB UB。
        pipe.InitBuffer(inQueueX, 2, TILE_LENGTH * sizeof(DT_X));
        pipe.InitBuffer(inQueueY, 2, TILE_LENGTH * sizeof(DT_X));
        pipe.InitBuffer(outQueueZ, 2, TILE_LENGTH * sizeof(DT_X));
    }

    __aicore__ inline void Process() {
        const uint32_t loopCount = length / TILE_LENGTH + (length % TILE_LENGTH != 0);
        for (int32_t i = 0; i < loopCount; ++i) {
            const uint32_t remaining = length - i * TILE_LENGTH;
            currentLength = remaining < TILE_LENGTH ? remaining : TILE_LENGTH;
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress) {
        AscendC::LocalTensor<DT_X> xLocal = inQueueX.AllocTensor<DT_X>();
        AscendC::LocalTensor<DT_X> yLocal = inQueueY.AllocTensor<DT_X>();
        const uint32_t offset = progress * TILE_LENGTH;

        // 按有效字节数搬运，最后不足 1024 个元素时也能处理。
        AscendC::DataCopyExtParams copyParams{
            1, static_cast<uint32_t>(currentLength * sizeof(DT_X)), 0, 0, 0};
        AscendC::DataCopyPadExtParams<DT_X> padParams{true, 0, 0, static_cast<DT_X>(0)};
        AscendC::DataCopyPad(xLocal, xGm[offset], copyParams, padParams);
        AscendC::DataCopyPad(yLocal, yGm[offset], copyParams, padParams);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }

    __aicore__ inline void Compute(int32_t progress) {
        AscendC::LocalTensor<DT_X> xLocal = inQueueX.DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> yLocal = inQueueY.DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> zLocal = outQueueZ.AllocTensor<DT_X>();
        AscendC::Mul(zLocal, xLocal, yLocal, currentLength);
        outQueueZ.EnQue(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress) {
        AscendC::LocalTensor<DT_X> zLocal = outQueueZ.DeQue<DT_X>();
        AscendC::DataCopyExtParams copyParams{
            1, static_cast<uint32_t>(currentLength * sizeof(DT_X)), 0, 0, 0};
        AscendC::DataCopyPad(zGm[progress * TILE_LENGTH], zLocal, copyParams);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    static constexpr uint32_t TILE_LENGTH = 1024;
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> inQueueX, inQueueY;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> outQueueZ;
    AscendC::GlobalTensor<DT_X> xGm, yGm, zGm;
    uint32_t length;
    uint32_t currentLength;
};

template <typename DT_X>
__global__ __aicore__ void mul(GM_ADDR x, GM_ADDR y, GM_ADDR z,
                             GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(MulTilingData);
    GET_TILING_DATA_WITH_STRUCT(MulTilingData, tiling_data, tiling);
    KernelMul<DT_X> op;
    op.Init(x, y, z, tiling_data.length);
    op.Process();
}
