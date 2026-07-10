#include "kernel_operator.h"
#include "sub_custom_template_tiling.h"

using namespace AscendC;

constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t USE_CORE_NUM = 8;
constexpr uint32_t TILE_NUM = 8;
constexpr uint32_t DT_FLOAT_VALUE = 1;

template <typename T>
class KernelSubCustomTemplate {
public:
    __aicore__ inline KernelSubCustomTemplate() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t totalLength)
    {
        const uint32_t loopCount = TILE_NUM * BUFFER_NUM;
        const uint32_t coreDataNum = (totalLength / USE_CORE_NUM) + ((totalLength % USE_CORE_NUM) != 0);
        this->blockOffset = coreDataNum * GetBlockIdx();
        this->blockLength = 0;
        if (this->blockOffset < totalLength) {
            const uint32_t remain = totalLength - this->blockOffset;
            this->blockLength = remain < coreDataNum ? remain : coreDataNum;
        }
        this->tileLength = this->blockLength == 0 ? 1 : (this->blockLength + loopCount - 1) / loopCount;

        xGm.SetGlobalBuffer((__gm__ T *)x + this->blockOffset, this->blockLength);
        yGm.SetGlobalBuffer((__gm__ T *)y + this->blockOffset, this->blockLength);
        zGm.SetGlobalBuffer((__gm__ T *)z + this->blockOffset, this->blockLength);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(T));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, this->tileLength * sizeof(T));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileLength * sizeof(T));
    }

    __aicore__ inline void Process()
    {
        const int32_t loopCount = TILE_NUM * BUFFER_NUM;
        for (int32_t i = 0; i < loopCount; ++i) {
            const uint32_t currentLength = GetCurrentLength(i);
            if (currentLength == 0) {
                break;
            }
            CopyIn(i, currentLength);
            Compute(currentLength);
            CopyOut(i, currentLength);
        }
    }

private:
    __aicore__ inline uint32_t GetCurrentLength(int32_t progress) const
    {
        const uint32_t offset = static_cast<uint32_t>(progress) * this->tileLength;
        if (offset >= this->blockLength) {
            return 0;
        }
        const uint32_t remain = this->blockLength - offset;
        return remain < this->tileLength ? remain : this->tileLength;
    }

    __aicore__ inline void CopyIn(int32_t progress, uint32_t currentLength)
    {
        LocalTensor<T> xLocal = inQueueX.template AllocTensor<T>();
        LocalTensor<T> yLocal = inQueueY.template AllocTensor<T>();
        DataCopy(xLocal, xGm[progress * this->tileLength], currentLength);
        DataCopy(yLocal, yGm[progress * this->tileLength], currentLength);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }

    __aicore__ inline void Compute(uint32_t currentLength)
    {
        LocalTensor<T> xLocal = inQueueX.template DeQue<T>();
        LocalTensor<T> yLocal = inQueueY.template DeQue<T>();
        LocalTensor<T> zLocal = outQueueZ.template AllocTensor<T>();
        Sub(zLocal, xLocal, yLocal, currentLength);
        outQueueZ.EnQue(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress, uint32_t currentLength)
    {
        LocalTensor<T> zLocal = outQueueZ.template DeQue<T>();
        DataCopy(zGm[progress * this->tileLength], zLocal, currentLength);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueY;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueZ;
    GlobalTensor<T> xGm;
    GlobalTensor<T> yGm;
    GlobalTensor<T> zGm;
    uint32_t blockOffset;
    uint32_t blockLength;
    uint32_t tileLength;
};

extern "C" __global__ __aicore__ void sub_custom_template(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(SubCustomTemplateTilingData);
    GET_TILING_DATA(tilingData, tiling);

    if (tilingData.dataType == DT_FLOAT_VALUE) {
        KernelSubCustomTemplate<float> op;
        op.Init(x, y, z, tilingData.size);
        op.Process();
    } else {
        KernelSubCustomTemplate<half> op;
        op.Init(x, y, z, tilingData.size);
        op.Process();
    }
}
