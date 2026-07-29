#include "kernel_operator.h"
#include "sub_custom_template_tiling.h"

constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t TILE_SIZE = 2048;

template<typename T>
class KernelSub {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z,
                                uint32_t totalLen, uint32_t offset, uint32_t localLen) {
        xGm.SetGlobalBuffer(reinterpret_cast<T*>(x) + offset, localLen);
        yGm.SetGlobalBuffer(reinterpret_cast<T*>(y) + offset, localLen);
        zGm.SetGlobalBuffer(reinterpret_cast<T*>(z) + offset, localLen);
        this->localLen = localLen;
        tileNum = (localLen + TILE_SIZE - 1) / TILE_SIZE;
        pipe.InitBuffer(inQueueX, TILE_SIZE * sizeof(T), BUFFER_NUM);
        pipe.InitBuffer(inQueueY, TILE_SIZE * sizeof(T), BUFFER_NUM);
        pipe.InitBuffer(outQueue, TILE_SIZE * sizeof(T), BUFFER_NUM);
    }

    __aicore__ inline void Process() {
        for (int32_t progress = 0; progress < tileNum; progress++) {
            CopyIn(progress);
            Compute(progress);
            CopyOut(progress);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress) {
        auto xLocal = inQueueX.AllocTensor<T>();
        auto yLocal = inQueueY.AllocTensor<T>();
        uint32_t offset = progress * TILE_SIZE;
        uint32_t copyLen = (progress == tileNum - 1) ? (localLen - offset) : TILE_SIZE;
        DataCopy(xLocal, xGm[offset], copyLen);
        DataCopy(yLocal, yGm[offset], copyLen);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }

    __aicore__ inline void Compute(int32_t progress) {
        auto xLocal = inQueueX.DeQue<T>();
        auto yLocal = inQueueY.DeQue<T>();
        auto zLocal = outQueue.AllocTensor<T>();
        uint32_t offset = progress * TILE_SIZE;
        uint32_t len = (progress == tileNum - 1) ? (localLen - offset) : TILE_SIZE;
        for (uint32_t i = 0; i < len; i++) {
            zLocal.SetValue(i, xLocal.GetValue(i) - yLocal.GetValue(i));
        }
        outQueue.EnQue(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress) {
        auto zLocal = outQueue.DeQue<T>();
        uint32_t offset = progress * TILE_SIZE;
        uint32_t len = (progress == tileNum - 1) ? (localLen - offset) : TILE_SIZE;
        DataCopy(zGm[offset], zLocal, len);
        outQueue.FreeTensor(zLocal);
    }

private:
    GlobalTensor<T> xGm, yGm, zGm;
    uint32_t localLen;
    uint32_t tileNum;
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX, inQueueY;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue;
};

extern "C" __global__ __aicore__ void sub_custom_template(GM_ADDR x, GM_ADDR y, GM_ADDR z,
                                                          GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(SubCustomTemplateTilingData);
    GET_TILING_DATA(tilingData, tiling);

    uint32_t totalLen = tilingData.size;
    uint32_t blockDim = 8;                     // 与主机端 TilingFunc 的 BlockDim 一致
    uint32_t blockId = GetBlockIdx();
    uint32_t perCoreSize = (totalLen + blockDim - 1) / blockDim;
    uint32_t start = blockId * perCoreSize;
    uint32_t end = (blockId == blockDim - 1) ? totalLen : (start + perCoreSize);
    uint32_t localLen = end - start;
    if (localLen == 0) return;

    // 默认使用 float，若需启用 float16 性能，可在编译时定义 ENABLE_FP16
#ifdef ENABLE_FP16
    using DataType = half;
#else
    using DataType = float;
#endif

    KernelSub<DataType> op;
    op.Init(x, y, z, totalLen, start, localLen);
    op.Process();
}