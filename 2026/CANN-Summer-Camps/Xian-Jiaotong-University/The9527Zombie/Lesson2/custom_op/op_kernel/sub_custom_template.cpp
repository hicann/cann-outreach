#include "kernel_operator.h"
#include "sub_custom_template_tiling.h"

using namespace AscendC;

// 类型无关的核函数模板, 编译期对 half/float 各生成一份实例
template <typename T>
__aicore__ inline void KernelSubImpl(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling,
                                      SubCustomTemplateTilingData& tData) {
    uint32_t blockIdx = GetBlockIdx();
    uint32_t offset = blockIdx * tData.blockLength;
    // 末 Block 可能不足 blockLength 个元素, 需裁剪
    uint32_t curLen = (blockIdx == 7) ? (tData.totalLength - offset) : tData.blockLength;
    if (curLen == 0) return;

    GlobalTensor<T> xGlobal, yGlobal, zGlobal;
    xGlobal.SetGlobalBuffer((__gm__ T*)x + offset, curLen);
    yGlobal.SetGlobalBuffer((__gm__ T*)y + offset, curLen);
    zGlobal.SetGlobalBuffer((__gm__ T*)z + offset, curLen);

    TPipe pipe;
    TQue<QuePosition::VECIN, 1> inQueueX, inQueueY;
    TQue<QuePosition::VECOUT, 1> outQueueZ;
    pipe.InitBuffer(inQueueX, 1, curLen * sizeof(T));
    pipe.InitBuffer(inQueueY, 1, curLen * sizeof(T));
    pipe.InitBuffer(outQueueZ, 1, curLen * sizeof(T));

    // CopyIn: GM -> VECIN Queue
    LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
    DataCopy(xLocal, xGlobal, curLen);
    inQueueX.EnQue(xLocal);

    LocalTensor<T> yLocal = inQueueY.AllocTensor<T>();
    DataCopy(yLocal, yGlobal, curLen);
    inQueueY.EnQue(yLocal);

    // Compute: z = x - y
    LocalTensor<T> xCompute = inQueueX.DeQue<T>();
    LocalTensor<T> yCompute = inQueueY.DeQue<T>();
    LocalTensor<T> zCompute = outQueueZ.AllocTensor<T>();
    Sub(zCompute, xCompute, yCompute, curLen);
    outQueueZ.EnQue<T>(zCompute);
    inQueueX.FreeTensor(xCompute);
    inQueueY.FreeTensor(yCompute);

    // CopyOut: VECOUT Queue -> GM
    LocalTensor<T> zLocal = outQueueZ.DeQue<T>();
    DataCopy(zGlobal, zLocal, curLen);
    outQueueZ.FreeTensor(zLocal);
}

extern "C" __global__ __aicore__ void sub_custom_template(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(SubCustomTemplateTilingData);
    GET_TILING_DATA(tilingData, tiling);

    // 根据 dataTypeSize 分发到 float 或 half 实例
    if (tilingData.dataTypeSize == 4) {
        KernelSubImpl<float>(x, y, z, workspace, tiling, tilingData);
    } else {
        KernelSubImpl<half>(x, y, z, workspace, tiling, tilingData);
    }
}