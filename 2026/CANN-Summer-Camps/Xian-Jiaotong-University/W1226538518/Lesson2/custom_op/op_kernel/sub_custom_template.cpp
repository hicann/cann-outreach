#include "kernel_operator.h"
#include "sub_custom_template_tiling.h"

using namespace AscendC;

constexpr uint32_t BUFFER_NUM = 2;  // 双Buffer乒乓

template<typename T>
class KernelSub {
public:
    __aicore__ inline KernelSub() {}

    __aicore__ inline void Init(
        GM_ADDR x, GM_ADDR y, GM_ADDR z,
        uint32_t totalLength, uint32_t tileNum, uint32_t tileLength)
    {
        // 按 block_idx 分核处理
        uint32_t blockLen = totalLength / block_num;
        uint32_t blockOffset = block_idx * blockLen;

        xGm.SetGlobalBuffer((__gm__ T *)x + blockOffset, blockLen);
        yGm.SetGlobalBuffer((__gm__ T *)y + blockOffset, blockLen);
        zGm.SetGlobalBuffer((__gm__ T *)z + blockOffset, blockLen);

        this->tileNum = tileNum;
        this->tileLength = tileLength;
        this->blockLen = blockLen;

        // 分配UB空间：每个Buffer一个tile长度，共BUFFER_NUM块做乒乓
        pipe.InitBuffer(xBuf, BUFFER_NUM * tileLength * sizeof(T));
        pipe.InitBuffer(yBuf, BUFFER_NUM * tileLength * sizeof(T));
        pipe.InitBuffer(zBuf, BUFFER_NUM * tileLength * sizeof(T));
    }

    __aicore__ inline void Process()
    {
        // 三段流水线：CopyIn → Compute → CopyOut，双Buffer乒乓
        uint32_t localTileNum = blockLen / tileLength;
        for (uint32_t progress = 0; progress < localTileNum + BUFFER_NUM - 1; progress++) {
            CopyIn(progress, localTileNum);
            Compute(progress, localTileNum);
            CopyOut(progress, localTileNum);
        }
    }

private:
    // 数据搬运：GM → UB
    __aicore__ inline void CopyIn(uint32_t progress, uint32_t localTileNum)
    {
        if (progress < localTileNum) {
            uint32_t bufIdx = progress % BUFFER_NUM;
            uint32_t offset = progress * tileLength;

            LocalTensor<T> xLocal = xBuf[bufIdx * tileLength];
            LocalTensor<T> yLocal = yBuf[bufIdx * tileLength];

            DataCopy(xLocal, xGm[offset], tileLength);
            DataCopy(yLocal, yGm[offset], tileLength);
        }
    }

    // 矢量计算：z = x - y
    __aicore__ inline void Compute(uint32_t progress, uint32_t localTileNum)
    {
        if (progress >= 1 && progress < localTileNum + 1) {
            uint32_t bufIdx = (progress - 1) % BUFFER_NUM;

            LocalTensor<T> xLocal = xBuf[bufIdx * tileLength];
            LocalTensor<T> yLocal = yBuf[bufIdx * tileLength];
            LocalTensor<T> zLocal = zBuf[bufIdx * tileLength];

            // 矢量减法：zLocal = xLocal - yLocal
            Sub(zLocal, xLocal, yLocal, tileLength);
        }
    }

    // 数据搬出：UB → GM
    __aicore__ inline void CopyOut(uint32_t progress, uint32_t localTileNum)
    {
        if (progress >= 2 && progress < localTileNum + 2) {
            uint32_t bufIdx = (progress - 2) % BUFFER_NUM;
            uint32_t offset = (progress - 2) * tileLength;

            LocalTensor<T> zLocal = zBuf[bufIdx * tileLength];
            DataCopy(zGm[offset], zLocal, tileLength);
        }
    }

private:
    GlobalTensor<T> xGm;
    GlobalTensor<T> yGm;
    GlobalTensor<T> zGm;

    TPipe pipe;
    TBuf<QuePosition::VECIN> xBuf;
    TBuf<QuePosition::VECIN> yBuf;
    TBuf<QuePosition::VECOUT> zBuf;

    uint32_t tileNum;
    uint32_t tileLength;
    uint32_t blockLen;
};

// 核函数入口（registry invoke 模式的标准签名）
extern "C" __global__ __aicore__ void sub_custom_template(
    GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(SubCustomTemplateTilingData);
    GET_TILING_DATA(tilingData, tiling);

    // 根据数据类型实例化不同的kernel
    switch (GetDataType()) {
        case DT_FLOAT:
        {
            KernelSub<float> kernel;
            kernel.Init(x, y, z, tilingData.size, tilingData.tileNum, tilingData.tileLength);
            kernel.Process();
            break;
        }
        case DT_FLOAT16:
        {
            KernelSub<half> kernel;
            kernel.Init(x, y, z, tilingData.size, tilingData.tileNum, tilingData.tileLength);
            kernel.Process();
            break;
        }
        default:
            break;
    }
}
