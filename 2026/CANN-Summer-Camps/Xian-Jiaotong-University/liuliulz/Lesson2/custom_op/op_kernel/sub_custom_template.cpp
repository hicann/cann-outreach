/**
 * @file    sub_custom_template.cpp  (op_kernel)
 * @brief   SubCustomTemplate 算子 Ascend C 核函数
 *          功能: z = x - y  (逐元素矢量减法)
 *
 * Shape: (8, 2048), Format: ND
 * Tiling: Host 侧 SetBlockDim(8), 8 个 AI Core 每个处理 2048 个元素
 */

#include "kernel_operator.h"
#include "sub_custom_template_tiling.h"

using namespace AscendC;

// ================================================================
//  Kernel 算子类 (模板: T = half | float)
// ================================================================
template <typename T>
class KernelSubCustomTemplate {
public:
    __aicore__ inline KernelSubCustomTemplate() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z,
                                uint32_t offset, uint32_t curLength)
    {
        curLength_ = curLength;

        xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(x) + offset, curLength_);
        yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(y) + offset, curLength_);
        zGm_.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(z) + offset, curLength_);

        pipe_.InitBuffer(inQueX_,  1, curLength_ * sizeof(T));
        pipe_.InitBuffer(inQueY_,  1, curLength_ * sizeof(T));
        pipe_.InitBuffer(outQueZ_, 1, curLength_ * sizeof(T));
    }

    __aicore__ inline void Process()
    {
        CopyIn();
        Compute();
        CopyOut();
    }

private:
    // ---------- CopyIn: GM → UB ----------
    __aicore__ inline void CopyIn()
    {
        LocalTensor<T> xLocal = inQueX_.AllocTensor<T>();
        LocalTensor<T> yLocal = inQueY_.AllocTensor<T>();

        DataCopy(xLocal, xGm_, curLength_);
        DataCopy(yLocal, yGm_, curLength_);

        inQueX_.EnQue(xLocal);
        inQueY_.EnQue(yLocal);
    }

    // ---------- Compute: z = x - y  on UB ----------
    __aicore__ inline void Compute()
    {
        LocalTensor<T> xLocal = inQueX_.DeQue<T>();
        LocalTensor<T> yLocal = inQueY_.DeQue<T>();
        LocalTensor<T> zLocal = outQueZ_.AllocTensor<T>();

        Sub(zLocal, xLocal, yLocal, curLength_);

        outQueZ_.EnQue<T>(zLocal);
        inQueX_.FreeTensor(xLocal);
        inQueY_.FreeTensor(yLocal);
    }

    // ---------- CopyOut: UB → GM ----------
    __aicore__ inline void CopyOut()
    {
        LocalTensor<T> zLocal = outQueZ_.DeQue<T>();
        DataCopy(zGm_, zLocal, curLength_);
        outQueZ_.FreeTensor(zLocal);
    }

private:
    TPipe pipe_;
    TQue<TPosition::VECIN,  1> inQueX_, inQueY_;
    TQue<TPosition::VECOUT, 1> outQueZ_;
    GlobalTensor<T> xGm_, yGm_, zGm_;
    uint32_t curLength_;
};

// ================================================================
//  核函数入口
//    - REGISTER_TILING_DEFAULT / GET_TILING_DATA 是 msopgen 标准宏
//    - Host SetBlockDim(8) → blockDim = 8, 每 core 2048 元素
//    - dtypeFlag 0→half(float16)  1→float
// ================================================================
extern "C" __global__ __aicore__ void sub_custom_template(
    GM_ADDR x, GM_ADDR y, GM_ADDR z,
    GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(SubCustomTemplateTilingData);
    GET_TILING_DATA(tilingData, tiling);

    uint32_t blockIdx    = GetBlockIdx();
    uint32_t blockNum    = static_cast<uint32_t>(GetBlockNum());  // 8
    uint32_t totalLength = tilingData.size;                       // 16384
    uint32_t blockLength = totalLength / blockNum;                // 2048
    uint32_t offset      = blockIdx * blockLength;

    // 末尾 block 处理剩余元素（整除时 = blockLength）
    uint32_t curLength   = (blockIdx == blockNum - 1)
                           ? (totalLength - offset)              // tail
                           : blockLength;

    if (tilingData.dtypeFlag == 0) {
        KernelSubCustomTemplate<half> op;
        op.Init(x, y, z, offset, curLength);
        op.Process();
    } else {
        KernelSubCustomTemplate<float> op;
        op.Init(x, y, z, offset, curLength);
        op.Process();
    }
}
