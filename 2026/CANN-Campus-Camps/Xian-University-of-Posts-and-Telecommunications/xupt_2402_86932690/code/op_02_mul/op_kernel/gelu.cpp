// Kernel 侧核函数实现
#include "kernel_operator.h"
#include <type_traits>
#include "gelu_tiling.h"
#include "tiling_key_gelu.h"

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t MAX_TILE_NUM = 4096;

template <typename T>
class KernelGelu {
public:
    __aicore__ inline KernelGelu() {}

    __aicore__ inline void Init(GM_ADDR input_x, GM_ADDR output, const GeluTilingData *td) {
        usedCoreNum = td->usedCoreNum;
        blockLength = td->blockLength;
        lastBlockLength = td->lastBlockLength;
        ubLength = td->ubLength;
        alignNum = td->alignNum;
        extraBlocks = td->extraBlocks;

        // 异常 tiling 保护
        if (blockLength <= 0 || ubLength <= 0) {
            return;
        }
        if (ubLength > MAX_TILE_NUM) {
            ubLength = MAX_TILE_NUM;
        }

        // 当前核的处理区间：[gmOffset, gmOffset+count)
        // 负载均衡：前 extraBlocks 个核各多处理 1 个对齐块，避免余数全部压在尾核上
        int64_t coreIdx = static_cast<int64_t>(GetBlockIdx());
        if (extraBlocks > 0) {
            if (coreIdx < extraBlocks) {
                count = blockLength + alignNum;
                gmOffset = coreIdx * (blockLength + alignNum);
            } else if (coreIdx == extraBlocks) {
                count = lastBlockLength; // blockLength + remTail（零头）
                gmOffset = extraBlocks * (blockLength + alignNum);
            } else {
                count = blockLength;
                gmOffset = coreIdx * blockLength + extraBlocks * alignNum;
            }
        } else {
            count = (coreIdx == usedCoreNum - 1) ? lastBlockLength : blockLength;
            gmOffset = coreIdx * blockLength;
        }

        xGm.SetGlobalBuffer((__gm__ T*)input_x + gmOffset, count);
        zGm.SetGlobalBuffer((__gm__ T*)output + gmOffset, count);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, static_cast<uint32_t>(ubLength * sizeof(T)));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, static_cast<uint32_t>(ubLength * sizeof(T)));
        // fp16 混合精度：x / 中间 / 结果各一块 fp32 缓冲（fp32 分支仅用其中两块）
        pipe.InitBuffer(xFp32, static_cast<uint32_t>(ubLength * sizeof(float)));
        pipe.InitBuffer(tmpFp32, static_cast<uint32_t>(ubLength * sizeof(float)));
        pipe.InitBuffer(zFp32, static_cast<uint32_t>(ubLength * sizeof(float)));
    }

    __aicore__ inline void Process() {
        if (count <= 0 || ubLength <= 0) {
            return;
        }

        // 双缓冲流水线：先预取首块，循环内预取下块并计算/输出当前块
        int64_t progress = 0;
        int64_t cur = ubLength < count ? ubLength : count;
        CopyIn(progress, cur);
        progress += cur;

        while (progress < count) {
            int64_t next = (count - progress) < ubLength ? (count - progress) : ubLength;
            CopyIn(progress, next);          // 预取下块（另一块缓冲）
            Compute(cur);                     // 计算当前块
            CopyOut(progress - cur, cur);     // 输出当前块
            cur = next;
            progress += next;
        }

        Compute(cur);
        CopyOut(count - cur, cur);
    }

private:
    __aicore__ inline void CopyIn(int64_t progress, int64_t currentNum) {
        LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
        DataCopy(xLocal, xGm[progress], static_cast<uint32_t>(currentNum));
        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(int64_t currentNum) {
        LocalTensor<T> xLocal = inQueueX.DeQue<T>();
        LocalTensor<T> zLocal = outQueueZ.AllocTensor<T>();

        const uint32_t count = static_cast<uint32_t>(currentNum);

        // GELU: y = 0.5 * x * (1 + erf(x / sqrt(2))),  1/sqrt(2) = 0.7071067811865475244
        if constexpr (std::is_same_v<T, half>) {
            // fp16 混合精度：全程 fp32 计算，避免 fp16 下 erf 饱和导致 1+erf 灾难性抵消
            LocalTensor<float> xF = xFp32.Get<float>();
            LocalTensor<float> tmpF = tmpFp32.Get<float>();
            LocalTensor<float> zF = zFp32.Get<float>();

            Cast(xF, xLocal, AscendC::RoundMode::CAST_NONE, count); // half -> float
            Muls(tmpF, xF, 0.7071067811865475244f, count); // tmp = x/sqrt(2)
            Erf<float, false>(zF, tmpF, count);            // z = erf(tmp)
            Adds(zF, zF, 1.0f, count);                     // z = 1 + erf
            Mul(zF, xF, zF, count);                        // z = x*(1+erf)
            Muls(zF, zF, 0.5f, count);                     // z = 0.5*x*(1+erf)
            Cast(zLocal, zF, AscendC::RoundMode::CAST_NONE, count); // float -> half
        } else {
            // float32 直接计算
            LocalTensor<float> tmpF = tmpFp32.Get<float>();
            Muls(tmpF, xLocal, 0.7071067811865475244f, count); // tmp = x/sqrt(2)
            Erf<float, false>(zLocal, tmpF, count);            // z = erf(tmp)
            Adds(zLocal, zLocal, 1.0f, count);                 // z = 1 + erf
            Mul(zLocal, xLocal, zLocal, count);                // z = x*(1+erf)
            Muls(zLocal, zLocal, 0.5f, count);                 // z = 0.5*x*(1+erf)
        }

        outQueueZ.EnQue(zLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(int64_t progress, int64_t currentNum) {
        LocalTensor<T> zLocal = outQueueZ.DeQue<T>();
        DataCopy(zGm[progress], zLocal, static_cast<uint32_t>(currentNum));
        outQueueZ.FreeTensor(zLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueZ;
    TBuf<QuePosition::VECCALC> xFp32;
    TBuf<QuePosition::VECCALC> tmpFp32;
    TBuf<QuePosition::VECCALC> zFp32;

    GlobalTensor<T> xGm;
    GlobalTensor<T> zGm;

    int64_t usedCoreNum;
    int64_t blockLength;
    int64_t lastBlockLength;
    int64_t ubLength;
    int64_t alignNum;    // 32B 对齐所需元素数（fp32=8, fp16=16）
    int64_t extraBlocks; // 负载均衡：前 extraBlocks 个核各多处理 1 个对齐块
    int64_t count;       // 当前核实际处理元素数
    int64_t gmOffset;    // 当前核在 GM 中的起始偏移
};

template <typename DT_X>
__global__ __aicore__ void gelu(GM_ADDR input_x, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(GeluTilingData);
    GET_TILING_DATA_WITH_STRUCT(GeluTilingData, tilingData, tiling);

    KernelGelu<DT_X> op;
    op.Init(input_x, output, &tilingData);
    op.Process();
}
