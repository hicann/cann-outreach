/**
 * @file atanh_kernel.cpp
 * @brief Atanh 算子 Kernel 实现（AscendC，AIV 矢量编程）
 *
 * 功能：y = atanh(x) = 0.5 * ln((1+x)/(1-x))，定义域 x ∈ (-1, 1)，
 *       超出定义域输入得到 NaN（由数学接口自然处理）。
 *
 * 数据流：Global Memory x
 *           -> 多核切分（blockDim 个核均分，余数归最后一个核）
 *           -> 每核内按 tile 双缓冲流水：DataCopy(搬入) -> Atanh(计算) -> DataCopy(搬出)
 *           -> 尾块（不足 32B 对齐）用 DataCopyPad 非对齐搬运
 *           -> Global Memory y
 *
 * 对齐说明：
 *   - DataCopy 搬运量要求为 32B 的整数倍（half 即 16 的倍数），
 *     因此主循环 tileLength 取 2048（16 的倍数）；
 *   - 尾块长度不定，使用支持非对齐搬运的 DataCopyPad；
 *   - 矢量计算要求操作数起始地址 32B 对齐（UB 分配器自动保证）。
 */
#include "kernel_operator.h"
#include "lib/math/math.hpp"
#include "atanh_tiling.h"

using namespace AscendC;

namespace {
constexpr int32_t BUFFER_NUM = 2;       // 双缓冲：搬入/计算/搬出可流水重叠
constexpr int32_t TILE_LENGTH = 2048;   // 主循环每次处理 2048 个 half（4KB）
constexpr int32_t TAIL_ALIGN = 16;      // half 的 32B 对齐单位（32B / 2B = 16 个元素）
}  // namespace

class KernelAtanh {
public:
    __aicore__ inline KernelAtanh() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength,
                                uint32_t blockDim) {
        uint32_t coreIdx = GetBlockIdx();

        // ---------- 多核切分：均分，余数全部归最后一个核 ----------
        // 例：totalLength=10000, blockDim=4 -> 每核 2500，核0~2 各 2500，核3 2500
        //     实际上 avgLength=2500，核3 处理 totalLength - 2500*3 = 2500
        uint32_t avgLength = totalLength / blockDim;
        if (coreIdx == blockDim - 1) {
            this->blockLength = totalLength - avgLength * (blockDim - 1);
        } else {
            this->blockLength = avgLength;
        }
        this->blockOffset = coreIdx * avgLength;

        // ---------- 绑定全局内存 ----------
        // SetGlobalBuffer 的第二个参数多给 TAIL_ALIGN 个元素，
        // 用于 Debug 模式下的越界检查冗余（实际搬运不会超过 blockLength）。
        xGm.SetGlobalBuffer((__gm__ half*)x + this->blockOffset,
                            this->blockLength + TAIL_ALIGN);
        yGm.SetGlobalBuffer((__gm__ half*)y + this->blockOffset,
                            this->blockLength + TAIL_ALIGN);

        // ---------- 申请 UB 缓冲（双缓冲 x2） ----------
        // 每份多分配 TAIL_ALIGN 个元素，供尾块对齐计算使用
        uint32_t bufLength = (TILE_LENGTH + TAIL_ALIGN) * sizeof(half);
        pipe.InitBuffer(inQueue, BUFFER_NUM, bufLength);
        pipe.InitBuffer(outQueue, BUFFER_NUM, bufLength);
    }

    __aicore__ inline void Process() {
        // ---------- 主循环：整 tile（32B 对齐，用 DataCopy） ----------
        uint32_t loopCount = this->blockLength / TILE_LENGTH;
        for (uint32_t i = 0; i < loopCount; i++) {
            CopyIn(i * TILE_LENGTH, TILE_LENGTH);
            Compute(TILE_LENGTH);
            CopyOut(i * TILE_LENGTH, TILE_LENGTH);
        }

        // ---------- 尾块：长度不足 32B 对齐，用 DataCopyPad ----------
        uint32_t tailLength = this->blockLength % TILE_LENGTH;
        if (tailLength > 0) {
            // 计算长度向上对齐到 16 的倍数（UB 已多分配，多算的元素丢弃）
            uint32_t calCount = (tailLength + TAIL_ALIGN - 1) / TAIL_ALIGN * TAIL_ALIGN;
            CopyInPad(loopCount * TILE_LENGTH, tailLength);
            Compute(calCount);
            CopyOutPad(loopCount * TILE_LENGTH, tailLength);
        }
    }

private:
    // ---------- 整块搬入：GM -> UB ----------
    __aicore__ inline void CopyIn(uint32_t offset, uint32_t length) {
        LocalTensor<half> xLocal = inQueue.AllocTensor<half>();
        DataCopy(xLocal, xGm[offset], length);
        inQueue.EnQue(xLocal);
    }

    // ---------- 计算：y = atanh(x) ----------
    __aicore__ inline void Compute(uint32_t length) {
        LocalTensor<half> xLocal = inQueue.DeQue<half>();
        LocalTensor<half> yLocal = outQueue.AllocTensor<half>();

        // 反双曲正切：atanh(x) = 0.5 * ln((1+x)/(1-x))，定义域 (-1, 1)
        // 优先使用 AscendC 数学库接口（half 支持）。
        // 若当前 CANN 版本无 Atanh 接口，可改用下方注释的公式组合实现：
        //   Muls(tmp, xLocal, -1.0f, length);   // tmp = -x
        //   Adds(tmp, tmp, 1.0f, length);       // tmp = 1-x
        //   Adds(xLocal, xLocal, 1.0f, length); // xLocal = 1+x
        //   Div(tmp, xLocal, tmp, length);      // tmp = (1+x)/(1-x)
        //   Log(tmp, tmp, length);              // tmp = ln(...)
        //   Muls(yLocal, tmp, 0.5f, length);    // y = 0.5*ln(...)
        // （公式实现需额外申请一个临时 LocalTensor<half> tmp）
        Atanh(yLocal, xLocal, length);

        outQueue.EnQue(yLocal);
        inQueue.FreeTensor(xLocal);
    }

    // ---------- 整块搬出：UB -> GM ----------
    __aicore__ inline void CopyOut(uint32_t offset, uint32_t length) {
        LocalTensor<half> yLocal = outQueue.DeQue<half>();
        DataCopy(yGm[offset], yLocal, length);
        outQueue.FreeTensor(yLocal);
    }

    // ---------- 尾块非对齐搬入：GM -> UB ----------
    __aicore__ inline void CopyInPad(uint32_t offset, uint32_t length) {
        LocalTensor<half> xLocal = inQueue.AllocTensor<half>();
        DataCopyParams param;
        param.blockCount = 1;                              // 连续数据块个数
        param.blockLen = length * sizeof(half);            // 每块长度，单位 Byte，支持非 32B 对齐
        param.srcStride = 0;                               // GM 侧间隔，单位 Byte
        param.dstStride = 0;                               // UB 侧间隔，单位 32B dataBlock
        DataCopyPad(xLocal, xGm[offset], param);
        inQueue.EnQue(xLocal);
    }

    // ---------- 尾块非对齐搬出：UB -> GM ----------
    __aicore__ inline void CopyOutPad(uint32_t offset, uint32_t length) {
        LocalTensor<half> yLocal = outQueue.DeQue<half>();
        DataCopyParams param;
        param.blockCount = 1;
        param.blockLen = length * sizeof(half);
        param.srcStride = 0;
        param.dstStride = 0;
        DataCopyPad(yGm[offset], yLocal, param);
        outQueue.FreeTensor(yLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueue;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueue;
    GlobalTensor<half> xGm;
    GlobalTensor<half> yGm;
    uint32_t blockLength;   // 当前核处理的元素个数
    uint32_t blockOffset;   // 当前核在全局数据中的起始偏移（元素）
};

extern "C" __global__ __aicore__ void atanh_kernel(GM_ADDR x, GM_ADDR y,
                                                   GM_ADDR workspace,
                                                   GM_ADDR tiling) {
    // 从 Global Memory 读取 host 侧生成的 tiling 数据（macro 内部搬运到核内）
    GET_TILING_DATA(tilingData, tiling);

    KernelAtanh op;
    op.Init(x, y, tilingData.totalLength, tilingData.blockDim);
    op.Process();
}
