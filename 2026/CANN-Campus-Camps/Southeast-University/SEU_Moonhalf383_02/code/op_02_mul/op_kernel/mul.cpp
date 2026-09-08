// Kernel侧核函数实现
#include "kernel_operator.h"

#include "mul_tiling.h"
#include "tiling_key_mul.h"

/*
 * Mul 参考实现（z = x * y，逐元素，fp16/fp32 由模板参数 DT_X 区分）
 *
 * 切分策略（Host 侧已设 blockDim = AIV 核数，tiling 只传总元素数 length）：
 *   1. 核内用 GetBlockNum()/GetBlockIdx() 均分 length，每核长度向上对齐到
 *      ALIGN_NUM（= 32B / sizeof(DT_X)，fp32 为 8、fp16 为 16），保证 GM 侧
 *      每段的起始地址满足 32 字节对齐；
 *   2. 核内再按 TILE_LENGTH 分块走 CopyIn → Compute(Mul) → CopyOut 流水，
 *      TQue 双缓冲（BUFFER_NUM=2）为搬运/计算的硬件级重叠留出空间；
 *   3. 对齐的整块用 DataCopy；末尾不足对齐粒度的残块用 DataCopyPad
 *      搬入/搬出，计算仍用 Mul 的前 n 个元素接口（起始地址天然对齐）。
 */

constexpr int32_t BUFFER_NUM = 2;      // 双缓冲
constexpr uint32_t TILE_LENGTH = 1024; // 每块元素数，须为 ALIGN_NUM(fp16=16) 的倍数

template <class DT_X>
class KernelMul {
public:
    __aicore__ inline KernelMul() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t length) {
        // fp32: 8 个元素/32B；fp16: 16 个元素/32B
        constexpr uint32_t alignNum = 32 / sizeof(DT_X);

        uint32_t blockNum = static_cast<uint32_t>(AscendC::GetBlockNum());
        uint32_t blockIdx = static_cast<uint32_t>(AscendC::GetBlockIdx());
        if (blockNum == 0) {
            blockNum = 1;
        }

        // 每核基线长度：ceil(length / blockNum)，再向上对齐到 alignNum
        uint32_t perCore = (length + blockNum - 1) / blockNum;
        perCore = (perCore + alignNum - 1) / alignNum * alignNum;

        uint32_t coreOffset = blockIdx * perCore;
        uint32_t remain = (coreOffset < length) ? (length - coreOffset) : 0;
        coreLength_ = (remain < perCore) ? remain : perCore; // 最后一个核只做尾巴
        tileNum_ = (coreLength_ + TILE_LENGTH - 1) / TILE_LENGTH;

        // 绑定本核负责的 GM 区间
        xGm_.SetGlobalBuffer((__gm__ DT_X *)x + coreOffset, coreLength_);
        yGm_.SetGlobalBuffer((__gm__ DT_X *)y + coreOffset, coreLength_);
        zGm_.SetGlobalBuffer((__gm__ DT_X *)z + coreOffset, coreLength_);

        // 为三个队列划分 UB（工作台）空间
        pipe_.InitBuffer(queX_, BUFFER_NUM, TILE_LENGTH * sizeof(DT_X));
        pipe_.InitBuffer(queY_, BUFFER_NUM, TILE_LENGTH * sizeof(DT_X));
        pipe_.InitBuffer(queZ_, BUFFER_NUM, TILE_LENGTH * sizeof(DT_X));
    }
    __aicore__ inline void Process() {
        for (uint32_t p = 0; p < tileNum_; p++) {
            CopyIn(p);
            Compute(p);
            CopyOut(p);
        }
    }
private:
    __aicore__ inline uint32_t TileCount(uint32_t progress) {
        uint32_t remain = coreLength_ - progress * TILE_LENGTH;
        return (remain < TILE_LENGTH) ? remain : TILE_LENGTH;
    }
    __aicore__ inline void CopyIn(uint32_t progress) {
        uint32_t curLen = TileCount(progress);
        AscendC::LocalTensor<DT_X> xLocal = queX_.AllocTensor<DT_X>();
        AscendC::LocalTensor<DT_X> yLocal = queY_.AllocTensor<DT_X>();
        if ((curLen % (32 / sizeof(DT_X))) == 0) {
            // 对齐块：普通搬运
            AscendC::DataCopy(xLocal, xGm_[progress * TILE_LENGTH], curLen);
            AscendC::DataCopy(yLocal, yGm_[progress * TILE_LENGTH], curLen);
        } else {
            // 残块：非对齐搬运（硬件按 32B 补齐，多余部分不会写出）
            // 注意：DataCopyExtParams.blockLen 的单位是字节（Byte），须乘 sizeof
            // CANN 9.0 起构造函数为 {blockCount, blockLen, srcStride, dstStride, rsv}
            AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(curLen * sizeof(DT_X)), 0, 0, 0};
            AscendC::DataCopyPadExtParams<DT_X> padParams{false, 0, 0, static_cast<DT_X>(0)};
            AscendC::DataCopyPad(xLocal, xGm_[progress * TILE_LENGTH], copyParams, padParams);
            AscendC::DataCopyPad(yLocal, yGm_[progress * TILE_LENGTH], copyParams, padParams);
        }
        queX_.EnQue(xLocal);
        queY_.EnQue(yLocal);
    }
    __aicore__ inline void Compute(uint32_t progress) {
        uint32_t curLen = TileCount(progress);
        AscendC::LocalTensor<DT_X> xLocal = queX_.DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> yLocal = queY_.DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> zLocal = queZ_.AllocTensor<DT_X>();
        AscendC::Mul(zLocal, xLocal, yLocal, curLen); // z = x * y，一条向量指令
        queX_.FreeTensor(xLocal);
        queY_.FreeTensor(yLocal);
        queZ_.EnQue(zLocal);
    }
    __aicore__ inline void CopyOut(uint32_t progress) {
        uint32_t curLen = TileCount(progress);
        AscendC::LocalTensor<DT_X> zLocal = queZ_.DeQue<DT_X>();
        if ((curLen % (32 / sizeof(DT_X))) == 0) {
            AscendC::DataCopy(zGm_[progress * TILE_LENGTH], zLocal, curLen);
        } else {
            AscendC::DataCopyExtParams copyParams{1, static_cast<uint32_t>(curLen * sizeof(DT_X)), 0, 0, 0};
            AscendC::DataCopyPad(zGm_[progress * TILE_LENGTH], zLocal, copyParams);
        }
        queZ_.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe pipe_;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> queX_;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> queY_;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> queZ_;
    AscendC::GlobalTensor<DT_X> xGm_;
    AscendC::GlobalTensor<DT_X> yGm_;
    AscendC::GlobalTensor<DT_X> zGm_;
    uint32_t coreLength_ = 0;
    uint32_t tileNum_ = 0;
};

template <typename DT_X>
 __global__ __aicore__ void mul(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(MulTilingData);
    GET_TILING_DATA_WITH_STRUCT(MulTilingData, tiling_data, tiling);
    KernelMul<DT_X> op;
    op.Init(x, y, z, tiling_data.length);
    op.Process();
}
