/**
 * @file add_custom.cpp
 * @brief add_custom 算子 Kernel 侧实现（Ascend C）
 *
 * 功能描述：对两个输入张量 x、y 执行逐元素加法，返回结果张量 z
 * 数学公式：z = x + y
 * 输入：x[N2, N1]、y[N2, N1]，数据类型 float16，数据格式 ND
 * 输出：z[N2, N1]，数据类型 float16，数据格式 ND
 *
 * 实现要点：
 *  1. 多核切分：元素总数为 totalLength，按 blockLength 均分到 blockNum 个 AI Core，
 *     每个核处理 [blockIdx * blockLength, ...) 区间，超出总长度的核直接退出，
 *     最后一个核处理的数据量不足 blockLength 时按实际长度处理（核间尾块）。
 *  2. 核内切分：每个核内部再切成 tileNum 个 tile，配合 TPipe/TQue 的 double buffer
 *     实现 "搬运 - 计算 - 搬出" 的流水，掩盖数据搬运开销。
 *  3. 对齐处理：tileLength 按 32Byte（16 个 float16）对齐，因此绝大多数 tile 使用
 *     高效的 DataCopy 对齐搬运；仅核内最后一个不足一个 tile 的尾块使用 DataCopyPad，
 *     向量计算按 32B 对齐长度进行，多算出的元素不回写 GM，不影响结果正确性。
 *  4. 数据格式 ND：2 维 shape 仅用于 Host 侧推导元素总个数，Kernel 侧按一维连续内存
 *     处理即可（逐元素加法与 shape 无关）。
 */

#include "kernel_operator.h"
#include "add_custom_tiling.h"

using namespace AscendC;

// 每个队列的 buffer 份数（double buffer）
constexpr int32_t BUFFER_NUM = 2;

/**
 * @brief 向上取整到 align 的倍数
 */
__aicore__ inline uint32_t AlignUpCustom(uint32_t value, uint32_t align)
{
    return (value + align - 1) / align * align;
}

/**
 * @brief add_custom 算子 Kernel 实现类：z = x + y
 */
class KernelAddCustom {
public:
    __aicore__ inline KernelAddCustom() {}

    /**
     * @brief 初始化：解析 Tiling、切分 GM、申请 UB buffer
     * @param x      输入 x 的 GM 地址
     * @param y      输入 y 的 GM 地址
     * @param z      输出 z 的 GM 地址
     * @param tiling Tiling 参数所在的 GM 地址
     */
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR tiling)
    {
        __gm__ AddCustomTiling* tilingData = (__gm__ AddCustomTiling*)tiling;

        coreIdx_ = GetBlockIdx();
        totalLength_ = tilingData->totalLength;
        blockNum_ = tilingData->blockNum;
        blockLength_ = tilingData->blockLength;
        // 核内循环次数由本核实际数据量 coreLength_ 与 tileLength_ 推导（见 Process），
        // 不使用 Tiling 中的 tileNum，避免核间尾块导致的多余循环
        tileLength_ = tilingData->tileLength;
        coreLength_ = 0;
        usedCore_ = false;

        // 核数超过实际计算数据的核直接退出，不参与计算
        if (coreIdx_ >= blockNum_) {
            return;
        }

        uint64_t start = (uint64_t)coreIdx_ * (uint64_t)blockLength_;
        if (start >= (uint64_t)totalLength_) {
            return;
        }

        // 最后一个核的数据量可能不足 blockLength（核间尾块）
        uint32_t remain = (uint32_t)((uint64_t)totalLength_ - start);
        coreLength_ = (remain < blockLength_) ? remain : blockLength_;
        usedCore_ = true;

        xGm_.SetGlobalBuffer((__gm__ half*)x + start, coreLength_);
        yGm_.SetGlobalBuffer((__gm__ half*)y + start, coreLength_);
        zGm_.SetGlobalBuffer((__gm__ half*)z + start, coreLength_);

        pipe_.InitBuffer(inQueueX_, BUFFER_NUM, tileLength_ * sizeof(half));
        pipe_.InitBuffer(inQueueY_, BUFFER_NUM, tileLength_ * sizeof(half));
        pipe_.InitBuffer(outQueueZ_, BUFFER_NUM, tileLength_ * sizeof(half));
    }

    /**
     * @brief 执行主体：核内按 tile 循环，搬运 - 计算 - 搬出
     */
    __aicore__ inline void Process()
    {
        if (!usedCore_) {
            return;
        }

        uint32_t loopCount = (coreLength_ + tileLength_ - 1) / tileLength_;
        for (uint32_t i = 0; i < loopCount; ++i) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    /**
     * @brief 计算第 progress 个 tile 的有效元素个数
     * @note 调用前需保证 offset < coreLength_
     */
    __aicore__ inline uint32_t GetTileLength(uint32_t progress)
    {
        uint32_t offset = progress * tileLength_;
        uint32_t remain = coreLength_ - offset;
        return (remain < tileLength_) ? remain : tileLength_;
    }

    /**
     * @brief 把 x、y 的一个 tile 从 GM 搬到 UB
     */
    __aicore__ inline void CopyIn(uint32_t progress)
    {
        uint32_t offset = progress * tileLength_;
        uint32_t len = GetTileLength(progress);

        LocalTensor<half> xLocal = inQueueX_.AllocTensor<half>();
        LocalTensor<half> yLocal = inQueueY_.AllocTensor<half>();

        if (len == tileLength_) {
            // 对齐场景：长度是 32B 的整数倍，直接使用 DataCopy
            DataCopy(xLocal, xGm_[offset], len);
            DataCopy(yLocal, yGm_[offset], len);
        } else {
            // 非对齐尾块：使用 DataCopyPad，避免越界读写。
            // isPad = true 且 paddingValue = 0：UB 中多出的对齐区被填 0，
            // 从而不读取未初始化数据（这些元素不参与回写，不影响结果正确性）。
            DataCopyExtParams copyParams{1, (uint32_t)(len * sizeof(half)), 0, 0, 0};
            DataCopyPadExtParams<half> padParams{true, 0, 0, (half)0};
            DataCopyPad(xLocal, xGm_[offset], copyParams, padParams);
            DataCopyPad(yLocal, yGm_[offset], copyParams, padParams);
        }

        inQueueX_.EnQue(xLocal);
        inQueueY_.EnQue(yLocal);
    }

    /**
     * @brief 向量计算：z = x + y
     */
    __aicore__ inline void Compute(uint32_t progress)
    {
        LocalTensor<half> xLocal = inQueueX_.DeQue<half>();
        LocalTensor<half> yLocal = inQueueY_.DeQue<half>();
        LocalTensor<half> zLocal = outQueueZ_.AllocTensor<half>();

        // 向量计算按 32B 对齐长度下发；尾块多算出的元素停留在 UB 中，不会回写 GM
        uint32_t calCount = AlignUpCustom(GetTileLength(progress), ADD_CUSTOM_ALIGN_ELEM);
        Add(zLocal, xLocal, yLocal, calCount);

        outQueueZ_.EnQue(zLocal);
        inQueueX_.FreeTensor(xLocal);
        inQueueY_.FreeTensor(yLocal);
    }

    /**
     * @brief 把结果 z 的一个 tile 从 UB 搬到 GM
     */
    __aicore__ inline void CopyOut(uint32_t progress)
    {
        uint32_t offset = progress * tileLength_;
        uint32_t len = GetTileLength(progress);

        LocalTensor<half> zLocal = outQueueZ_.DeQue<half>();
        if (len == tileLength_) {
            DataCopy(zGm_[offset], zLocal, len);
        } else {
            DataCopyExtParams copyParams{1, (uint32_t)(len * sizeof(half)), 0, 0, 0};
            DataCopyPad(zGm_[offset], zLocal, copyParams);
        }
        outQueueZ_.FreeTensor(zLocal);
    }

private:
    TPipe pipe_;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX_;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueY_;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueZ_;
    GlobalTensor<half> xGm_;
    GlobalTensor<half> yGm_;
    GlobalTensor<half> zGm_;
    uint32_t coreIdx_;
    uint32_t totalLength_;
    uint32_t blockNum_;
    uint32_t blockLength_;
    uint32_t coreLength_;
    uint32_t tileLength_;
    bool usedCore_;
};

/**
 * @brief add_custom 算子 Kernel 入口函数
 * @note 函数名与文件名保持一致，框架会据此生成 aclrtlaunch_add_custom.h
 */
extern "C" __global__ __aicore__ void add_custom(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR tiling)
{
    KernelAddCustom op;
    op.Init(x, y, z, tiling);
    op.Process();
}
