/**
 * @file add_custom.cpp
 * @brief add_custom 算子的 Kernel 侧实现（Ascend C / Ascend 910B）。
 *
 * 计算流程（单核视角）：
 *
 *   GM(x) ──MTE2──┐
 *                 ├─► UB(xLocal) ─┐
 *   GM(y) ──MTE2──┘                ├─► Vector Add ─► UB(zLocal) ──MTE3──► GM(z)
 *                                  │
 *   通过 TQue(BUFFER_NUM = 2) 做双缓冲，使 MTE2 / Vector / MTE3 三个流水阶段重叠执行。
 *
 * 非对齐处理：
 *   每个 tile 的起始地址都保证落在 32B 边界上（tileNum 恒为 16 的整数倍），
 *   唯一可能非 32B 对齐的是最后一个 tile 的长度，统一用 DataCopyPad 完成
 *   GM <-> UB 搬运，从而无需为头部/尾部单独写分支。
 */

#include "kernel_operator.h"
#include "../op_host/add_custom_tiling.h"

using namespace AscendC;

// 双缓冲：VECIN / VECOUT 队列各申请 2 个 tile
constexpr int32_t BUFFER_NUM = 2;

// 该 kernel 只使用 Vector 计算单元，显式声明为纯 Vector 任务。
// 若不声明，910B 上会按 MIX（AIC+AIV）模式下发，GetBlockIdx() 的取值域
// 会变成 2 * blockDim，与 host 侧按 GetCoreNumAiv() 设置的 blockDim 不匹配。
KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

class KernelAdd {
public:
    __aicore__ inline KernelAdd() {}

    /**
     * @param x,y,z        算子输入输出全局内存地址
     * @param totalLength  元素总数 N2 * N1
     * @param tileNum      单 tile 元素个数（16 的整数倍）
     * @param tileCount    tile 总数
     * @param perCoreTile  每个核分到的 tile 个数
     */
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z,
                                uint32_t totalLength, uint32_t tileNum,
                                uint32_t tileCount, uint32_t perCoreTile)
    {
        this->totalLength = totalLength;
        this->tileNum     = tileNum;
        this->tileCount   = tileCount;

        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        ASSERT(tileNum != 0 && "tile num can not be zero!");

        const uint32_t blockIdx = GetBlockIdx();

        // 把 tileCount 个 tile 按 perCoreTile 分给各核。
        // 最后一个核的 loopCount 可能为 0（tileCount 无法被 coreNum 整除时）。
        uint32_t startTile = blockIdx * perCoreTile;
        uint32_t endTile   = startTile + perCoreTile;
        if (startTile > tileCount) { startTile = tileCount; }
        if (endTile   > tileCount) { endTile   = tileCount; }

        this->startOffset = static_cast<uint64_t>(startTile) * static_cast<uint64_t>(tileNum);
        // 末核可能拿到空分片，此时 startTile 被夹到 tileCount，startOffset 会落在
        // totalLength 之后；这里再夹一次，保证 GM 指针始终落在合法范围内
        if (this->startOffset > this->totalLength) { this->startOffset = this->totalLength; }
        this->loopCount = (endTile > startTile) ? (endTile - startTile) : 0U;

        // 本核实际负责的元素个数（尾部核不足一个完整分片，或 tileCount 不能整除时为 0）
        uint64_t endOffset = static_cast<uint64_t>(endTile) * static_cast<uint64_t>(tileNum);
        if (endOffset > this->totalLength) { endOffset = this->totalLength; }
        const uint32_t coreLength =
            (endOffset > this->startOffset) ? static_cast<uint32_t>(endOffset - this->startOffset) : 0U;

        // 每个核只映射到自己负责的那一段 GM，避免越界访问
        xGm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(x) + this->startOffset, coreLength);
        yGm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(y) + this->startOffset, coreLength);
        zGm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(z) + this->startOffset, coreLength);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, tileNum * sizeof(half));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, tileNum * sizeof(half));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, tileNum * sizeof(half));
    }

    __aicore__ inline void Process()
    {
        for (uint32_t i = 0; i < loopCount; ++i) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    /** 本次 tile 实际需要搬运/计算的元素个数（只有最后一个 tile 会小于 tileNum） */
    __aicore__ inline uint32_t GetTileLength(uint32_t index) const
    {
        const uint64_t globalStart =
            this->startOffset + static_cast<uint64_t>(index) * static_cast<uint64_t>(tileNum);
        const uint32_t remain = static_cast<uint32_t>(this->totalLength - globalStart);
        return (remain < this->tileNum) ? remain : this->tileNum;
    }

    __aicore__ inline void CopyIn(uint32_t index)
    {
        LocalTensor<half> xLocal = inQueueX.AllocTensor<half>();
        LocalTensor<half> yLocal = inQueueY.AllocTensor<half>();

        const uint64_t offset = static_cast<uint64_t>(index) * static_cast<uint64_t>(tileNum);
        const uint32_t len    = GetTileLength(index);

        // blockCount = 1，blockLen 以字节为单位；srcStride/dstStride = 0 表示连续搬运
        DataCopyExtParams copyParams{1U, static_cast<uint32_t>(len * sizeof(half)), 0U, 0U, 0U};
        // 不使用 pad 填充：UB 中超出 len 的尾部数据是上一轮的残留值，不会被写回 GM
        DataCopyPadExtParams<half> padParams{false, 0U, 0U, static_cast<half>(0)};

        DataCopyPad(xLocal, xGm[offset], copyParams, padParams);
        DataCopyPad(yLocal, yGm[offset], copyParams, padParams);

        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }

    __aicore__ inline void Compute(uint32_t index)
    {
        LocalTensor<half> xLocal = inQueueX.DeQue<half>();
        LocalTensor<half> yLocal = inQueueY.DeQue<half>();
        LocalTensor<half> zLocal = outQueueZ.AllocTensor<half>();

        const uint32_t len = GetTileLength(index);

        // 逐元素加法 z = x + y。
        // 当 len 不是 16 的整数倍时，该接口内部会按 32B 向上对齐参与计算，
        // 对齐后最多多算 15 个元素。因为 UB tile 的容量恒为 16 的整数倍且 >= len，
        // 这些多算的位置不会越界；同时 CopyOut 只写回 len 个元素，残留值不会被看到。
        Add(zLocal, xLocal, yLocal, len);

        outQueueZ.EnQue<half>(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOut(uint32_t index)
    {
        LocalTensor<half> zLocal = outQueueZ.DeQue<half>();

        const uint64_t offset = static_cast<uint64_t>(index) * static_cast<uint64_t>(tileNum);
        const uint32_t len    = GetTileLength(index);

        DataCopyExtParams copyParams{1U, static_cast<uint32_t>(len * sizeof(half)), 0U, 0U, 0U};
        DataCopyPad(zGm[offset], zLocal, copyParams);

        outQueueZ.FreeTensor(zLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM>  inQueueX;
    TQue<QuePosition::VECIN, BUFFER_NUM>  inQueueY;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueZ;

    GlobalTensor<half> xGm;
    GlobalTensor<half> yGm;
    GlobalTensor<half> zGm;

    uint64_t startOffset = 0U;   // 本核负责区段在整张张量中的元素偏移
    uint32_t totalLength = 0U;
    uint32_t tileNum     = 0U;
    uint32_t tileCount   = 0U;
    uint32_t loopCount   = 0U;
};

// ---------------------------------------------------------------------------
// Kernel 入口
// 函数名必须与算子类型 AddCustom 的 snake_case 形式一致，由框架据此完成绑定。
// ---------------------------------------------------------------------------
extern "C" __global__ __aicore__ void add_custom(GM_ADDR x, GM_ADDR y, GM_ADDR z,
                                                 GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;  // 本算子不需要额外 workspace

    GET_TILING_DATA(tilingData, tiling);

    KernelAdd op;
    op.Init(x, y, z,
            tilingData.totalLength,
            tilingData.tileNum,
            tilingData.tileCount,
            tilingData.perCoreTile);
    op.Process();
}
