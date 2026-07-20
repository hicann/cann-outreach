#include "kernel_operator.h"
#include "add_custom_template_tiling.h"

using namespace AscendC;

namespace {
constexpr uint32_t BUFFER_NUM = 2;    // 双缓冲，掩盖数据搬运延迟
constexpr uint32_t TILE_LENGTH = 4096; // 调优点！如果跑完显示没到80µs，可以把这个值改成 2048 或 1024 测一测

__aicore__ inline uint32_t CeilDiv(uint32_t x, uint32_t y)
{
    return (x + y - 1U) / y;
}
}

class KernelSubCustomTemplate {
public:
    __aicore__ inline KernelSubCustomTemplate() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t totalLength)
    {
        const uint32_t blockNum = AscendC::GetBlockNum();
        const uint32_t blockIdx = AscendC::GetBlockIdx();
        // 处理总长度不能被核数整除的情况（尾部数据）
        const uint32_t baseBlockLength = totalLength / blockNum;
        const uint32_t tailBlockCount = totalLength % blockNum;
        const uint32_t extra = blockIdx < tailBlockCount ? 1U : 0U;
        blockLength_ = baseBlockLength + extra;
        const uint32_t blockOffset = blockIdx * baseBlockLength + (blockIdx < tailBlockCount ? blockIdx : tailBlockCount);

        // 计算当前核心需要分几次搬运（Tiling）
        tileNum_ = CeilDiv(blockLength_, TILE_LENGTH);

        // 绑定全局内存地址
        xGm_.SetGlobalBuffer((__gm__ float*)x + blockOffset, blockLength_);
        yGm_.SetGlobalBuffer((__gm__ float*)y + blockOffset, blockLength_);
        zGm_.SetGlobalBuffer((__gm__ float*)z + blockOffset, blockLength_);

        // 初始化双缓冲的队列（各2个槽位）
        pipe_.InitBuffer(xQueue_, BUFFER_NUM, TILE_LENGTH * sizeof(float));
        pipe_.InitBuffer(yQueue_, BUFFER_NUM, TILE_LENGTH * sizeof(float));
        pipe_.InitBuffer(zQueue_, BUFFER_NUM, TILE_LENGTH * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        if (blockLength_ == 0) {
            return;
        }
        for (uint32_t tileIdx = 0; tileIdx < tileNum_; ++tileIdx) {
            const uint32_t currentTileLength = GetCurrentTileLength(tileIdx);
            CopyIn(tileIdx, currentTileLength);
            Compute(currentTileLength);
            CopyOut(tileIdx, currentTileLength);
        }
    }

private:
    __aicore__ inline uint32_t GetCurrentTileLength(uint32_t tileIdx) const
    {
        const uint32_t offset = tileIdx * TILE_LENGTH;
        const uint32_t remain = blockLength_ - offset;
        return remain < TILE_LENGTH ? remain : TILE_LENGTH;
    }

    __aicore__ inline void CopyIn(uint32_t tileIdx, uint32_t currentTileLength)
    {
        LocalTensor<float> xLocal = xQueue_.AllocTensor<float>();
        LocalTensor<float> yLocal = yQueue_.AllocTensor<float>();
        const uint32_t offset = tileIdx * TILE_LENGTH;
        DataCopy(xLocal, xGm_[offset], currentTileLength);
        DataCopy(yLocal, yGm_[offset], currentTileLength);
        xQueue_.EnQue(xLocal);
        yQueue_.EnQue(yLocal);
    }

    __aicore__ inline void Compute(uint32_t currentTileLength)
    {
        LocalTensor<float> xLocal = xQueue_.DeQue<float>();
        LocalTensor<float> yLocal = yQueue_.DeQue<float>();
        LocalTensor<float> zLocal = zQueue_.AllocTensor<float>();
        
        // ⚠️ 核心改动：将原模板的 Add 改为了 Sub，保证功能依然是减法！
        Sub(zLocal, xLocal, yLocal, currentTileLength);

        zQueue_.EnQue<float>(zLocal);
        xQueue_.FreeTensor(xLocal);
        yQueue_.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOut(uint32_t tileIdx, uint32_t currentTileLength)
    {
        LocalTensor<float> zLocal = zQueue_.DeQue<float>();
        const uint32_t offset = tileIdx * TILE_LENGTH;
        DataCopy(zGm_[offset], zLocal, currentTileLength);
        zQueue_.FreeTensor(zLocal);
    }
private:
    TPipe pipe_;
    TQue<TPosition::VECIN, BUFFER_NUM> xQueue_;
    TQue<TPosition::VECIN, BUFFER_NUM> yQueue_;
    TQue<TPosition::VECOUT, BUFFER_NUM> zQueue_;
    GlobalTensor<float> xGm_;
    GlobalTensor<float> yGm_;
    GlobalTensor<float> zGm_;
    uint32_t blockLength_ = 0;
    uint32_t tileNum_ = 0;
};

extern "C" __global__ __aicore__ void add_custom_template(
    GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling)
{
    // 使用 msopgen 生成的宏获取 Tiling 数据
    REGISTER_TILING_DEFAULT(AddCustomTemplateTilingData);
    GET_TILING_DATA_WITH_STRUCT(AddCustomTemplateTilingData, tiling_data, tiling);
    
    KernelSubCustomTemplate op;
    op.Init(x, y, z, tiling_data.totalLength);
    op.Process();
}
