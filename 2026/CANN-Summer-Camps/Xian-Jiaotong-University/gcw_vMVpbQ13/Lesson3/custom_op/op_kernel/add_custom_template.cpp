#include "kernel_operator.h"
#include "add_custom_template_tiling.h"

constexpr int32_t BUFFER_NUM = 2;  // 双缓冲，提升流水线效率

template <class dtypeX, class dtypeY, class dtypeZ>
class KernelAdd {
public:
    __aicore__ inline KernelAdd() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t totalLength, uint32_t tileNum)
    {
        // 获取当前核心编号和总核心数
        uint32_t blockId = AscendC::GetBlockIdx();
        uint32_t blockNum = AscendC::GetBlockNum();
        
        // 计算当前核心负责的数据区间（尽量均匀）
        uint32_t perCoreLen = (totalLength + blockNum - 1) / blockNum;
        uint32_t start = blockId * perCoreLen;
        uint32_t end = (blockId == blockNum - 1) ? totalLength : (start + perCoreLen);
        this->blockLen = end - start;   // 实际长度可能小于 perCoreLen
        
        // 每个核心的 tile 数（从 Tiling 传入，但可能超过实际需要的，以实际长度为准重新计算）
        uint32_t tileSize = (this->blockLen + tileNum - 1) / tileNum; // 每个 tile 平均大小
        // 为使每个 tile 大小接近，这里采用均匀分配，但为了简化，直接使用 Tiling 给的 tileNum，并保证 tileLength = blockLen / tileNum（整除）
        // 但可能不整除，需处理最后一个 tile
        this->tileNum = tileNum;
        this->tileLength = this->blockLen / tileNum;   // 整除，余数放在最后一个 tile
        
        // 绑定全局内存，偏移到当前核心起始位置
        xGm.SetGlobalBuffer((__gm__ dtypeX *)x + start, this->blockLen);
        yGm.SetGlobalBuffer((__gm__ dtypeY *)y + start, this->blockLen);
        zGm.SetGlobalBuffer((__gm__ dtypeZ *)z + start, this->blockLen);
        
        // 初始化双缓冲队列
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(dtypeX));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, this->tileLength * sizeof(dtypeY));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileLength * sizeof(dtypeZ));
    }

    __aicore__ inline void Process()
    {
        // 处理所有 tile，最后一个 tile 长度可能不同
        for (int32_t i = 0; i < this->tileNum; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        AscendC::LocalTensor<dtypeX> xLocal = inQueueX.AllocTensor<dtypeX>();
        AscendC::LocalTensor<dtypeY> yLocal = inQueueY.AllocTensor<dtypeY>();
        uint32_t copyLen = this->tileLength;
        // 如果是最后一个 tile，且 blockLen 不是 tileLength 的整数倍，则处理余数
        if (progress == this->tileNum - 1) {
            copyLen = this->blockLen - progress * this->tileLength;
        }
        AscendC::DataCopy(xLocal, xGm[progress * this->tileLength], copyLen);
        AscendC::DataCopy(yLocal, yGm[progress * this->tileLength], copyLen);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }
    __aicore__ inline void Compute(int32_t progress)
    {
        AscendC::LocalTensor<dtypeX> xLocal = inQueueX.DeQue<dtypeX>();
        AscendC::LocalTensor<dtypeY> yLocal = inQueueY.DeQue<dtypeY>();
        AscendC::LocalTensor<dtypeZ> zLocal = outQueueZ.AllocTensor<dtypeZ>();
        // 实际计算长度
        uint32_t len = (progress == this->tileNum - 1) ? 
                       this->blockLen - progress * this->tileLength : this->tileLength;
        AscendC::Add(zLocal, xLocal, yLocal, len);
        outQueueZ.EnQue<dtypeZ>(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<dtypeZ> zLocal = outQueueZ.DeQue<dtypeZ>();
        uint32_t len = (progress == this->tileNum - 1) ? 
                       this->blockLen - progress * this->tileLength : this->tileLength;
        AscendC::DataCopy(zGm[progress * this->tileLength], zLocal, len);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueY;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueZ;
    AscendC::GlobalTensor<dtypeX> xGm;
    AscendC::GlobalTensor<dtypeY> yGm;
    AscendC::GlobalTensor<dtypeZ> zGm;
    uint32_t blockLen;
    uint32_t tileNum;
    uint32_t tileLength;
};

__global__ __aicore__ void add_custom_template(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(AddCustomTemplateTilingData);
    GET_TILING_DATA_WITH_STRUCT(AddCustomTemplateTilingData, tiling_data, tiling);
    KernelAdd<DTYPE_X, DTYPE_Y, DTYPE_Z> op;
    op.Init(x, y, z, tiling_data.totalLength, tiling_data.tileNum);
    op.Process();
}