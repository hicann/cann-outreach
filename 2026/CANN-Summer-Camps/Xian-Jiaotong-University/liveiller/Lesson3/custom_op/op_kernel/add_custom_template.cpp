#include "kernel_operator.h"
#include "add_custom_template_tiling.h"

constexpr int32_t BUFFER_NUM = 2;  // 每个队列的 Tensor 数量（用于双缓冲控制）

template <class dtypeX, class dtypeY, class dtypeZ>
class KernelAdd {
public:
    __aicore__ inline KernelAdd() {}

    // 初始化函数：计算当前核的切块策略并分配 Unified Buffer 内存
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, 
                                uint32_t totalLength, uint32_t blockLength, uint32_t tileNum, 
                                uint32_t tileLength, uint32_t blockDim, uint32_t typeSize)
    {
        this->tileLength = tileLength;

        // 获取当前 AI Core 的 hardware ID
        uint32_t blockIdx = AscendC::GetBlockIdx();
        uint32_t blockOffset = blockIdx * blockLength;
        
        // 默认非尾核的处理数据量和切块数
        uint32_t currentBlockLength = blockLength;
        uint32_t currentTileNum = tileNum;

        // 尾核（最后一个 AI Core）特殊处理：防止数据量不足导致越界
        if (blockIdx == blockDim - 1) {
            currentBlockLength = totalLength - blockOffset;
            currentTileNum = (currentBlockLength + tileLength - 1) / tileLength;
        }

        this->actualLength = currentBlockLength;
        this->actualTileNum = currentTileNum;

        // 设置当前核在 Global Memory 上的起始偏移地址
        xGm.SetGlobalBuffer((__gm__ dtypeX *)x + blockOffset, this->actualLength);
        yGm.SetGlobalBuffer((__gm__ dtypeY *)y + blockOffset, this->actualLength);
        zGm.SetGlobalBuffer((__gm__ dtypeZ *)z + blockOffset, this->actualLength);

        // 为本地队列申请内存空间（单位：字节）
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * typeSize);
        pipe.InitBuffer(inQueueY, BUFFER_NUM, this->tileLength * typeSize);
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileLength * typeSize);
    }

    // 主处理流程：循环对每个 Tile 进行 CopyIn -> Compute -> CopyOut
    __aicore__ inline void Process()
    {
        int32_t loopCount = this->actualTileNum;
        for (uint32_t i = 0; i < loopCount; i++) {
            uint32_t offset = i * tileLength;
            uint32_t actualLen = tileLength;

            // 最后一个 Tile 长度可能不足，进行截断处理
            if (offset + tileLength > this->actualLength) {
                actualLen = this->actualLength - offset;
            }

            CopyIn(offset, actualLen);
            Compute(offset, actualLen);
            CopyOut(offset, actualLen);
        }
    }

private:
    // 将数据从 Global Memory 拷贝到 Local Memory (Unified Buffer)
    __aicore__ inline void CopyIn(uint32_t offset, uint32_t currentTileLength)
    {
        AscendC::LocalTensor<dtypeX> xLocal = inQueueX.AllocTensor<dtypeX>();
        AscendC::LocalTensor<dtypeY> yLocal = inQueueY.AllocTensor<dtypeY>();
        
        AscendC::DataCopy(xLocal, xGm[offset], currentTileLength);
        AscendC::DataCopy(yLocal, yGm[offset], currentTileLength);
        
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }

    // 执行矢量加法计算
    __aicore__ inline void Compute(uint32_t offset, uint32_t currentTileLength)
    {
        AscendC::LocalTensor<dtypeX> xLocal = inQueueX.DeQue<dtypeX>();
        AscendC::LocalTensor<dtypeY> yLocal = inQueueY.DeQue<dtypeY>();
        AscendC::LocalTensor<dtypeZ> zLocal = outQueueZ.AllocTensor<dtypeZ>();
        
        AscendC::Add(zLocal, xLocal, yLocal, currentTileLength);
        
        outQueueZ.EnQue<dtypeZ>(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }

    // 将计算结果从 Local Memory 写回 Global Memory
    __aicore__ inline void CopyOut(uint32_t offset, uint32_t currentTileLength)
    {
        AscendC::LocalTensor<dtypeZ> zLocal = outQueueZ.DeQue<dtypeZ>();
        
        AscendC::DataCopy(zGm[offset], zLocal, currentTileLength);
        
        outQueueZ.FreeTensor(zLocal);
    }

private:
    // 管道内存管理（已从全局移至此处 🌟）
    AscendC::TPipe pipe;

    // 管道队列定义
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueY;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueZ;
    
    // 全局内存指针
    AscendC::GlobalTensor<dtypeX> xGm;
    AscendC::GlobalTensor<dtypeY> yGm;
    AscendC::GlobalTensor<dtypeZ> zGm;
    
    // Tiling 切块配置参数
    uint32_t tileLength;
    uint32_t actualLength;     // 当前核实际需要处理的数据总长度
    uint32_t actualTileNum;    // 当前核实际需要循环的 Tile 次数
};

// 算子入口核函数
extern "C" __global__ __aicore__ void add_custom_template(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(AddCustomTemplateTilingData);
    GET_TILING_DATA_WITH_STRUCT(AddCustomTemplateTilingData, tiling_data, tiling);
    
    KernelAdd<DTYPE_X, DTYPE_Y, DTYPE_Z> op;
    op.Init(x, y, z, 
            tiling_data.totalLength,
            tiling_data.BlockLength,
            tiling_data.tileNum,
            tiling_data.tileLength,
            tiling_data.blockDim,
            tiling_data.typeSize);
    op.Process();
}