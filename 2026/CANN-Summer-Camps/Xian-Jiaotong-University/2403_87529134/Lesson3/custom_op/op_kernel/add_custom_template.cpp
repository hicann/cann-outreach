#include "kernel_operator.h"
#include "add_custom_template_tiling.h"
constexpr int32_t BUFFER_NUM = 2;  // tensor num for each queue
AscendC::TPipe pipe;
template <class dtypeX, class dtypeY, class dtypeZ>
class KernelAdd {
public:
    __aicore__ inline KernelAdd() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, 
uint32_t totalLength, uint32_t BlockLength, uint32_t tileNum, 
uint32_t tileLength, uint32_t blockDim, uint32_t typeSize)
    {
        this->blockLength = BlockLength;
        this->tileNum = tileNum;
        this->tileLength = tileLength;
        this->typeSize = typeSize;

        uint32_t blockIdx = AscendC::GetBlockIdx();
        uint32_t blockOffset = blockIdx * BlockLength;
        
        // 处理最后一个核可能数据不足的情况// 重新计算当前核的 tileNum（可能因尾块变化）
        uint32_t actualLength = BlockLength;
	uint32_t actualTileNum = tileNum;
        if (blockIdx == blockDim - 1) {
            actualLength = totalLength - blockOffset;
            actualTileNum=(actualLength - 1) / (tileLength )+1;
        }
        this->actualLength = actualLength;
	this->actualTileNum = actualTileNum;

        
        xGm.SetGlobalBuffer((__gm__ dtypeX *)x + blockOffset, actualLength);
        yGm.SetGlobalBuffer((__gm__ dtypeY *)y + blockOffset, actualLength);
        zGm.SetGlobalBuffer((__gm__ dtypeZ *)z + blockOffset, actualLength);
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * typeSize);
        pipe.InitBuffer(inQueueY, BUFFER_NUM, this->tileLength * typeSize);
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileLength * typeSize);
    }

    __aicore__ inline void Process()
    {

        int32_t loopCount =this->actualTileNum ;
	//int32_t loopCount =this->tileNum ;
        for (uint32_t i = 0; i < loopCount; i++) {
    	uint32_t offset = i * tileLength;
    	uint32_t actualLen = tileLength;

    	// 最后一个 tile 可能不足
    	if (offset + tileLength > this->actualLength) {
        	actualLen =  this->actualLength - offset;  // 截断到实际剩余
    	}
    	CopyIn(offset, actualLen);
    	Compute(offset, actualLen);
    	CopyOut(offset, actualLen);
	}

    }

private:
    __aicore__ inline void CopyIn(uint32_t offset,uint32_t tileLength)
    {
        uint32_t currentTileLength = tileLength;
        
        AscendC::LocalTensor<dtypeX> xLocal = inQueueX.AllocTensor<dtypeX>();
        AscendC::LocalTensor<dtypeY> yLocal = inQueueY.AllocTensor<dtypeY>();
        AscendC::DataCopy(xLocal, xGm[offset], currentTileLength);
        AscendC::DataCopy(yLocal, yGm[offset], currentTileLength);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }
    __aicore__ inline void Compute(uint32_t offset,uint32_t tileLength)
    {
        AscendC::LocalTensor<dtypeX> xLocal = inQueueX.DeQue<dtypeX>();
        AscendC::LocalTensor<dtypeY> yLocal = inQueueY.DeQue<dtypeY>();
        AscendC::LocalTensor<dtypeZ> zLocal = outQueueZ.AllocTensor<dtypeZ>();
        uint32_t currentTileLength = tileLength;
        AscendC::Add(zLocal, xLocal, yLocal, currentTileLength);
        outQueueZ.EnQue<dtypeZ>(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }
    __aicore__ inline void CopyOut(uint32_t offset,uint32_t tileLength)
    {
        AscendC::LocalTensor<dtypeZ> zLocal = outQueueZ.DeQue<dtypeZ>();
        uint32_t currentTileLength = tileLength;
        AscendC::DataCopy(zGm[offset], zLocal,currentTileLength);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueY;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueZ;
    AscendC::GlobalTensor<dtypeX> xGm;
    AscendC::GlobalTensor<dtypeY> yGm;
    AscendC::GlobalTensor<dtypeZ> zGm;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
    uint32_t typeSize;
    uint32_t actualLength;     // 当前核实际处理量
    uint32_t actualTileNum;    // 当前核实际 tile 数
};

 __global__ __aicore__ void add_custom_template(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(AddCustomTemplateTilingData);
    GET_TILING_DATA_WITH_STRUCT(AddCustomTemplateTilingData, tiling_data, tiling);
    KernelAdd<DTYPE_X, DTYPE_Y, DTYPE_Z> op;
    op.Init(x, y, z,tiling_data.totalLength,
            tiling_data.BlockLength,
            tiling_data.tileNum,
            tiling_data.tileLength,
            tiling_data.blockDim,
            tiling_data.typeSize);
    op.Process();
}
