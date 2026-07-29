#include "kernel_operator.h"
#include "add_custom_template_tiling.h"

constexpr int32_t DOUBLE_BUFFER = 2;
constexpr int32_t VECTOR_ALIGN = 16;

template <class DtypeX, class DtypeY, class DtypeZ>
class KernelAdd {
public:
    __aicore__ inline KernelAdd() {}
    
    __aicore__ inline void Init(GM_ADDR xPtr, GM_ADDR yPtr, GM_ADDR zPtr, 
                                  uint32_t totalLen, uint32_t tilesPerBlock)
    {
        this->blockSize = totalLen / AscendC::GetBlockNum();
        this->tilesPerBlock = tilesPerBlock;
        
        // Calculate tile size with vector alignment
        this->tileSize = (this->blockSize / tilesPerBlock / VECTOR_ALIGN) * VECTOR_ALIGN;
        if (this->tileSize < VECTOR_ALIGN) {
            this->tileSize = (this->blockSize < VECTOR_ALIGN) ? this->blockSize : VECTOR_ALIGN;
        }

        uint32_t blockStart = this->blockSize * AscendC::GetBlockIdx();
        xData.SetGlobalBuffer((__gm__ DtypeX *)xPtr + blockStart, this->blockSize);
        yData.SetGlobalBuffer((__gm__ DtypeY *)yPtr + blockStart, this->blockSize);
        zData.SetGlobalBuffer((__gm__ DtypeZ *)zPtr + blockStart, this->blockSize);

        pipe.InitBuffer(xQueue, DOUBLE_BUFFER, this->tileSize * sizeof(DtypeX));
        pipe.InitBuffer(yQueue, DOUBLE_BUFFER, this->tileSize * sizeof(DtypeY));
        pipe.InitBuffer(zQueue, DOUBLE_BUFFER, this->tileSize * sizeof(DtypeZ));
    }

    __aicore__ inline void Execute()
    {
        int32_t totalTiles = this->tilesPerBlock;

        // Prefetch first tile to fill the pipeline
        FetchInput(0);

        for (int32_t idx = 0; idx < totalTiles - 1; idx++) {
            FetchInput(idx + 1);    // Start DMA for next tile (overlaps with compute)
            ComputeTile(idx);       // Execute current tile on vector unit
            StoreOutput(idx);       // DMA out current result (overlaps with next fetch)
        }

        // Process the final tile without prefetching
        ComputeTile(totalTiles - 1);
        StoreOutput(totalTiles - 1);
    }

private:
    // Fetch input data for a specific tile index
    __aicore__ inline void FetchInput(int32_t tileIdx)
    {
        AscendC::LocalTensor<DtypeX> xLocal = xQueue.AllocTensor<DtypeX>();
        AscendC::LocalTensor<DtypeY> yLocal = yQueue.AllocTensor<DtypeY>();
        
        AscendC::DataCopy(xLocal, xData[tileIdx * this->tileSize], this->tileSize);
        AscendC::DataCopy(yLocal, yData[tileIdx * this->tileSize], this->tileSize);
        
        xQueue.EnQue(xLocal);
        yQueue.EnQue(yLocal);
    }

    // Perform addition computation on a tile
    __aicore__ inline void ComputeTile(int32_t tileIdx)
    {
        AscendC::LocalTensor<DtypeX> xLocal = xQueue.DeQue<DtypeX>();
        AscendC::LocalTensor<DtypeY> yLocal = yQueue.DeQue<DtypeY>();
        AscendC::LocalTensor<DtypeZ> zLocal = zQueue.AllocTensor<DtypeZ>();
        
        AscendC::Add(zLocal, xLocal, yLocal, this->tileSize);
        
        zQueue.EnQue<DtypeZ>(zLocal);
        xQueue.FreeTensor(xLocal);
        yQueue.FreeTensor(yLocal);
    }

    // Store computed result for a tile back to global memory
    __aicore__ inline void StoreOutput(int32_t tileIdx)
    {
        AscendC::LocalTensor<DtypeZ> zLocal = zQueue.DeQue<DtypeZ>();
        AscendC::DataCopy(zData[tileIdx * this->tileSize], zLocal, this->tileSize);
        zQueue.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, DOUBLE_BUFFER> xQueue;
    AscendC::TQue<AscendC::TPosition::VECIN, DOUBLE_BUFFER> yQueue;
    AscendC::TQue<AscendC::TPosition::VECOUT, DOUBLE_BUFFER> zQueue;
    AscendC::GlobalTensor<DtypeX> xData;
    AscendC::GlobalTensor<DtypeY> yData;
    AscendC::GlobalTensor<DtypeZ> zData;
    uint32_t blockSize;
    uint32_t tilesPerBlock;
    uint32_t tileSize;
};

// Kernel entry point for addition operation
__global__ __aicore__ void add_custom_template(GM_ADDR x, GM_ADDR y, GM_ADDR z, 
                                                  GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(AddCustomTemplateTilingData);
    GET_TILING_DATA_WITH_STRUCT(AddCustomTemplateTilingData, tiling_info, tiling);
    
    KernelAdd<DTYPE_X, DTYPE_Y, DTYPE_Z> addKernel;
    addKernel.Init(x, y, z, tiling_info.totalLength, tiling_info.tileNum);
    addKernel.Execute();
}
