#include "kernel_operator.h"
#include "tanh_custom_tiling.h"

constexpr int32_t BUFFER_NUM = 2; 

class KernelTanh {
public:
    __aicore__ inline KernelTanh() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength, uint32_t tileNum)
    {
        this->totalLength = totalLength;
        uint32_t coreNum = AscendC::GetBlockNum();
        uint32_t coreId = AscendC::GetBlockIdx();
        
        // 强制 32 Byte 对齐 (FP16 对应 16 元素)
        uint32_t alignNum = 32 / sizeof(DTYPE_X); 
        
        uint32_t blockBase = (totalLength + coreNum - 1) / coreNum;
        blockBase = ((blockBase + alignNum - 1) / alignNum) * alignNum;
        
        uint32_t coreOffset = coreId * blockBase;
        if (coreOffset >= totalLength) {
            this->blockLength = 0; 
        } else {
            this->blockLength = (coreOffset + blockBase > totalLength) ? (totalLength - coreOffset) : blockBase;
        }

        if (this->blockLength == 0) return;

        this->tileNum = tileNum;
        uint32_t baseTileLen = (this->blockLength + tileNum - 1) / tileNum;
        this->tileLengthAligned = ((baseTileLen + alignNum - 1) / alignNum) * alignNum;

        xGm.SetGlobalBuffer((__gm__ DTYPE_X*)x + coreOffset, this->blockLength);
        yGm.SetGlobalBuffer((__gm__ DTYPE_Y*)y + coreOffset, this->blockLength);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLengthAligned * sizeof(DTYPE_X));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileLengthAligned * sizeof(DTYPE_Y));
        
        // 核心修正：为高阶 Tanh 算子初始化专属计算内存队列 (VECCALC)
        // 分配充裕的临时内存空间以承载内部多项式计算缓存
        pipe.InitBuffer(calcBuf, this->tileLengthAligned * sizeof(DTYPE_X) * 4);
    }
    
    __aicore__ inline void Process()
    {
        if (this->blockLength == 0) return;
        
        uint32_t loopCount = (this->blockLength + this->tileLengthAligned - 1) / this->tileLengthAligned;
        
        for (uint32_t i = 0; i < loopCount; i++) {
            uint32_t curLength = this->tileLengthAligned;
            if (i == loopCount - 1) {
                curLength = this->blockLength - i * this->tileLengthAligned; 
            }
            CopyIn(i, curLength);
            Compute(curLength);
            CopyOut(i, curLength);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t progress, uint32_t len)
    {
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.AllocTensor<DTYPE_X>();
        uint32_t alignNum = 32 / sizeof(DTYPE_X);
        uint32_t lenAligned = ((len + alignNum - 1) / alignNum) * alignNum;
        
        AscendC::DataCopy(xLocal, xGm[progress * this->tileLengthAligned], lenAligned);
        inQueueX.EnQue(xLocal);
    }
    
    __aicore__ inline void Compute(uint32_t len)
    {
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.DeQue<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_Y> yLocal = outQueueY.AllocTensor<DTYPE_Y>();
        
        // 获取预分配的计算临时 Buffer
        AscendC::LocalTensor<uint8_t> tmpLocal = calcBuf.Get<uint8_t>();

        uint32_t alignNum = 32 / sizeof(DTYPE_X);
        uint32_t lenAligned = ((len + alignNum - 1) / alignNum) * alignNum;

        // 彻底抛弃手动指令级拆解，直接调用高阶硬件固化接口
        AscendC::Tanh(yLocal, xLocal, tmpLocal, lenAligned);

        outQueueY.EnQue<DTYPE_Y>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }
    
    __aicore__ inline void CopyOut(uint32_t progress, uint32_t len)
    {
        AscendC::LocalTensor<DTYPE_Y> yLocal = outQueueY.DeQue<DTYPE_Y>();
        uint32_t alignNum = 32 / sizeof(DTYPE_X);
        uint32_t lenAligned = ((len + alignNum - 1) / alignNum) * alignNum;
        
        AscendC::DataCopy(yGm[progress * this->tileLengthAligned], yLocal, lenAligned);
        outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> calcBuf; // 高阶算子依赖的 TmpBuffer 声明
    AscendC::GlobalTensor<DTYPE_X> xGm;
    AscendC::GlobalTensor<DTYPE_Y> yGm;
    uint32_t totalLength;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLengthAligned;
};

extern "C" __global__ __aicore__ void tanh_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    auto tilingData = reinterpret_cast<__gm__ TanhCustomTilingData*>(tiling);
    
    KernelTanh op;
    op.Init(x, y, tilingData->totalLength, tilingData->tileNum);
    op.Process();
}
