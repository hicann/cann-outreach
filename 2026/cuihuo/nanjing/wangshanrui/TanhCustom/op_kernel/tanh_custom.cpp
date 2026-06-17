#include "kernel_operator.h"
#include "tanh_custom_tiling.h"

constexpr int32_t BUFFER_NUM = 2; // tensor num for each queue

class KernelTanh {
public:
    __aicore__ inline KernelTanh() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength, uint32_t tileNum)
    {
        // TODO: 考生自行补齐初始化函数
        this->xGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_X*>(x));
        this->yGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_Y*>(y));
        
        this->blockLength = tileNum;
        
        pipe.InitBuffer(inQueueX, 1, blockLength * sizeof(DTYPE_X));
        pipe.InitBuffer(outQueueY, 1, blockLength * sizeof(DTYPE_Y));
    }
    __aicore__ inline void Process()
    {
        CopyIn();
        Compute();
        CopyOut();
    }

private:
    __aicore__ inline void CopyIn()
    {
        // TODO: 考生自行补齐
        auto tensor = inQueueX.AllocTensor<DTYPE_X>();
        uint32_t offset = AscendC::GetBlockIdx() * blockLength;
        
        AscendC::DataCopy(tensor, xGm[offset], blockLength);
        inQueueX.EnQue(tensor);
    }
    __aicore__ inline void Compute()
    {
        // TODO: 考生自行补齐
        auto inTensor = inQueueX.DeQue<DTYPE_X>();
        auto outTensor = outQueueY.AllocTensor<DTYPE_Y>();
        
        AscendC::Tanh(outTensor, inTensor, blockLength);
        
        outQueueY.EnQue(outTensor);
        inQueueX.FreeTensor(inTensor);
    }
    __aicore__ inline void CopyOut()
    {
        // TODO: 考生自行补齐
        auto tensor = outQueueY.DeQue<DTYPE_Y>();
        uint32_t offset = AscendC::GetBlockIdx() * blockLength;
        
        AscendC::DataCopy(yGm[offset], tensor, blockLength);
        outQueueY.FreeTensor(tensor);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmpBuf0,tmpBuf1,tmpBuf2;
    AscendC::GlobalTensor<DTYPE_X> xGm;
    AscendC::GlobalTensor<DTYPE_Y> yGm;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
};

extern "C" __global__ __aicore__ void tanh_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(TanhCustomTilingData);
    GET_TILING_DATA(tilingData, tiling);
    // TODO: 考生自行补齐
    KernelTanh op;
    op.Init(x, y, tilingData.totalElements, tilingData.elementsPerBlock);
    op.Process();
}