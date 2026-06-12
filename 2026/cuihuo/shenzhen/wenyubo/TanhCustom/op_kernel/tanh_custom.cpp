#include "kernel_operator.h"
#include "tanh_custom_tiling.h"
using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2; // tensor num for each queue

class KernelTanh {
public:
    __aicore__ inline KernelTanh() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength, uint32_t tileNum)
    {
        // TODO: 考生自行补齐初始化函数
        ASSERT(GetBlockNum() != 0 && "block dim can not be zero!");
        this->blockLength = totalLength / GetBlockNum();
        this->tileNum = tileNum;
        ASSERT(tileNum != 0 && "tile num can not be zero!");
        this->tileLength = this->blockLength / tileNum / BUFFER_NUM;

        xGm.SetGlobalBuffer((__gm__ half*)x + this->blockLength * GetBlockIdx(), this->blockLength);
        yGm.SetGlobalBuffer((__gm__ half*)y + this->blockLength * GetBlockIdx(), this->blockLength);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(half));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileLength * sizeof(half));
        pipe.InitBuffer(tmpBuffer,  this->tileLength * sizeof(half));
    }
    __aicore__ inline void Process()
    {
        int32_t loopCount = this->tileNum * BUFFER_NUM;
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        // TODO: 考生自行补齐
        LocalTensor<half> xLocal = inQueueX.AllocTensor<half>();
        DataCopy(xLocal, xGm[progress * this->tileLength], this->tileLength);
        inQueueX.EnQue(xLocal);

    }
    __aicore__ inline void Compute(int32_t progress)
    {
        // TODO: 考生自行补齐
        // sinh(x) = (exp(x) - exp(-x)) / 2.0
        //LocalTensor<half> xLocal = inQueueX.DeQue<half>();
        //LocalTensor<half> yLocal = outQueueY.AllocTensor<half>();
        //LocalTensor<half> tmpTensor = tmpBuffer.Get<half>();
        // exp(x)
        //Exp(yLocal, xLocal, this->tileLength);
        //half inputVal(-1.0);
        //Duplicate<half>(tmpTensor, inputVal, this->tileLength);
        //Div(tmpTensor,xLocal,tmpTensor,this->tileLength);
        // exp(-x)
        //Exp(tmpTensor, tmpTensor, this->tileLength);
        // exp(x) - exp(-x)
        //yLocal = yLocal - tmpTensor;
        //half inputVal2(2.0);
        //Duplicate<half>(tmpTensor, inputVal2, this->tileLength);
        // (exp(x) - exp(-x)) / 2.0
        //Div(yLocal,yLocal,tmpTensor,this->tileLength);
        //outQueueY.EnQue<half>(yLocal);
        //inQueueX.FreeTensor(xLocal);
        //+++++++++++++++++++++++++++++++++
            // tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))

        LocalTensor<half> xLocal = inQueueX.DeQue<half>();
        LocalTensor<half> yLocal = outQueueY.AllocTensor<half>();
        LocalTensor<half> tmpTensor = tmpBuffer.Get<half>();

        // yLocal = exp(x)
        Exp(yLocal, xLocal, this->tileLength);

        // tmpTensor = -x
        half negVal(-1.0);
        Duplicate<half>(tmpTensor, negVal, this->tileLength);
        Mul(tmpTensor, xLocal, tmpTensor, this->tileLength);

        // tmpTensor = exp(-x)
        Exp(tmpTensor, tmpTensor, this->tileLength);

        // yLocal = exp(x) - exp(-x)，分子
        Sub(yLocal, yLocal, tmpTensor, this->tileLength);

        // tmpTensor = 2 * exp(-x)
        Add(tmpTensor, tmpTensor, tmpTensor, this->tileLength);

        // tmpTensor = exp(x) + exp(-x)，分母
        Add(tmpTensor, yLocal, tmpTensor, this->tileLength);

        // yLocal = tanh(x)
        Div(yLocal, yLocal, tmpTensor, this->tileLength);

        outQueueY.EnQue<half>(yLocal);
        inQueueX.FreeTensor(xLocal);

        
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        // TODO: 考生自行补齐
        LocalTensor<half> yLocal = outQueueY.DeQue<half>();
        DataCopy(yGm[progress * this->tileLength], yLocal, this->tileLength);
        outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmpBuffer;
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
    op.Init(x, y, tilingData.totalLength, tilingData.tileNum);
    if (TILING_KEY_IS(1)) {
        op.Process();
    }

}