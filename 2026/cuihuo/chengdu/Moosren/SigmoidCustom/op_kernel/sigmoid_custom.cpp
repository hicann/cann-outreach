#include "kernel_operator.h"
using namespace AscendC;
constexpr int32_t BUFFER_NUM = 2;
class KernelSigmoid {
public:
    __aicore__ inline KernelSigmoid() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength, uint32_t tileNum)
    {
        //考生补充初始化代码
        ASSERT(GetBlockNum() !=0 && "block dim can not be zero!");
        this->blockLength = totalLength / GetBlockNum();
        this->tileNum =tileNum;
        ASSERT(tileNum !=0 && "tile num can not be zero!" );
        this->tileLength = this->blockLength / tileNum / BUFFER_NUM;
        xGm.SetGlobalBuffer((__gm__ DTYPE_X *)x + this->blockLength * GetBlockIdx(),this->blockLength);
        yGm.SetGlobalBuffer((__gm__ DTYPE_Y *)y + this->blockLength * GetBlockIdx(),this->blockLength);
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(half));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileLength * sizeof(half));
        //pipe.InitBuffer(tmpBuffer1, this->tileLength * sizeof(DTYPE_X));
	    //pipe.InitBuffer(tmpBuffer2, this->tileLength * sizeof(DTYPE_X));
	    //pipe.InitBuffer(tmpBuffer3, this->tileLength * sizeof(DTYPE_X));

    }
    __aicore__ inline void Process()
    {
        //考生补充对“loopCount”的定义，注意对Tiling的处理
        int32_t loopCount = this->tileNum*BUFFER_NUM;
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        //考生补充算子代码
        AscendC::LocalTensor<half> xLocal = inQueueX.AllocTensor<half>();
        AscendC::DataCopy(xLocal,xGm[progress * this->tileLength], this->tileLength);
        inQueueX.EnQue(xLocal);

    }
    __aicore__ inline void Compute(int32_t progress)
    {
        //考生补充算子计算代码
        AscendC::LocalTensor<half> xLocal = inQueueX.DeQue<half>();
        AscendC::LocalTensor<half> yLocal = outQueueY.AllocTensor<half>();
        //LocalTensor<DTYPE_X> tmpTensor1 = tmpBuffer1.Get<DTYPE_X>();
	    //LocalTensor<DTYPE_X> tmpTensor2 = tmpBuffer2.Get<DTYPE_X>();
	    //LocalTensor<DTYPE_X> tmpTensor3 = tmpBuffer3.Get<DTYPE_X>();

        const half input1 = -1;
        const half input2 = 1;
        // Sigmoid 公式 sigmoid (x) = 1/(1 + exp (-x))
        AscendC::Muls(xLocal, xLocal, input1,this->tileLength); //x=-x
        AscendC::Exp(xLocal, xLocal, this->tileLength); // y=exp(-x)
        AscendC::Adds(xLocal, xLocal, input2, this->tileLength); //y = 1+exp(-x)
        //Reciprocal(yLocal, yLocal, this->tileLength);
        AscendC::Power<half,false>(yLocal,xLocal,input1);  //y=1+exp(-x)^-1
        outQueueY.EnQue<half>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        //考生补充算子代码
        AscendC::LocalTensor<half> yLocal = outQueueY.DeQue<half>();
        AscendC::DataCopy(yGm[progress * this->tileLength], yLocal, this->tileLength);
        outQueueY.FreeTensor(yLocal);
    }

private:
    TPipe pipe;
    //create queue for input, in this case depth is equal to buffer num
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    //create queue for output, in this case depth is equal to buffer num
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    GlobalTensor<half> xGm;
    GlobalTensor<half> yGm;

    //考生补充自定义成员变量
    TBuf<QuePosition::VECCALC>; //tmpBuffer1, tmpBuffer2,tmpBuffer3;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;


};
extern "C" __global__ __aicore__ void sigmoid_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    GET_TILING_DATA(tiling_data, tiling);
    KernelSigmoid op;
    //补充init和process函数调用内容
    op.Init(x, y, tiling_data.totalLength, tiling_data.tileNum);
    op.Process();
}
