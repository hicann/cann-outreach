#include "kernel_operator.h"
#include "tanh_custom_tiling.h"

constexpr int32_t BUFFER_NUM = 2; // tensor num for each queue

class KernelTanh {
public:
    __aicore__ inline KernelTanh() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength, uint32_t tileNum)
    {
        // TODO: 考生自行补齐初始化函数
                 ASSERT(AscendC::GetBlockNum() != 0 && "block dim is zero");
 	         this->blockLength = totalLength / AscendC::GetBlockNum();
 	         this->tileNum = tileNum;
 	         this->tileLength = this->blockLength / (this->tileNum * BUFFER_NUM);
 	 
 	         xGm.SetGlobalBuffer((__gm__ DTYPE_X*)x + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
 	         yGm.SetGlobalBuffer((__gm__ DTYPE_Y*)y + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
 	 
 	         pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(DTYPE_X));
 	         pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileLength * sizeof(DTYPE_Y));
 	         pipe.InitBuffer(tmpBuf0, this->tileLength * sizeof(float));
 	         pipe.InitBuffer(tmpBuf1, this->tileLength * sizeof(float));
 	         pipe.InitBuffer(tmpBuf2, this->tileLength * sizeof(float)); 

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
       
    AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.AllocTensor<DTYPE_X>();       
    AscendC::DataCopy(xLocal, xGm[progress * this->tileLength], this->tileLength);       
    inQueueX.EnQue(xLocal);
    
    }
    __aicore__ inline void Compute(int32_t progress)
    {
        // TODO: 考生自行补齐
                 AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.DeQue<DTYPE_X>();
 	         AscendC::LocalTensor<DTYPE_Y> yLocal = outQueueY.AllocTensor<DTYPE_Y>();
 	 
 	         AscendC::LocalTensor<float> tmpExpX = tmpBuf0.Get<float>();   // 存 e^x
 	         AscendC::LocalTensor<float> tmpExpNegX = tmpBuf1.Get<float>(); // 存 e^(-x)
 	         AscendC::LocalTensor<float> tmpNumerator = tmpBuf2.Get<float>(); // 存分子
 	         AscendC::LocalTensor<float> tmpDenominator;  // 可复用tmpBuf空间
 	 
 	         // 1. 转成float
 	         AscendC::Cast(tmpExpX, xLocal, AscendC::RoundMode::CAST_NONE, this->tileLength);
 	         
 	         // 2. 计算 e^x
 	         AscendC::Exp(tmpExpX, tmpExpX, this->tileLength);
 	         
 	         // 3. 计算 e^(-x) = 1 / e^x
 	         AscendC::Reciprocal(tmpExpNegX, tmpExpX, this->tileLength);
 	         
 	         // 4. 分子：e^x - e^(-x)
 	         AscendC::Sub(tmpNumerator, tmpExpX, tmpExpNegX, this->tileLength);
 	         
 	         // 5. 分母：e^x + e^(-x)
 	         AscendC::Add(tmpExpNegX, tmpExpX, tmpExpNegX, this->tileLength);  // 复用tmpExpNegX存分母
 	         
 	         // 6. 除法
 	         AscendC::Div(tmpExpX, tmpNumerator, tmpExpNegX, this->tileLength);  // 复用tmpExpX存结果
 	         
 	         // 7. 转回目标类型
 	         AscendC::Cast(yLocal, tmpExpX, AscendC::RoundMode::CAST_ROUND, this->tileLength);
 	 
 	         outQueueY.EnQue<DTYPE_Y>(yLocal);
 	         inQueueX.FreeTensor(xLocal);

     }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        // TODO: 考生自行补齐
         AscendC::LocalTensor<DTYPE_Y> yLocal = outQueueY.DeQue<DTYPE_Y>();       
         AscendC::DataCopy(yGm[progress * this->tileLength], yLocal, this->tileLength);      
         outQueueY.FreeTensor(yLocal);


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
    op.Init(x, y, tilingData.totalLength, tilingData.tileNum);
    op.Process();
}
