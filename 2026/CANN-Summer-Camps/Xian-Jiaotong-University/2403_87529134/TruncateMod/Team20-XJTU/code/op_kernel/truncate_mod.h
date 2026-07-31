/*!
 * \file truncate_mod.h
 * \brief TruncateMod 算子 kernel 类定义
 */


#ifndef TRUNCATEMOD_H
#define TRUNCATEMOD_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "truncate_mod_tiling_data.h"
#include "truncate_mod_tiling_key.h"

namespace NsTruncateMod {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;

template <typename T>
class TruncateMod {
public:
    __aicore__ inline TruncateMod(){};

    __aicore__ inline void Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, const TruncateModTilingData* tilingData);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(int64_t progress, int64_t currentNum);
    __aicore__ inline void CopyOut(int64_t progress, int64_t currentNum);
    __aicore__ inline void Compute(int64_t currentNum);

private:
    TPipe pipe;
    AscendC::TBuf<AscendC::TPosition::VECCALC> bufFloat;  // 或 VECIN/VECOUT
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX1;
    TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX2;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueY;

    GlobalTensor<T> inputGMX1;
    GlobalTensor<T> inputGMX2;
    GlobalTensor<T> outputGMY;

    //int64_t blockLength_ = 0;
    //int64_t ubLength_ = 0;
    
    //int64_t totalNum = 0;
    int64_t blockFactor= 0;
    int64_t ubFactor = 0;

    uint32_t blockDim=0;
    uint32_t tileNum=0;

    

};

//d TODO: 实现具体的 kernel 逻辑
template <typename T>
__aicore__ inline void TruncateMod<T>::Init(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, const TruncateModTilingData* tilingData)
{
    //d TODO: 实现 Init 逻辑
        //this->totalNum = tilingData->totalNum;
            
        this->blockFactor = tilingData->blockFactor;
        //this->blockLength = tilingData->blockFactor;
        
        this->ubFactor = tilingData->ubFactor;
        
        this->blockDim = (tilingData->totalNum -1)/this->blockFactor+1;
        this->tileNum = (this->blockFactor -1)/this->ubFactor+1;
        uint32_t blockIdx = AscendC::GetBlockIdx();
        if(blockIdx == this->blockDim - 1){
          this->blockFactor = tilingData->totalNum-tilingData->blockFactor*(blockDim - 1);
          this->tileNum = (this->blockFactor -1)/this->ubFactor+1;
        }

        int64_t blockOffset = blockIdx * this->blockFactor;
        
        inputGMX1.SetGlobalBuffer((__gm__ T *)x1 + blockOffset, this->blockFactor);
        inputGMX2.SetGlobalBuffer((__gm__ T *)x2 + blockOffset, this->blockFactor);
        outputGMY.SetGlobalBuffer((__gm__ T *)y + blockOffset, this->blockFactor);
        pipe.InitBuffer(inputQueueX1, BUFFER_NUM, this->blockFactor * sizeof(T));
        pipe.InitBuffer(inputQueueX2, BUFFER_NUM, this->blockFactor * sizeof(T));
        pipe.InitBuffer(outputQueueY, BUFFER_NUM, this->blockFactor * sizeof(T));
        pipe.InitBuffer(bufFloat, this->ubFactor *( sizeof(float)*4+sizeof(int32_t)));
    
}

template <typename T>
__aicore__ inline void TruncateMod<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    //d TODO: 实现 CopyIn 逻辑
    AscendC::LocalTensor<T> x1Local = inputQueueX1.AllocTensor<T>();
    AscendC::LocalTensor<T> x2Local = inputQueueX2.AllocTensor<T>();
    AscendC::DataCopy(x1Local, inputGMX1[progress*this->ubFactor], currentNum);
    AscendC::DataCopy(x2Local, inputGMX2[progress*this->ubFactor], currentNum);
    inputQueueX1.EnQue(x1Local);
    inputQueueX2.EnQue(x2Local);
}

template <typename T>
__aicore__ inline void TruncateMod<T>::Compute(int64_t currentNum)
{
    //d TODO: 实现 Compute 逻辑
    AscendC::UnaryRepeatParams repeatParams;
    
    AscendC::LocalTensor<T> x1Local = inputQueueX1.DeQue<T>();
    AscendC::LocalTensor<T> x2Local = inputQueueX2.DeQue<T>();
    AscendC::LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();
    
    AscendC::LocalTensor<float> Float1 = bufFloat.Get<float>(currentNum);
    AscendC::LocalTensor<float> Float2 = bufFloat.Get<float>(currentNum);
    AscendC::LocalTensor<float> Float3 = bufFloat.Get<float>(currentNum);
    AscendC::LocalTensor<float> Float4 = bufFloat.Get<float>(currentNum);
    AscendC::LocalTensor<int32_t> Int3 = bufFloat.Get<int32_t>(currentNum);

    //y=x1-|_x1/x2_|*x2
    AscendC::Cast<float,T>(Float1, x1Local, AscendC::RoundMode::CAST_TRUNC,currentNum, 1,repeatParams);
    AscendC::Cast<float,T>(Float2, x2Local, AscendC::RoundMode::CAST_TRUNC,currentNum, 1,repeatParams);
    AscendC::Div(Float3, Float1, Float2, currentNum);
    AscendC::Cast<float, int32_t>( Float3, Int3, AscendC::RoundMode::CAST_TRUNC,currentNum, 1,repeatParams);
    AscendC::Cast<int32_t, float>(Int3, Float3, AscendC::RoundMode::CAST_TRUNC,currentNum, 1,repeatParams);
    AscendC::Mul(Float4, Float3, Float2, currentNum);
    AscendC::Sub(Float2, Float1, Float4, currentNum);
    AscendC::Cast<T,float>(yLocal, Float2,AscendC::RoundMode::CAST_TRUNC,currentNum, 1,repeatParams);
    
    outputQueueY.EnQue<T>(yLocal);
    inputQueueX1.FreeTensor(x1Local);
    inputQueueX2.FreeTensor(x2Local);

}

template <typename T>
__aicore__ inline void TruncateMod<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    //d TODO: 实现 CopyOut 逻辑
        AscendC::LocalTensor<T> yLocal = outputQueueY.DeQue<T>();
            
        AscendC::DataCopy(outputGMY[progress*this->ubFactor], yLocal,currentNum);
        outputQueueY.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void TruncateMod<T>::Process()
{
     //d TODO: 实现 Process 逻辑
	   int32_t loopCount =this->tileNum ;
     for (uint32_t i = 0; i < loopCount; i++) {
     
    	uint32_t actualLen = ubFactor;
    	// 最后一个 tile 可能不足
      if ((i+1) * this->ubFactor > this->blockFactor) {
        	actualLen =  this->blockFactor - i* this->ubFactor;  // 截断到实际剩余
    	}
    	CopyIn(i, actualLen);
    	Compute(actualLen);
    	CopyOut(i, actualLen);
      }
}
} // namespace NsTruncateMod
#endif // TRUNCATEMOD_H
