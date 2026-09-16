/*!
 	  * \file relu.h
 	  * \brief Relu 算子 kernel 类定义
 	  */
 	 
 	 #ifndef RELU_H
 	 #define RELU_H
 	 
 	 #include "kernel_operator.h"
 	 #include "kernel_tiling/kernel_tiling.h"
 	 #include "relu_tiling_data.h"
 	 #include "relu_tiling_key.h"
 	 
 	 namespace NsRelu {
 	 
 	 using namespace AscendC;
 	 
 	 constexpr int32_t BUFFER_NUM = 2;
 	 
 	 template <typename T>
 	 class Relu {
 	 public:
 	     __aicore__ inline Relu(){};
 	 
 	     __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const ReluTilingData* tilingData);
 	 
 	     __aicore__ inline void Process();
 	 
 	 private:
 	     __aicore__ inline void CopyIn(int64_t progress, int64_t currentNum);
 	 
 	     __aicore__ inline void CopyOut(int64_t progress, int64_t currentNum);
 	 
 	     __aicore__ inline void Compute(int64_t currentNum);
 	 
 	 private:
 	     TPipe pipe;
 	 
 	     TQue<QuePosition::VECIN, BUFFER_NUM> inputQueueX;
 	 
 	     TQue<QuePosition::VECOUT, BUFFER_NUM> outputQueueY;
 	 
 	     GlobalTensor<T> inputGMX;
 	     GlobalTensor<T> outputGMY;
 	 
 	     int64_t blockLength_ = 0;
 	     int64_t ubLength_ = 0;
 	 };
 	 
 	 template <typename T>
 	 __aicore__ inline void Relu<T>::Init(GM_ADDR x, GM_ADDR y, const ReluTilingData* tilingData)
 	 {
 	     blockLength_ = tilingData->blockFactor;
 	     ubLength_ = tilingData->ubFactor;
 	 
 	     // 参数合法性检查
 	     if (ubLength_ <= 0)
 	     {
 	         ubLength_ = blockLength_;
 	     }
 	     if (ubLength_ > blockLength_)
 	     {
 	         ubLength_ = blockLength_;
 	     }
 	 
 	     constexpr int64_t maxTileElems = 4096;
 	     if (ubLength_ > maxTileElems)
 	     {
 	         ubLength_ = maxTileElems;
 	     }
 	 
 	     if (ubLength_ <= 0)
 	     {
 	         ubLength_ = 1;
 	     }
 	 
 	     // 获取当前核对应的偏移量
 	     int64_t blockIdx;
 	     blockIdx = static_cast<int64_t>(GetBlockIdx());
 	     const int64_t gmOffset = blockLength_ * blockIdx;
 	 
 	     inputGMX.SetGlobalBuffer((__gm__ T*)x + gmOffset, blockLength_);
 	     outputGMY.SetGlobalBuffer((__gm__ T*)y + gmOffset, blockLength_);
 	 
 	     // 输入队列缓冲区大小
 	     int64_t inputByteNum;
 	     inputByteNum = ubLength_ * static_cast<int64_t>(sizeof(T));
 	     uint32_t inputBufSize = static_cast<uint32_t>(inputByteNum);
 	 
 	     pipe.InitBuffer(inputQueueX, BUFFER_NUM, inputBufSize);
 	 
 	     // 输出队列缓冲区大小
 	     int64_t outputByteNum;
 	     outputByteNum = ubLength_ * static_cast<int64_t>(sizeof(T));
 	     uint32_t outputBufSize = static_cast<uint32_t>(outputByteNum);
 	 
 	     pipe.InitBuffer(outputQueueY, BUFFER_NUM, outputBufSize);
 	 }
 	 
 	 template <typename T>
 	 __aicore__ inline void Relu<T>::CopyIn(int64_t progress, int64_t currentNum)
 	 {
 	     LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
 	 
 	     uint32_t copyLen;
 	     copyLen = static_cast<uint32_t>(currentNum);
 	 
 	     AscendC::DataCopy(xLocal, inputGMX[progress], copyLen);
 	 
 	     inputQueueX.EnQue(xLocal);
 	 }
 	 
 	 template <typename T>
 	 __aicore__ inline void Relu<T>::Compute(int64_t currentNum)
 	 {
 	     LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
 	     LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();
 	 
 	     uint32_t computeLen;
 	     computeLen = static_cast<uint32_t>(currentNum);
 	 
 	     AscendC::Relu(yLocal, xLocal, computeLen);
 	 
 	     outputQueueY.EnQue(yLocal);
 	     inputQueueX.FreeTensor(xLocal);
 	 }
 	 
 	 template <typename T>
 	 __aicore__ inline void Relu<T>::CopyOut(int64_t progress, int64_t currentNum)
 	 {
 	     LocalTensor<T> yLocal = outputQueueY.DeQue<T>();
 	 
 	     uint32_t copyLen;
 	     copyLen = static_cast<uint32_t>(currentNum);
 	 
 	     AscendC::DataCopy(outputGMY[progress], yLocal, copyLen);
 	 
 	     outputQueueY.FreeTensor(yLocal);
 	 }
 	 
 	 template <typename T>
 	 __aicore__ inline void Relu<T>::Process()
 	 {
 	     int64_t progress;
 	     progress = 0;
 	 
 	     while (progress < blockLength_)
 	     {
 	         int64_t currentNum;
 	         currentNum = blockLength_ - progress;
 	 
 	         if (currentNum > ubLength_)
 	         {
 	             currentNum = ubLength_;
 	         }
 	 
 	         CopyIn(progress, currentNum);
 	         Compute(currentNum);
 	         CopyOut(progress, currentNum);
 	 
 	         progress = progress + currentNum;
 	     }
 	 }
 	 
 	 } // namespace NsRelu
 	 
 	 #endif // RELU_H