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
    int64_t totalNum_ = 0;     // 总元素数量
    int64_t blockLength_ = 0;  // 每个核处理的元素数量
    int64_t ubLength_ = 0;     // 每次 UB 循环处理的元素数量
    int64_t coreLength_ = 0;   // 当前核实际处理的元素数量（尾部钳制）
    int64_t loopNum_ = 0;      // 当前核的循环段数
};
// 初始化：多核切分 + UB 缓冲分配
template <typename T>
__aicore__ inline void Relu<T>::Init(GM_ADDR x, GM_ADDR y, const ReluTilingData* tilingData)
{
    this->totalNum_ = tilingData->totalNum;
    this->blockLength_ = tilingData->blockFactor;
    this->ubLength_ = tilingData->ubFactor;
    if (this->ubLength_ < 1) {
        this->ubLength_ = 1;
    }

    int64_t coreIdx = GetBlockIdx();
    int64_t offset = coreIdx * this->blockLength_;
    this->coreLength_ = (offset >= this->totalNum_)
                            ? 0
                            : ((this->totalNum_ - offset) < this->blockLength_
                                   ? (this->totalNum_ - offset)
                                   : this->blockLength_);
    this->loopNum_ = (this->coreLength_ + this->ubLength_ - 1) / this->ubLength_;

    inputGMX.SetGlobalBuffer((__gm__ T*)x + offset, (int32_t)this->coreLength_);
    outputGMY.SetGlobalBuffer((__gm__ T*)y + offset, (int32_t)this->coreLength_);

    pipe.InitBuffer(inputQueueX, BUFFER_NUM, this->ubLength_ * sizeof(T));
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, this->ubLength_ * sizeof(T));
}
// 主流程：按 UB 段循环 CopyIn -> Compute -> CopyOut
template <typename T>
__aicore__ inline void Relu<T>::Process()
{
    for (int64_t progress = 0; progress < this->loopNum_; progress++) {
        int64_t remain = this->coreLength_ - progress * this->ubLength_;
        int64_t currentNum = (remain < this->ubLength_) ? remain : this->ubLength_;
        CopyIn(progress, currentNum);
        Compute(currentNum);
        CopyOut(progress, currentNum);
    }
}
// 将 Global Memory 数据搬入 UB 输入队列
template <typename T>
__aicore__ inline void Relu<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
    DataCopy(xLocal, inputGMX[progress * this->ubLength_], (int32_t)currentNum);
    inputQueueX.EnQue(xLocal);
}
// 矢量计算：y = Relu(x) = max(x, 0)
template <typename T>
__aicore__ inline void Relu<T>::Compute(int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();
    AscendC::Relu(yLocal, xLocal, (int32_t)currentNum);
    inputQueueX.FreeTensor(xLocal);
    outputQueueY.EnQue(yLocal);
}
// 将 UB 输出队列结果写回 Global Memory
template <typename T>
__aicore__ inline void Relu<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> yLocal = outputQueueY.DeQue<T>();
    DataCopy(outputGMY[progress * this->ubLength_], yLocal, (int32_t)currentNum);
    outputQueueY.FreeTensor(yLocal);
}
} // namespace NsRelu
#endif // RELU_H
