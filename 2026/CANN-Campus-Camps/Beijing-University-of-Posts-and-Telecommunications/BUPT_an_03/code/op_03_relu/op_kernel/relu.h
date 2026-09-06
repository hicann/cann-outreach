/*!
 * \file relu.h
 * \brief Relu 绠楀瓙 kernel 绫诲畾涔?
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

// 璁惧渚ф暣闄ゅ悜涓婂彇鏁达紙int64_t锛?
__aicore__ inline int64_t ReluCeilDiv(int64_t a, int64_t b)
{
    return (a + b - 1) / b;
}

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

    int64_t blockLength_ = 0;   // 鏈牳瀹為檯澶勭悊鐨勫厓绱犳暟锛堝熬鏍稿彲鑳藉皬浜?blockFactor锛?
    int64_t ubLength_ = 0;      // 姣忚疆杩?UB 鐨勫厓绱犳暟
    int64_t totalNum_ = 0;      // 鏁版嵁鎬婚暱搴?
    int64_t roundCount_ = 0;    // 寰幆杞暟 = ceil(blockLength_ / ubLength_)锛岀┖闂叉牳涓?0
};

// Init锛氳В鏋?tiling + 缁戝畾鏈牳 GM 绐楀彛 + 鐢虫姤 UB 缂撳啿
template <typename T>
__aicore__ inline void Relu<T>::Init(GM_ADDR x, GM_ADDR y, const ReluTilingData* tilingData)
{
    totalNum_ = tilingData->totalNum;
    ubLength_ = tilingData->ubFactor;

    // 鏍搁棿绐楀彛锛氭湰鏍歌礋璐?[blockIdx * blockFactor, +blockFactor)
    int64_t startOffset = static_cast<int64_t>(GetBlockIdx()) * tilingData->blockFactor;
    if (startOffset >= totalNum_) {
        // 鏁版嵁宸茶鍓嶉潰鐨勬牳瑕嗙洊瀹岋紝鏈牳绌洪棽
        roundCount_ = 0;
        return;
    }
    // 灏炬牳瀹為檯澶勭悊閲忓彲鑳藉皬浜?blockFactor
    int64_t remain = totalNum_ - startOffset;
    blockLength_ = (remain < tilingData->blockFactor) ? remain : tilingData->blockFactor;

    inputGMX.SetGlobalBuffer((__gm__ T *)x + startOffset, blockLength_);
    outputGMY.SetGlobalBuffer((__gm__ T *)y + startOffset, blockLength_);

    // UB 鍒囧垎鏂规锛氳緭鍏?杈撳嚭闃熷垪鍚?BUFFER_NUM 浠界紦鍐诧紙鍙岀紦鍐诧細
    // 绗?i+1 杞?MTE 鎼叆涓庣 i 杞?Maxs 璁＄畻閲嶅彔锛屾帺鐩栨惉杩愬欢杩燂級
    pipe.InitBuffer(inputQueueX, BUFFER_NUM, ubLength_ * sizeof(T));
    pipe.InitBuffer(outputQueueY, BUFFER_NUM, ubLength_ * sizeof(T));

    roundCount_ = ReluCeilDiv(blockLength_, ubLength_);
}

// CopyIn锛歁TE2 寮傛鎼叆绗?progress 杞紙currentNum 涓厓绱狅級鍒?UB
template <typename T>
__aicore__ inline void Relu<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();
    DataCopy(xLocal, inputGMX[progress * ubLength_], static_cast<uint32_t>(currentNum));
    inputQueueX.EnQue(xLocal);
}

// Compute锛歽 = max(x, 0)锛孷ector 鍗曞厓 128-lane 鎸囦护鎸?repeat 灞曞紑
// 浣跨敤 AscendC 涓撶敤楂橀樁 API Relu(dst, src, count)锛屽 half/float 涓ょ簿搴﹀潎鏈夐噸杞?
template <typename T>
__aicore__ inline void Relu<T>::Compute(int64_t currentNum)
{
    LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
    LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();
    AscendC::Relu(yLocal, xLocal, static_cast<uint32_t>(currentNum));
    outputQueueY.EnQue<T>(yLocal);
    inputQueueX.FreeTensor(xLocal);
}

// CopyOut锛歁TE3 寮傛鎼嚭绗?progress 杞粨鏋滃埌 GM
template <typename T>
__aicore__ inline void Relu<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    LocalTensor<T> yLocal = outputQueueY.DeQue<T>();
    DataCopy(outputGMY[progress * ubLength_], yLocal, static_cast<uint32_t>(currentNum));
    outputQueueY.FreeTensor(yLocal);
}

// Process锛氭祦姘寸嚎涓诲惊鐜€傚墠鑻ュ共杞弧 ubLength_锛屽熬杞寜鍓╀綑閲忥紙瀵归綈鏈変繚璇侊細
// blockFactor 涓?ubFactor 鍧囦负 32B 鍏冪礌鏁扮殑鍊嶆暟锛屽熬杞?currentNum 浜︾劧锛?
template <typename T>
__aicore__ inline void Relu<T>::Process()
{
    if (roundCount_ == 0) {
        return;
    }
    for (int64_t i = 0; i < roundCount_; i++) {
        int64_t currentNum = (i == roundCount_ - 1)
                                 ? (blockLength_ - (roundCount_ - 1) * ubLength_)
                                 : ubLength_;
        CopyIn(i, currentNum);    // MTE2锛欻BM 鈫?UB
        Compute(currentNum);      // Vector锛歎B 鍐?max(x, 0)
        CopyOut(i, currentNum);   // MTE3锛歎B 鈫?HBM
    }
}

} // namespace NsRelu
#endif // RELU_H