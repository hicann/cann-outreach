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
    /*
     * 褰撳墠 Core 璐熻矗鐨勬暟鎹噺銆?
     */
    blockLength_ = tilingData->blockFactor;

    /*
     * 鍗曟 UB 澶勭悊鐨勬暟鎹噺銆?
     */
    ubLength_ = tilingData->ubFactor;

    /*
     * Kernel 渚т繚鎶ゃ€?
     *
     * 鍘熷伐绋嬭嚜甯︾殑 kernel UT 浼氭墜宸ユ瀯閫狅細
     *
     * ubFactor = totalNum
     *
     * 濡傛灉鐩存帴鎸夌収璇ュ€肩敵璇?UB锛屼細杩滃ぇ浜庢甯?UB銆?
     *
     * Host 姝ｅ父鎵ц鏃?ubFactor 宸茬粡鏄悎娉曞€硷紝
     * 杩欓噷涓昏鐢ㄤ簬淇濊瘉 Kernel 鑷韩瓒冲鍋ュ．銆?
     */
    if (ubLength_ <= 0 || ubLength_ > blockLength_) {
        ubLength_ = blockLength_;
    }

    constexpr int64_t maxTileElems = 4096;

    if (ubLength_ > maxTileElems) {
        ubLength_ = maxTileElems;
    }

    if (ubLength_ <= 0) {
        ubLength_ = 1;
    }

    /*
     * 褰撳墠 Core 鍦?GM 涓殑璧峰浣嶇疆銆?
     */
    const int64_t gmOffset = blockLength_ * static_cast<int64_t>(GetBlockIdx());

    inputGMX.SetGlobalBuffer((__gm__ T*)x + gmOffset, blockLength_);

    outputGMY.SetGlobalBuffer((__gm__ T*)y + gmOffset, blockLength_);

    /*
     * Input / Output 鍧囦娇鐢?Double Buffer銆?
     */
    pipe.InitBuffer(
        inputQueueX,
        BUFFER_NUM,
        static_cast<uint32_t>(ubLength_ * static_cast<int64_t>(sizeof(T))));

    pipe.InitBuffer(
        outputQueueY,
        BUFFER_NUM,
        static_cast<uint32_t>(ubLength_ * static_cast<int64_t>(sizeof(T))));
}

template <typename T>
__aicore__ inline void Relu<T>::CopyIn(int64_t progress, int64_t currentNum)
{
    /*
     * 浠?VECIN Queue 鐢宠 LocalTensor銆?
     */
    LocalTensor<T> xLocal = inputQueueX.AllocTensor<T>();

    /*
     * GM -> UB
     */
    AscendC::DataCopy(
        xLocal,
        inputGMX[progress],
        static_cast<uint32_t>(currentNum));

    /*
     * 鏀惧叆杈撳叆 Queue锛岀瓑寰?Compute銆?
     */
    inputQueueX.EnQue(xLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::Compute(int64_t currentNum)
{
    /*
     * 鑾峰彇杈撳叆 Tensor銆?
     */
    LocalTensor<T> xLocal = inputQueueX.DeQue<T>();

    /*
     * 涓鸿緭鍑虹敵璇?Tensor銆?
     */
    LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();

    /*
     * y = max(0, x)
     *
     * 鏄惧紡鍐?AscendC::Relu锛?
     * 閬垮厤鍜屽綋鍓?NsRelu::Relu 绫绘ā鏉垮悓鍚嶅啿绐併€?
     */
    AscendC::Relu(
        yLocal,
        xLocal,
        static_cast<uint32_t>(currentNum));

    /*
     * 杈撳嚭杩涘叆 VECOUT Queue銆?
     */
    outputQueueY.EnQue(yLocal);

    /*
     * 杈撳叆 Tensor 宸茬粡浣跨敤瀹屾瘯銆?
     */
    inputQueueX.FreeTensor(xLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::CopyOut(int64_t progress, int64_t currentNum)
{
    /*
     * 浠庤緭鍑?Queue 鑾峰彇璁＄畻缁撴灉銆?
     */
    LocalTensor<T> yLocal = outputQueueY.DeQue<T>();

    /*
     * UB -> GM
     */
    AscendC::DataCopy(
        outputGMY[progress],
        yLocal,
        static_cast<uint32_t>(currentNum));

    /*
     * 閲婃斁杈撳嚭 Tensor銆?
     */
    outputQueueY.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void Relu<T>::Process()
{
    /*
     * 褰撳墠 Core 鍐呴儴缁х画鎸夌収 ubLength_ 鍒?Tile銆?
     */
    int64_t progress = 0;

    while (progress < blockLength_) {
        int64_t currentNum = blockLength_ - progress;

        if (currentNum > ubLength_) {
            currentNum = ubLength_;
        }

        CopyIn(progress, currentNum);

        Compute(currentNum);

        CopyOut(progress, currentNum);

        progress += currentNum;
    }
}

} // namespace NsRelu

#endif // RELU_H