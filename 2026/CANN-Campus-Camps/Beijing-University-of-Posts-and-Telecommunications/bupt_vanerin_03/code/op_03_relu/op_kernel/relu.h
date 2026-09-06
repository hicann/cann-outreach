/*!
 * \file relu.h
 * \brief Relu 算子 kernel 类定义 — 裸UB + 显式同步
 */

#ifndef RELU_H
#define RELU_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "relu_tiling_data.h"
#include "relu_tiling_key.h"

namespace NsRelu {

using namespace AscendC;

template <typename T>
class Relu {
public:
    __aicore__ inline Relu(){};

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const ReluTilingData* tilingData)
    {
        const int64_t blockIdx = static_cast<int64_t>(GetBlockIdx());
        const int64_t remain = tilingData->totalNum - blockIdx * tilingData->blockFactor;
        blockLength_ = remain < tilingData->blockFactor ? remain : tilingData->blockFactor;

        const int64_t blockOffset = blockIdx * tilingData->blockFactor;
        inputGMX_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(x) + blockOffset, blockLength_);
        outputGMY_.SetGlobalBuffer(reinterpret_cast<__gm__ T*>(y) + blockOffset, blockLength_);

        // 只分配输入缓冲; 输出原地写回同一缓冲, 省掉独立输出缓冲与一次 InitBuffer
        pipe_.InitBuffer(bufX_, blockLength_ * sizeof(T));
    }

    __aicore__ inline void Process()
    {
        if (blockLength_ <= 0) {
            return;
        }
        LocalTensor<T> xLocal = bufX_.Get<T>();

        // 搬入: GM -> UB
        DataCopy(xLocal, inputGMX_, static_cast<uint32_t>(blockLength_));

        // 等搬入完成
        int32_t evtMte2 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        SetFlag<HardEvent::MTE2_V>(evtMte2);
        WaitFlag<HardEvent::MTE2_V>(evtMte2);

        // 计算: y = max(0, x), 结果原地写回 xLocal
        // 类名 Relu 遮蔽全局函数 AscendC::Relu, 必须显式限定
        AscendC::Relu(xLocal, xLocal, static_cast<int32_t>(blockLength_));

        // 等计算完成
        int32_t evtV = static_cast<int32_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        SetFlag<HardEvent::V_MTE3>(evtV);
        WaitFlag<HardEvent::V_MTE3>(evtV);

        // 搬出: UB -> GM
        DataCopy(outputGMY_, xLocal, static_cast<uint32_t>(blockLength_));
    }

private:
    TPipe pipe_;
    TBuf<TPosition::VECIN> bufX_;  // 输入 UB 缓冲(输出原地复用)
    GlobalTensor<T> inputGMX_;
    GlobalTensor<T> outputGMY_;
    int64_t blockLength_ = 0;
};

} // namespace NsRelu
#endif // RELU_H