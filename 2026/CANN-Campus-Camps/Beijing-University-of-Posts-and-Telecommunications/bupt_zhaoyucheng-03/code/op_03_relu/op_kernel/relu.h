/*!
 * \file relu.h
 * \brief Relu 算子 kernel 类定义（手工事件流水版）
 */

#ifndef RELU_H
#define RELU_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "relu_tiling_data.h"

namespace NsRelu {

using namespace AscendC;

template <typename DT_X>
class Relu {
public:
    __aicore__ inline Relu() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, const ReluTilingData* tilingData)
    {
        // 每核处理的总数据长度（多核均分）
        this->blockLength = tilingData->totalLength / AscendC::GetBlockNum();
        // 总轮数：fp32=2（双缓冲流水），fp16=1（单块大 burst）
        this->loopTotal = tilingData->tileNum;
        this->tileLength = this->blockLength / this->loopTotal;
        // 设置每个核的 Global Memory 起始地址（多核处理的数据段互不重叠）
        xGm.SetGlobalBuffer((__gm__ DT_X*)x + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        yGm.SetGlobalBuffer((__gm__ DT_X*)y + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        // x/y 各双槽轮转（单块时仅用槽 0）
        pipe.InitBuffer(xBuf[0], this->tileLength * sizeof(DT_X));
        pipe.InitBuffer(xBuf[1], this->tileLength * sizeof(DT_X));
        pipe.InitBuffer(yBuf[0], this->tileLength * sizeof(DT_X));
        pipe.InitBuffer(yBuf[1], this->tileLength * sizeof(DT_X));
    }

    __aicore__ inline void Process()
    {
        // 手工事件三级流水：MTE2(GM->UB) -> V(向量计算) -> MTE3(UB->GM)
        // 预取第 0 块
        LocalTensor<DT_X> x0 = xBuf[0].Get<DT_X>();
        DataCopy(x0, xGm[0], this->tileLength);
        SetFlag<HardEvent::MTE2_V>(0);
        for (int32_t i = 0; i < this->loopTotal; i++) {
            int32_t cur = i & 1;
            WaitFlag<HardEvent::MTE2_V>(cur);
            if (i + 1 < this->loopTotal) {
                // 当前块计算期间，提前发起下一块搬运（MTE2 与 V 并行）
                LocalTensor<DT_X> xNext = xBuf[cur ^ 1].Get<DT_X>();
                DataCopy(xNext, xGm[(i + 1) * this->tileLength], this->tileLength);
                SetFlag<HardEvent::MTE2_V>(cur ^ 1);
            }
            LocalTensor<DT_X> xCur = xBuf[cur].Get<DT_X>();
            LocalTensor<DT_X> yCur = yBuf[cur].Get<DT_X>();
            // y = max(0, x)，逐元素与标量 0 取最大值
            AscendC::Maxs(yCur, xCur, (DT_X)0, this->tileLength);
            SetFlag<HardEvent::V_MTE3>(cur);
            WaitFlag<HardEvent::V_MTE3>(cur);
            DataCopy(yGm[i * this->tileLength], yCur, this->tileLength);
        }
    }

private:
    TPipe pipe; // TPipe 内存管理对象
    TBuf<QuePosition::VECIN> xBuf[2];   // 输入 buffer，双槽轮转
    TBuf<QuePosition::VECOUT> yBuf[2];  // 输出 buffer，双槽轮转
    GlobalTensor<DT_X> xGm; // 输入 Global Memory 管理对象
    GlobalTensor<DT_X> yGm; // 输出 Global Memory 管理对象

    uint32_t blockLength; // 每核处理元素数
    uint32_t loopTotal;   // 总轮数
    uint32_t tileLength;  // 每块元素数
};

} // namespace NsRelu
#endif // RELU_H
