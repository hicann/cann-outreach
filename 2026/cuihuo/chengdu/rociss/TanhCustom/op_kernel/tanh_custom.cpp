#include "kernel_operator.h"
#include "tanh_custom_tiling.h"

constexpr int32_t BUFFER_NUM = 2; // tensor num for each queue
using namespace AscendC;
class KernelTanh {
public:
    __aicore__ inline KernelTanh() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength, uint32_t tileNum)
    {
        // TODO: 考生自行补齐初始化函数
          this->blockLength = totalLength / GetBlockNum();
        this->tileNum = tileNum;
        ASSERT(totalLength >= (this->blockLength * GetBlockNum()));
        ASSERT((this->blockLength % tileNum) == 0);
        this->tileLength = this->blockLength / tileNum / BUFFER_NUM;

        xGm.SetGlobalBuffer((__gm__ DTYPE_X*)x + this->blockLength * GetBlockIdx(), this->blockLength);
        yGm.SetGlobalBuffer((__gm__ DTYPE_Y*)y + this->blockLength * GetBlockIdx(), this->blockLength);
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
         LocalTensor<DTYPE_X> xLocal = inQueueX.AllocTensor<DTYPE_X>();
        DataCopy(xLocal, xGm[progress * this->tileLength], this->tileLength);
        inQueueX.EnQue(xLocal);

    }
    __aicore__ inline void Compute(int32_t progress)
    {
        // TODO: 考生自行补齐
         LocalTensor<DTYPE_X> xLocal = inQueueX.DeQue<DTYPE_X>();
        LocalTensor<DTYPE_Y> yLocal = outQueueY.AllocTensor<DTYPE_Y>();
        // LocalTensor<DTYPE_X> tmpTensor1 = tmpBuf0.Get<DTYPE_X>();
        // LocalTensor<DTYPE_X> tmpTensor2 = tmpBuf1.Get<DTYPE_X>();
        // LocalTensor<DTYPE_X> tmpTensor3 = tmpBuf2.Get<DTYPE_X>();
          // 中间计算使用fp32：fp16下e^(2x)最大~403（ulp=0.25），误差会超出1e-3比对阈值
        LocalTensor<float> tmpTensor1 = tmpBuf0.Get<float>();
        LocalTensor<float> tmpTensor2 = tmpBuf1.Get<float>();
        LocalTensor<float> tmpTensor3 = tmpBuf2.Get<float>();

        // tanh(x) = (e^(2x) - 1) / (e^(2x) + 1)
        // AscendC::Muls(tmpTensor1, xLocal, (DTYPE_X)2.0, this->tileLength);       // tmp1 = 2x
        // AscendC::Exp(tmpTensor1, tmpTensor1, this->tileLength);                  // tmp1 = e^(2x)
        // AscendC::Adds(tmpTensor2, tmpTensor1, (DTYPE_X)1.0, this->tileLength);   // tmp2 = e^(2x) + 1
        // AscendC::Adds(tmpTensor3, tmpTensor1, (DTYPE_X)-1.0, this->tileLength);   // tmp3 = e^(2x) - 1
        // AscendC::Div(yLocal, tmpTensor3, tmpTensor2, this->tileLength);          // y = tmp3 / tmp2
          // tanh(x) = (e^(2x) - 1) / (e^(2x) + 1)
        AscendC::Cast(tmpTensor1, xLocal, AscendC::RoundMode::CAST_NONE, this->tileLength); // tmp1 = x (fp32)
        AscendC::Muls(tmpTensor2, tmpTensor1, (float)2.0, this->tileLength);               // tmp2 = 2x
        AscendC::Exp(tmpTensor2, tmpTensor2, this->tileLength);                            // tmp2 = e^(2x)
        AscendC::Adds(tmpTensor3, tmpTensor2, (float)1.0, this->tileLength);               // tmp3 = e^(2x) + 1
        AscendC::Adds(tmpTensor1, tmpTensor2, (float)-1.0, this->tileLength);              // tmp1 = e^(2x) - 1（无Subs，加-1等价）
        AscendC::Div(tmpTensor2, tmpTensor1, tmpTensor3, this->tileLength);                // tmp2 = tmp1 / tmp3
        AscendC::Cast(yLocal, tmpTensor2, AscendC::RoundMode::CAST_NONE, this->tileLength); // y = fp16
        outQueueY.EnQue<DTYPE_Y>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        // TODO: 考生自行补齐
        LocalTensor<DTYPE_Y> yLocal = outQueueY.DeQue<DTYPE_Y>();
        DataCopy(yGm[progress * this->tileLength], yLocal, this->tileLength);
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