#include "kernel_operator.h"
#include "tanh_custom_tiling.h"

constexpr int32_t BUFFER_NUM = 2; // tensor num for each queue

class KernelTanh {
public:
    __aicore__ inline KernelTanh() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength, uint32_t tileNum)
    {
        // TODO: 考生自行补齐初始化函数
        this->blockLength = totalLength;
        this->tileNum = tileNum;
        this->tileLength = totalLength / (tileNum * BUFFER_NUM);

        xGm.SetGlobalBuffer( (__gm__ DTYPE_X*)x, totalLength );
        yGm.SetGlobalBuffer( (__gm__ DTYPE_Y*)y, totalLength );

        pipe.InitBuffer( inQueueX, BUFFER_NUM, this->tileLength * sizeof(DTYPE_X) );
        pipe.InitBuffer( outQueueY, BUFFER_NUM, this->tileLength * sizeof(DTYPE_Y) );
        pipe.InitBuffer( tmpBuf0, this->tileLength * sizeof(float) );
        pipe.InitBuffer( tmpBuf1, this->tileLength * sizeof(float) );
        pipe.InitBuffer( tmpBuf2, this->tileLength * sizeof(float) );
        pipe.InitBuffer( tmpBuf3, this->tileLength * sizeof(float) );
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
        auto xLocal = inQueueX.AllocTensor<DTYPE_X>();
        AscendC::DataCopy( xLocal, xGm[progress * tileLength], tileLength );
        inQueueX.EnQue(xLocal);
    }
    __aicore__ inline void Compute(int32_t progress)
    {
        // TODO: 考生自行补齐
        auto xLocal = inQueueX.DeQue<DTYPE_X>();
        auto yLocal = outQueueY.AllocTensor<DTYPE_Y>();
        auto xFloat = tmpBuf3.Get<float>();
        auto expX = tmpBuf0.Get<float>();
        auto expNegX = tmpBuf1.Get<float>();
        auto result = tmpBuf2.Get<float>();

        AscendC::Cast(
            xFloat,
            xLocal,
            AscendC::RoundMode::CAST_NONE,
            tileLength
        );

        AscendC::Exp(
            expX,
            xFloat,
            tileLength
        );

        AscendC::Muls(
            expNegX,
            xFloat,
            -1.0f,
            tileLength
        );

        AscendC::Exp(
            expNegX,
            expNegX,
            tileLength
        );

        AscendC::Sub(
            result,
            expX,
            expNegX,
            tileLength
        );

        AscendC::Add(
            expX,
            expX,
            expNegX,
            tileLength
        );

        AscendC::Div(
            result,
            result,
            expX,
            tileLength
        );

        AscendC::Cast(
            yLocal,
            result,
            AscendC::RoundMode::CAST_NONE,
            tileLength
        );
    
        outQueueY.EnQue(yLocal);
        inQueueX.FreeTensor(xLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        // TODO: 考生自行补齐
        auto yLocal = outQueueY.DeQue<DTYPE_Y>();
        AscendC::DataCopy( yGm[progress * tileLength], yLocal, tileLength );
        outQueueY.FreeTensor(yLocal);    
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmpBuf0,tmpBuf1,tmpBuf2,tmpBuf3;
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
    op.Init( x, y, tilingData.totalLength, tilingData.tileNum );
    op.Process();
}