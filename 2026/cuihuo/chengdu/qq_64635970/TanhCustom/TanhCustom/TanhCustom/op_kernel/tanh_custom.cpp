#include "kernel_operator.h"
#include "tanh_custom_tiling.h"

constexpr int32_t BUFFER_NUM = 2; // tensor num for each queue

class KernelTanh {
public:
    __aicore__ inline KernelTanh() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength, uint32_t tileNum)
    {
        // TODO: 考生自行补齐初始化函数
        this->tileNum = tileNum;
        this->tileLength = totalLength / tileNum + 1;
        this->blockLength = totalLength;

        xGm.SetGlobalBuffer((__gm__ DTYPE_X*)x, totalLength);
        yGm.SetGlobalBuffer((__gm__ DTYPE_Y*)y, totalLength);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, tileLength * sizeof(DTYPE_X));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, tileLength * sizeof(DTYPE_Y));
        pipe.InitBuffer(tmpBuf0, tileLength * sizeof(DTYPE_X));
        pipe.InitBuffer(tmpBuf1, tileLength * sizeof(DTYPE_X));
        pipe.InitBuffer(tmpBuf2, tileLength * sizeof(DTYPE_X));
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
        int32_t tileIdx = progress / BUFFER_NUM;
        uint32_t startOffset = tileIdx * tileLength;
        uint32_t copyLength = tileLength;

        if (startOffset + copyLength > blockLength) {
            copyLength = blockLength - startOffset;
        }

        auto localX = inQueueX.AllocTensor<DTYPE_X>();
        DataCopy(localX, xGm[startOffset], copyLength);
        inQueueX.EnQue(localX);
    }
    
    __aicore__ inline void Compute(int32_t progress)
    {
        // TODO: 考生自行补齐
        auto localX = inQueueX.DeQue<DTYPE_X>();

        auto tmpX = tmpBuf0.Get<DTYPE_X>();
        auto tmpExpPos = tmpBuf1.Get<DTYPE_X>();
        auto tmpExpNeg = tmpBuf2.Get<DTYPE_X>();
        auto tmpNegX = tmpBuf0.Get<DTYPE_X>();
        
        DataCopy(tmpX, localX, tileLength);

        // 1. 计算 e^x
        Exp(tmpExpPos, tmpX, tileLength);

        // 2. 计算 e^(-x)：先取负，再求exp
        using AscendC::Muls;
        Muls(tmpNegX, tmpX, (DTYPE_X)(-1.0f), tileLength);
        Exp(tmpExpNeg, tmpNegX, tileLength);

        // 3. 计算 e^x - e^(-x)
        auto numerator = tmpBuf0.Get<DTYPE_X>();
        using AscendC::Sub;
        Sub(numerator, tmpExpPos, tmpExpNeg, tileLength);

        // 4. 计算 e^x + e^(-x)
        auto denominator = tmpBuf1.Get<DTYPE_X>();
        using AscendC::Add;
        Add(denominator, tmpExpPos, tmpExpNeg, tileLength);

        // 5. 计算 tanh = (e^x - e^(-x)) / (e^x + e^(-x))
        auto result = tmpBuf0.Get<DTYPE_X>();
        using AscendC::Div;
        Div(result, numerator, denominator, tileLength);

        // 6. 输出结果
        auto localY = outQueueY.AllocTensor<DTYPE_Y>();
        // <<<<<<< 修复: DTYPE_X 和 DTYPE_Y 都是 half，不需要 Cast，直接用 DataCopy
        DataCopy(localY, result, tileLength);  // 原来: Cast(localY, result, AscendC::RoundMode::CAST_NONE, tileLength);
        // >>>>>>>
        outQueueY.EnQue(localY);

        inQueueX.FreeTensor(localX);
    }
    
    __aicore__ inline void CopyOut(int32_t progress)
    {
        // TODO: 考生自行补齐
        int32_t tileIdx = progress / BUFFER_NUM;
        uint32_t startOffset = tileIdx * tileLength;
        uint32_t copyLength = tileLength;

        if (startOffset + copyLength > blockLength) {
            copyLength = blockLength - startOffset;
        }

        auto localY = outQueueY.DeQue<DTYPE_Y>();
        DataCopy(yGm[startOffset], localY, copyLength);
        outQueueY.FreeTensor(localY);
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
    uint32_t totalLength = tilingData.totalNum;
    uint32_t tileNum = tilingData.blockNum;
    
    KernelTanh op;
    op.Init(x, y, totalLength, tileNum);
    op.Process();
}
