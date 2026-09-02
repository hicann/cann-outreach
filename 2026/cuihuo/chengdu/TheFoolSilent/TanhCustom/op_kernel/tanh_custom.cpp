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


        // 每个核处理的数据长度
        this->blockLength =
            totalLength / AscendC::GetBlockNum();


        // 每个tile长度
        this->tileLength =
            this->blockLength / tileNum;


        uint32_t blockOffset =
            AscendC::GetBlockIdx() * blockLength;


        xGm.SetGlobalBuffer(
            (__gm__ DTYPE_X *)x + blockOffset,
            blockLength
        );


        yGm.SetGlobalBuffer(
            (__gm__ DTYPE_Y *)y + blockOffset,
            blockLength
        );


        // 输入buffer
        pipe.InitBuffer(
            inQueueX,
            BUFFER_NUM,
            tileLength * sizeof(DTYPE_X)
        );


        // 输出buffer
        pipe.InitBuffer(
            outQueueY,
            BUFFER_NUM,
            tileLength * sizeof(DTYPE_Y)
        );


        // FP32中间buffer
        pipe.InitBuffer(
            tmpBuf0,
            tileLength * sizeof(float)
        );

        pipe.InitBuffer(
            tmpBuf1,
            tileLength * sizeof(float)
        );

        pipe.InitBuffer(
            tmpBuf2,
            tileLength * sizeof(float)
        );
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
        auto xLocal =
        inQueueX.AllocTensor<DTYPE_X>();


    uint32_t tileIndex =
        progress / BUFFER_NUM;


    uint32_t offset =
        tileIndex * tileLength;


    AscendC::DataCopy(
        xLocal,
        xGm[offset],
        tileLength
    );


    inQueueX.EnQue(xLocal);

    }
    __aicore__ inline void Compute(int32_t progress)
    {
        // TODO: 考生自行补齐
        auto xLocal =
            inQueueX.DeQue<DTYPE_X>();


        auto yLocal =
            outQueueY.AllocTensor<DTYPE_Y>();



        // 临时buffer
        auto tmp0 =
            tmpBuf0.Get<float>();

        auto tmp1 =
            tmpBuf1.Get<float>();

        auto tmp2 =
            tmpBuf2.Get<float>();

        AscendC::Cast(
    tmp0,
    xLocal,
    AscendC::RoundMode::CAST_NONE,
    tileLength
);


// tmp1 = exp(x)
AscendC::Exp(
    tmp1,
    tmp0,
    tileLength
);


// tmp2 = -x
AscendC::Muls(
    tmp2,
    tmp0,
    -1.0f,
    tileLength
);


// tmp2 = exp(-x)
AscendC::Exp(
    tmp2,
    tmp2,
    tileLength
);


// tmp0 = exp(x)+exp(-x)
// denominator
AscendC::Add(
    tmp0,
    tmp1,
    tmp2,
    tileLength
);


// tmp1 = exp(x)-exp(-x)
// numerator
AscendC::Sub(
    tmp1,
    tmp1,
    tmp2,
    tileLength
);


// tmp1 = numerator / denominator
AscendC::Div(
    tmp1,
    tmp1,
    tmp0,
    tileLength
);


// FP32 -> FP16
AscendC::Cast(
    yLocal,
    tmp1,
    AscendC::RoundMode::CAST_RINT,
    tileLength
);


        outQueueY.EnQue(yLocal);


        inQueueX.FreeTensor(xLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        // TODO: 考生自行补齐
        auto yLocal =
            outQueueY.DeQue<DTYPE_Y>();


        uint32_t tileIndex =
    progress / BUFFER_NUM;

        uint32_t offset =
    tileIndex * tileLength;

        AscendC::DataCopy(
            yGm[offset],
            yLocal,
            tileLength
        );


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
   // 创建算子对象
    KernelTanh op;

    // 初始化
    op.Init(
        x,
        y,
        tilingData.totalLength,
        tilingData.tileNum
    );


    // 执行
    op.Process();
}