#include "kernel_operator.h"
#include "tanh_custom_tiling.h"

// 每个 Queue 的 buffer 数量，采用 double buffer（乒乓）机制。
constexpr int32_t BUFFER_NUM = 2;

class KernelTanh {
public:
    __aicore__ inline KernelTanh() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t blockLength, uint32_t tileNum)
    {
        // 此处入参 blockLength 即单个核需要处理的元素总个数（由 Host 侧 Tiling 计算得出）。
        this->blockLength = blockLength;
        this->tileNum = tileNum;
        // 单核内每个 tile 处理的元素数 = 单核总元素数 / (tile 数 * 双缓冲数)
        this->tileLength = blockLength / (tileNum * BUFFER_NUM);

        // 每个核按其 BlockIdx 偏移，访问全局内存中属于自己的数据区间。
        xGm.SetGlobalBuffer((__gm__ DTYPE_X *)x + AscendC::GetBlockIdx() * this->blockLength,
                             this->blockLength);
        yGm.SetGlobalBuffer((__gm__ DTYPE_Y *)y + AscendC::GetBlockIdx() * this->blockLength,
                             this->blockLength);

        // 为输入/输出队列及临时计算 buffer 申请 UB 空间。
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(DTYPE_X));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileLength * sizeof(DTYPE_Y));
        pipe.InitBuffer(tmpBuf0, this->tileLength * sizeof(DTYPE_X));
        pipe.InitBuffer(tmpBuf1, this->tileLength * sizeof(DTYPE_X));
        pipe.InitBuffer(tmpBuf2, this->tileLength * sizeof(DTYPE_X));
    }

    __aicore__ inline void Process()
    {
        // 双缓冲场景下，单核需循环 tileNum * BUFFER_NUM 次，每次处理 tileLength 个元素。
        const int32_t loopCount = static_cast<int32_t>(this->tileNum * BUFFER_NUM);
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.AllocTensor<DTYPE_X>();
        AscendC::DataCopy(xLocal, xGm[progress * tileLength], tileLength);
        inQueueX.EnQue<DTYPE_X>(xLocal);
    }

    __aicore__ inline void Compute([[maybe_unused]] int32_t progress)
    {
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.DeQue<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_Y> yLocal = outQueueY.AllocTensor<DTYPE_Y>();

        AscendC::LocalTensor<DTYPE_X> expX = tmpBuf0.Get<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_X> expNX = tmpBuf1.Get<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_X> tmp = tmpBuf2.Get<DTYPE_X>();

        // 计算 tanh(x) = (e^x - e^-x) / (e^x + e^-x)
        AscendC::Muls(tmp, xLocal, (DTYPE_X)(-1.0f), tileLength);   // tmp  = -x
        AscendC::Exp(expX, xLocal, tileLength);                     // expX = e^x
        AscendC::Exp(expNX, tmp, tileLength);                       // expNX = e^-x
        AscendC::Sub(tmp, expX, expNX, tileLength);                 // tmp  = e^x - e^-x（复用 tmp 作分子）
        AscendC::Add(expX, expX, expNX, tileLength);                // expX = e^x + e^-x（复用 expX 作分母）
        AscendC::Div(yLocal, tmp, expX, tileLength);                // y    = (e^x - e^-x) / (e^x + e^-x)

        outQueueY.EnQue<DTYPE_Y>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<DTYPE_Y> yLocal = outQueueY.DeQue<DTYPE_Y>();
        AscendC::DataCopy(yGm[progress * tileLength], yLocal, tileLength);
        outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmpBuf0, tmpBuf1, tmpBuf2;
    AscendC::GlobalTensor<DTYPE_X> xGm;
    AscendC::GlobalTensor<DTYPE_Y> yGm;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
};

extern "C" __global__ __aicore__ void tanh_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(TanhCustomTilingData);
    GET_TILING_DATA(tilingData, tiling);
    KernelTanh op;
    op.Init(x, y, tilingData.blockLength, tilingData.tileNum);
    op.Process();
}
