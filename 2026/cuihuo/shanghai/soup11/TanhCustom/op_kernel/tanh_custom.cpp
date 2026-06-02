#include "kernel_operator.h"
#include "tanh_custom_tiling.h"

// 数据类型定义（若 tiling 头文件未定义，默认 half）
#ifndef DTYPE_X
#define DTYPE_X half
#endif
#ifndef DTYPE_Y
#define DTYPE_Y half
#endif

constexpr int32_t BUFFER_NUM = 2; // tensor num for each queue

class KernelTanh {
public:
    __aicore__ inline KernelTanh() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength, uint32_t tileNum)
    {
        this->totalLength = totalLength;
        this->tileNum     = tileNum;
        this->tileLength  = totalLength / tileNum;

        // 绑定全局内存
        xGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_X*>(x), totalLength);
        yGm.SetGlobalBuffer(reinterpret_cast<__gm__ DTYPE_Y*>(y), totalLength);

        // 初始化双缓冲队列与临时缓冲区
        pipe.InitBuffer(inQueueX,  BUFFER_NUM, tileLength * sizeof(DTYPE_X));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, tileLength * sizeof(DTYPE_Y));
        pipe.InitBuffer(tmpBuf0,  tileLength * sizeof(DTYPE_X));
        pipe.InitBuffer(tmpBuf1,  tileLength * sizeof(DTYPE_X));
        pipe.InitBuffer(tmpBuf2,  tileLength * sizeof(DTYPE_X));
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
        uint32_t tileIdx = progress % this->tileNum;
        uint32_t offset  = tileIdx * this->tileLength;

        // 从输入队列分配 LocalTensor，搬入数据，然后入队
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.AllocTensor<DTYPE_X>();
        DataCopy(xLocal, xGm[offset], tileLength);   // 需要指定拷贝个数
        inQueueX.EnQue(xLocal);                       // 入队后自动通知消费者
    }

    __aicore__ inline void Compute(int32_t progress)
    {
        // 取出输入数据（阻塞直到数据就绪）
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.DeQue<DTYPE_X>();

        // 分配输出空间
        AscendC::LocalTensor<DTYPE_Y> yLocal = outQueueY.AllocTensor<DTYPE_Y>();

        // 获取临时缓冲区
        AscendC::LocalTensor<DTYPE_X> expTensor  = tmpBuf0.Get<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_X> negTensor  = tmpBuf1.Get<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_X> denomTensor = tmpBuf2.Get<DTYPE_X>();

        // tanh(x) = (e^x - e^{-x}) / (e^x + e^{-x})
        AscendC::Exp(expTensor, xLocal, tileLength);                        // exp(x)
        AscendC::Muls(negTensor, xLocal, DTYPE_X(-1.0f), tileLength);       // -x
        AscendC::Exp(negTensor, negTensor, tileLength);                     // exp(-x)
        AscendC::Add(denomTensor, expTensor, negTensor, tileLength);        // exp(x)+exp(-x)
        AscendC::Sub(expTensor, expTensor, negTensor, tileLength);          // exp(x)-exp(-x)
        AscendC::Div(yLocal, expTensor, denomTensor, tileLength);           // 最终结果

        // 将计算结果入队，供 CopyOut 使用
        outQueueY.EnQue(yLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress)
    {
        // 取出计算结果（阻塞直到计算完成并入队）
        AscendC::LocalTensor<DTYPE_Y> yLocal = outQueueY.DeQue<DTYPE_Y>();

        uint32_t tileIdx = progress % this->tileNum;
        uint32_t offset  = tileIdx * this->tileLength;

        // 写回全局内存
        DataCopy(yGm[offset], yLocal, tileLength);

        // 归还队列空间（复用缓冲区）
        outQueueY.EnQue(yLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmpBuf0, tmpBuf1, tmpBuf2;
    AscendC::GlobalTensor<DTYPE_X> xGm;
    AscendC::GlobalTensor<DTYPE_Y> yGm;

    uint32_t totalLength;
    uint32_t tileNum;
    uint32_t tileLength;
    uint32_t blockLength;   // 可用于后续扩展，当前未使用
};

extern "C" __global__ __aicore__ void tanh_custom(GM_ADDR x, GM_ADDR y,
                                                  GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(TanhCustomTilingData);
    GET_TILING_DATA(tilingData, tiling);

    KernelTanh op;
    op.Init(x, y, tilingData.totalLength, tilingData.tileNum);
    op.Process();
}