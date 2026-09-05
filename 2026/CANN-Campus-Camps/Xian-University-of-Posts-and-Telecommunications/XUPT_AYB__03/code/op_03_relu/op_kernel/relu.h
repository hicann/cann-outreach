#ifndef RELU_H
#define RELU_H

#include "kernel_operator.h"

using namespace AscendC;

template <typename T>
class KernelRelu {
public:
    __aicore__ inline KernelRelu()
    {
    }

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR y,
        uint32_t blockLen,
        uint32_t tileLen,
        uint32_t tileNumPerCore)
    {
        uint32_t blockIdx = GetBlockIdx();

        xGm.SetGlobalBuffer(
            (__gm__ T*)x + blockIdx * blockLen,
            blockLen);

        yGm.SetGlobalBuffer(
            (__gm__ T*)y + blockIdx * blockLen,
            blockLen);

        this->tileLen = tileLen;
        this->tileNumPerCore = tileNumPerCore;

        constexpr int32_t BUFFER_NUM = 2;

        pipe.InitBuffer(
            inQueueX,
            BUFFER_NUM,
            tileLen * sizeof(T));

        pipe.InitBuffer(
            outQueueY,
            BUFFER_NUM,
            tileLen * sizeof(T));
    }

    __aicore__ inline void CopyIn(int32_t progress)
    {
        LocalTensor<T> inX =
            inQueueX.AllocTensor<T>();

        DataCopy(
            inX,
            xGm[progress * tileLen],
            tileLen);

        inQueueX.EnQue(inX);
    }

    __aicore__ inline void Compute(int32_t progress)
    {
        LocalTensor<T> inX =
            inQueueX.DeQue<T>();

        LocalTensor<T> outY =
            outQueueY.AllocTensor<T>();

        Relu(
            outY,
            inX,
            tileLen);

        outQueueY.EnQue(outY);

        inQueueX.FreeTensor(inX);
    }

    __aicore__ inline void CopyOut(int32_t progress)
    {
        LocalTensor<T> outY =
            outQueueY.DeQue<T>();

        DataCopy(
            yGm[progress * tileLen],
            outY,
            tileLen);

        outQueueY.FreeTensor(outY);
    }

    __aicore__ inline void Process()
    {
        for (int32_t i = 0;
             i < static_cast<int32_t>(tileNumPerCore);
             ++i) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    GlobalTensor<T> xGm;
    GlobalTensor<T> yGm;

    uint32_t tileLen;
    uint32_t tileNumPerCore;

    TPipe pipe;

    TQue<QuePosition::VECIN, 1> inQueueX;
    TQue<QuePosition::VECOUT, 1> outQueueY;
};

#endif