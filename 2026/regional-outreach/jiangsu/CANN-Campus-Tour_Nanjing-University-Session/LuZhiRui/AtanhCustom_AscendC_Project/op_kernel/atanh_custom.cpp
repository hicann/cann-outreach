#include "kernel_operator.h"
#include "atanh_custom_tiling.h"

using namespace AscendC;

class KernelAtanh {
public:

__aicore__ inline void Init(
    GM_ADDR x,
    GM_ADDR y,
    const AtanhCustomTilingData* t)
{
    id = GetBlockIdx();

    uint32_t offset =
        id * t->numPerCore;

    length =
        (offset + t->numPerCore >
         t->totalLength)
        ? t->totalLength - offset
        : t->numPerCore;

    xGm.SetGlobalBuffer(
        (__gm__ half*)x + offset,
        length);

    yGm.SetGlobalBuffer(
        (__gm__ half*)y + offset,
        length);

    pipe.InitBuffer(
        inQueue,
        2,
        t->tileLength*sizeof(half));

    pipe.InitBuffer(
        outQueue,
        2,
        t->tileLength*sizeof(half));

    pipe.InitBuffer(
        tmp,
        t->tmpBufferSize);
}

__aicore__ inline void Process()
{
    if(length==0)return;

    LocalTensor<half> in =
        inQueue.AllocTensor<half>();

    DataCopy(in,xGm,length);

    inQueue.EnQue(in);

    LocalTensor<half> src =
        inQueue.DeQue<half>();

    LocalTensor<half> dst =
        outQueue.AllocTensor<half>();

    LocalTensor<uint8_t> buf =
        tmp.Get<uint8_t>();

    Atanh(dst,src,buf,length);

    outQueue.EnQue(dst);

    LocalTensor<half> out =
        outQueue.DeQue<half>();

    DataCopy(yGm,out,length);

    inQueue.FreeTensor(src);
    outQueue.FreeTensor(out);
}

private:
    TPipe pipe;
    TQue<QuePosition::VECIN,2> inQueue;
    TQue<QuePosition::VECOUT,2> outQueue;
    TBuf<QuePosition::VECCALC> tmp;

    GlobalTensor<half> xGm;
    GlobalTensor<half> yGm;

    uint32_t id=0;
    uint32_t length=0;
};

extern "C" __global__ __aicore__
void atanh_custom(
    GM_ADDR x,
    GM_ADDR y,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    GET_TILING_DATA(t,tiling);

    KernelAtanh op;
    op.Init(x,y,&t);
    op.Process();
}
