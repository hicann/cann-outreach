#include "kernel_operator.h"
#include "add_custom_template_tiling.h"

constexpr int32_t BUF_N = 2;
constexpr int32_t ALGN = 16;

template <class TX, class TY, class TZ>
class KernelAddCustom {
public:
    __aicore__ inline KernelAddCustom() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t total, uint32_t tnum)
    {
        blen = total / AscendC::GetBlockNum();
        tcnt = tnum;
        tlen = (blen / tcnt / ALGN) * ALGN;
        if (tlen < ALGN) tlen = (blen < ALGN) ? blen : ALGN;

        uint32_t off = blen * AscendC::GetBlockIdx();
        gX.SetGlobalBuffer((__gm__ TX *)x + off, blen);
        gY.SetGlobalBuffer((__gm__ TY *)y + off, blen);
        gZ.SetGlobalBuffer((__gm__ TZ *)z + off, blen);

        pipe.InitBuffer(iX, BUF_N, tlen * sizeof(TX));
        pipe.InitBuffer(iY, BUF_N, tlen * sizeof(TY));
        pipe.InitBuffer(oZ, BUF_N, tlen * sizeof(TZ));
    }

    __aicore__ inline void Process()
    {
        fetch(0);
        for (uint32_t i = 0; i < tcnt - 1; i++) {
            fetch(i + 1);
            exec(i);
            flush(i);
        }
        exec(tcnt - 1);
        flush(tcnt - 1);
    }

private:
    __aicore__ inline void fetch(uint32_t step)
    {
        auto lx = iX.AllocTensor<TX>();
        auto ly = iY.AllocTensor<TY>();
        AscendC::DataCopy(lx, gX[step * tlen], tlen);
        AscendC::DataCopy(ly, gY[step * tlen], tlen);
        iX.EnQue(lx); iY.EnQue(ly);
    }

    __aicore__ inline void exec(uint32_t step)
    {
        auto lx = iX.DeQue<TX>();
        auto ly = iY.DeQue<TY>();
        auto lz = oZ.AllocTensor<TZ>();
        AscendC::Add(lz, lx, ly, tlen);
        oZ.EnQue<TZ>(lz);
        iX.FreeTensor(lx); iY.FreeTensor(ly);
    }

    __aicore__ inline void flush(uint32_t step)
    {
        auto lz = oZ.DeQue<TZ>();
        AscendC::DataCopy(gZ[step * tlen], lz, tlen);
        oZ.FreeTensor(lz);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUF_N> iX, iY;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUF_N> oZ;
    AscendC::GlobalTensor<TX> gX;
    AscendC::GlobalTensor<TY> gY;
    AscendC::GlobalTensor<TZ> gZ;
    uint32_t blen, tcnt, tlen;
};

__global__ __aicore__ void add_custom_template(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR ws, GM_ADDR tl)
{
    REGISTER_TILING_DEFAULT(AddCustomTemplateTilingData);
    GET_TILING_DATA_WITH_STRUCT(AddCustomTemplateTilingData, tdata, tl);
    KernelAddCustom<DTYPE_X, DTYPE_Y, DTYPE_Z> op;
    op.Init(x, y, z, tdata.totalLength, tdata.tileNum);
    op.Process();
}
