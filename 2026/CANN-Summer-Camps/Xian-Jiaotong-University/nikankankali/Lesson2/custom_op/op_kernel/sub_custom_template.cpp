#include "kernel_operator.h"
#include "sub_custom_template_tiling.h"

constexpr int32_t QUEUE_CAP = 2;

template <class T_X, class T_Y, class T_Z>
class KernelSubCustom {
public:
    __aicore__ inline KernelSubCustom() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t totalLen, uint32_t tNum)
    {
        blkLen = totalLen / AscendC::GetBlockNum();
        tileCnt = tNum;
        tileLen = blkLen / tileCnt / QUEUE_CAP;

        uint32_t off = blkLen * AscendC::GetBlockIdx();
        gmX.SetGlobalBuffer((__gm__ T_X *)x + off, blkLen);
        gmY.SetGlobalBuffer((__gm__ T_Y *)y + off, blkLen);
        gmZ.SetGlobalBuffer((__gm__ T_Z *)z + off, blkLen);

        p.InitBuffer(qX, QUEUE_CAP, tileLen * sizeof(T_X));
        p.InitBuffer(qY, QUEUE_CAP, tileLen * sizeof(T_Y));
        p.InitBuffer(qZ, QUEUE_CAP, tileLen * sizeof(T_Z));
    }

    __aicore__ inline void Process()
    {
        int32_t total = tileCnt * QUEUE_CAP;
        for (int32_t i = 0; i < total; i++) {
            load(i);
            calc(i);
            store(i);
        }
    }

private:
    __aicore__ inline void load(int32_t step)
    {
        auto lx = qX.AllocTensor<T_X>();
        auto ly = qY.AllocTensor<T_Y>();
        AscendC::DataCopy(lx, gmX[step * tileLen], tileLen);
        AscendC::DataCopy(ly, gmY[step * tileLen], tileLen);
        qX.EnQue(lx); qY.EnQue(ly);
    }

    __aicore__ inline void calc(int32_t step)
    {
        auto lx = qX.DeQue<T_X>();
        auto ly = qY.DeQue<T_Y>();
        auto lz = qZ.AllocTensor<T_Z>();
        AscendC::Sub(lz, lx, ly, tileLen);
        qZ.EnQue<T_Z>(lz);
        qX.FreeTensor(lx); qY.FreeTensor(ly);
    }

    __aicore__ inline void store(int32_t step)
    {
        auto lz = qZ.DeQue<T_Z>();
        AscendC::DataCopy(gmZ[step * tileLen], lz, tileLen);
        qZ.FreeTensor(lz);
    }

private:
    AscendC::TPipe p;
    AscendC::TQue<AscendC::TPosition::VECIN, QUEUE_CAP> qX, qY;
    AscendC::TQue<AscendC::TPosition::VECOUT, QUEUE_CAP> qZ;
    AscendC::GlobalTensor<T_X> gmX;
    AscendC::GlobalTensor<T_Y> gmY;
    AscendC::GlobalTensor<T_Z> gmZ;
    uint32_t blkLen, tileCnt, tileLen;
};

__global__ __aicore__ void sub_custom_template(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(TilingDataTemplate);
    GET_TILING_DATA_WITH_STRUCT(TilingDataTemplate, td, tiling);
    KernelSubCustom<DTYPE_X, DTYPE_Y, DTYPE_Z> op;
    op.Init(x, y, z, td.totalLength, td.tileNum);
    op.Process();
}
