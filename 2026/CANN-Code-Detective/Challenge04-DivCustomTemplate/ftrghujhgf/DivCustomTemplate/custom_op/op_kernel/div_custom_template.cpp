#include "kernel_operator.h"
#include "div_custom_template_tiling.h"
using namespace AscendC;
constexpr int BUFFER_NUM = 2;

template<typename T>
class KernelOp {
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, DivCustomTemplateTilingData tiling)
    {
        this->tileNum = tiling.tileNum;
        blockLength = tiling.size / GetBlockNum();
        tileLength = blockLength / BUFFER_NUM / tileNum;
        
        gmX.SetGlobalBuffer((__gm__ T*)x + blockLength * GetBlockIdx(), blockLength);
        gmY.SetGlobalBuffer((__gm__ T*)y + blockLength * GetBlockIdx(), blockLength);
        gmZ.SetGlobalBuffer((__gm__ T*)z + blockLength * GetBlockIdx(), blockLength);

        pipe.InitBuffer(inX, BUFFER_NUM, tileLength * sizeof(T));
        pipe.InitBuffer(inY, BUFFER_NUM, tileLength * sizeof(T));
        pipe.InitBuffer(outZ, BUFFER_NUM, tileLength * sizeof(T));
        // pipe.InitBuffer(buf1, tileLength * sizeof(T));
        // pipe.InitBuffer(buf2, tileLength * sizeof(T));
    }
    __aicore__ inline void Process()
    {
        for(int i = 0; i < tileNum * BUFFER_NUM; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        LocalTensor<T> x = inX.AllocTensor<T>();
        LocalTensor<T> y = inY.AllocTensor<T>();
        DataCopy(x, gmX[progress * tileLength], tileLength);
        DataCopy(y, gmY[progress * tileLength], tileLength);
        inX.EnQue(x);
        inY.EnQue(y);
    }
    __aicore__ inline void Compute(int32_t progress)
    {
        LocalTensor<T> x = inX.DeQue<T>();
        LocalTensor<T> y = inY.DeQue<T>();
        LocalTensor<T> z = outZ.AllocTensor<T>();

        Div(z, x, y, tileLength);
        
        outZ.EnQue(z);
        inX.FreeTensor(x);
        inY.FreeTensor(y);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        LocalTensor<T> z = outZ.DeQue<T>();
        DataCopy(gmZ[progress * tileLength], z, tileLength);
        outZ.FreeTensor(z);
    }

private:
    int tileLength, blockLength, tileNum;
    TPipe pipe;
    TQue<TPosition::VECIN, BUFFER_NUM> inX, inY;
    TQue<TPosition::VECOUT, BUFFER_NUM> outZ;
    TBuf<TPosition::VECCALC> buf1, buf2;
    GlobalTensor<T> gmX, gmY, gmZ;
};

extern "C" __global__ __aicore__ void div_custom_template(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(DivCustomTemplateTilingData);
    GET_TILING_DATA(tilingData, tiling);
    // AscendC::PRINTF("tilingSize=%d, tileNum=%d", (int) tilingData.size, (int)tilingData.tileNum);
    // TODO: user kernel impl
    KernelOp<DTYPE_X>op;
    op.Init(x, y, z, tilingData);
    op.Process();
}