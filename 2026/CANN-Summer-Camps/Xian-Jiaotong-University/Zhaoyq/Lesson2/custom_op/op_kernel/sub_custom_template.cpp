#include "kernel_operator.h"
#include "sub_custom_template_tiling.h"

#ifndef DTYPE_X
#define DTYPE_X half
#endif
#ifndef DTYPE_Y
#define DTYPE_Y half
#endif
#ifndef DTYPE_Z
#define DTYPE_Z half
#endif

constexpr int32_t QUE_DEPTH = 2;       

template<typename TX, typename TY, typename TZ>
class SubtractKernel {
public:
    __aicore__ inline SubtractKernel() {}

    __aicore__ inline void Init(GM_ADDR xPtr, GM_ADDR yPtr, GM_ADDR zPtr,
                                uint32_t elemTotal, uint32_t slicesPerCore) {
        uint32_t numCores = AscendC::GetBlockNum();
        uint32_t coreIdx  = AscendC::GetBlockIdx();
        uint32_t elemPerCore = elemTotal / numCores;
        this->elemPerSlice  = elemPerCore / slicesPerCore;
        this->totalSlices   = slicesPerCore;

        xGlobal.SetGlobalBuffer(
            reinterpret_cast<__gm__ TX*>(xPtr) + coreIdx * elemPerCore,
            elemPerCore);
        yGlobal.SetGlobalBuffer(
            reinterpret_cast<__gm__ TY*>(yPtr) + coreIdx * elemPerCore,
            elemPerCore);
        zGlobal.SetGlobalBuffer(
            reinterpret_cast<__gm__ TZ*>(zPtr) + coreIdx * elemPerCore,
            elemPerCore);

        pipe.InitBuffer(qX, QUE_DEPTH, this->elemPerSlice * sizeof(TX));
        pipe.InitBuffer(qY, QUE_DEPTH, this->elemPerSlice * sizeof(TY));
        pipe.InitBuffer(qZ, QUE_DEPTH, this->elemPerSlice * sizeof(TZ));
    }

    __aicore__ inline void Run() {
        for (int32_t slice = 0; slice < totalSlices; ++slice) {
            Fetch(slice);
            ElementwiseSub();
            WriteBack(slice);
        }
    }

private:
    __aicore__ inline void Fetch(int32_t sliceIdx) {
        uint32_t offset = sliceIdx * elemPerSlice;
        AscendC::LocalTensor<TX> xChunk = qX.AllocTensor<TX>();
        AscendC::LocalTensor<TY> yChunk = qY.AllocTensor<TY>();
        AscendC::DataCopy(xChunk, xGlobal[offset], elemPerSlice);
        AscendC::DataCopy(yChunk, yGlobal[offset], elemPerSlice);
        qX.EnQue(xChunk);
        qY.EnQue(yChunk);
    }

    __aicore__ inline void ElementwiseSub() {
        AscendC::LocalTensor<TX> xChunk = qX.DeQue<TX>();
        AscendC::LocalTensor<TY> yChunk = qY.DeQue<TY>();
        AscendC::LocalTensor<TZ> zChunk = qZ.AllocTensor<TZ>();
        AscendC::Sub(zChunk, xChunk, yChunk, elemPerSlice);
        qZ.EnQue(zChunk);
        qX.FreeTensor(xChunk);
        qY.FreeTensor(yChunk);
    }

    __aicore__ inline void WriteBack(int32_t sliceIdx) {
        uint32_t offset = sliceIdx * elemPerSlice;
        AscendC::LocalTensor<TZ> zChunk = qZ.DeQue<TZ>();
        AscendC::DataCopy(zGlobal[offset], zChunk, elemPerSlice);
        qZ.FreeTensor(zChunk);
    }

private:
    uint32_t elemPerSlice;
    int32_t  totalSlices;
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN,  QUE_DEPTH> qX, qY;
    AscendC::TQue<AscendC::QuePosition::VECOUT, QUE_DEPTH> qZ;
    AscendC::GlobalTensor<TX> xGlobal;
    AscendC::GlobalTensor<TY> yGlobal;
    AscendC::GlobalTensor<TZ> zGlobal;
};


extern "C" __global__ __aicore__ void sub_custom_template(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(SubCustomTemplateTilingData);
    GET_TILING_DATA(tilingData, tiling);
    SubtractKernel<DTYPE_X, DTYPE_Y, DTYPE_Z> op;
    op.Init(x, y, z, tilingData.size, 8);
    op.Run();               
}