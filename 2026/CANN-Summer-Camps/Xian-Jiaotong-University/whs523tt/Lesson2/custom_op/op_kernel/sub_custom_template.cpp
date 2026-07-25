#include "kernel_operator.h"
#include "sub_custom_template_tiling.h"

using namespace AscendC;

constexpr uint32_t TILE_LENGTH = 2048;

class KernelSubCustomTemplate {
public:
    __aicore__ inline KernelSubCustomTemplate() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t totalLength)
    {
        uint32_t blockIdx = GetBlockIdx();
        uint32_t blockNum = GetBlockNum();

        uint32_t baseLength = totalLength / blockNum;
        uint32_t tail = totalLength % blockNum;

        if (blockIdx < tail) {
            blockLength = baseLength + 1;
            blockOffset = blockIdx * blockLength;
        } else {
            blockLength = baseLength;
            blockOffset = tail * (baseLength + 1) + (blockIdx - tail) * baseLength;
        }

        xGm.SetGlobalBuffer((__gm__ DTYPE_X*)x + blockOffset, blockLength);
        yGm.SetGlobalBuffer((__gm__ DTYPE_Y*)y + blockOffset, blockLength);
        zGm.SetGlobalBuffer((__gm__ DTYPE_Z*)z + blockOffset, blockLength);

        pipe.InitBuffer(xBuf, TILE_LENGTH * sizeof(DTYPE_X));
        pipe.InitBuffer(yBuf, TILE_LENGTH * sizeof(DTYPE_Y));
        pipe.InitBuffer(zBuf, TILE_LENGTH * sizeof(DTYPE_Z));
    }

    __aicore__ inline void Process()
    {
        LocalTensor<DTYPE_X> xLocal = xBuf.Get<DTYPE_X>();
        LocalTensor<DTYPE_Y> yLocal = yBuf.Get<DTYPE_Y>();
        LocalTensor<DTYPE_Z> zLocal = zBuf.Get<DTYPE_Z>();

        uint32_t offset = 0;
        while (offset < blockLength) {
            uint32_t remain = blockLength - offset;
            uint32_t curLength = remain > TILE_LENGTH ? TILE_LENGTH : remain;

            DataCopy(xLocal, xGm[offset], curLength);
            DataCopy(yLocal, yGm[offset], curLength);

            PipeBarrier<PIPE_ALL>();

            Sub(zLocal, xLocal, yLocal, curLength);

            PipeBarrier<PIPE_ALL>();

            DataCopy(zGm[offset], zLocal, curLength);

            offset += curLength;
        }
    }

private:
    TPipe pipe;

    TBuf<QuePosition::VECCALC> xBuf;
    TBuf<QuePosition::VECCALC> yBuf;
    TBuf<QuePosition::VECCALC> zBuf;

    GlobalTensor<DTYPE_X> xGm;
    GlobalTensor<DTYPE_Y> yGm;
    GlobalTensor<DTYPE_Z> zGm;

    uint32_t blockOffset;
    uint32_t blockLength;
};

extern "C" __global__ __aicore__ void sub_custom_template(
    GM_ADDR x,
    GM_ADDR y,
    GM_ADDR z,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(SubCustomTemplateTilingData);
    GET_TILING_DATA(tilingData, tiling);

    KernelSubCustomTemplate op;
    op.Init(x, y, z, tilingData.size);
    op.Process();
}