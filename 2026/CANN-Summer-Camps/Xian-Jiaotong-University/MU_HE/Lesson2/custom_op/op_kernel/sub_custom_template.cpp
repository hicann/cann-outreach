#include "kernel_operator.h"

constexpr uint32_t BUFFER_NUM = 2;

struct SubCustomTilingData {
    uint32_t totalElemNum;
    uint32_t tileCount;
};

class KernelSubCustomTemplate {
public:
    __aicore__ inline KernelSubCustomTemplate() {}

    __aicore__ inline void Init(GM_ADDR xGm, GM_ADDR yGm, GM_ADDR zGm, SubCustomTilingData tiling)
    {
        gmX = xGm;
        gmY = yGm;
        gmZ = zGm;
        totalNum = tiling.totalElemNum;
        tileCnt = tiling.tileCount;
        singleTileSize = totalNum / tileCnt;

        pipeX.InitBuffer(singleTileSize, BUFFER_NUM);
        pipeY.InitBuffer(singleTileSize, BUFFER_NUM);
        pipeZ.InitBuffer(singleTileSize, BUFFER_NUM);
    }

    __aicore__ inline void Process()
    {
        for (uint32_t i = 0; i < tileCnt; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t step)
    {
        uint32_t offset = step * singleTileSize;
        uint32_t bufId = step % BUFFER_NUM;
        uint32_t byteSize = singleTileSize * sizeof(float);

        CopyParams copyX;
        copyX.src = gmX + offset * sizeof(float);
        copyX.dst = pipeX.GetBuf<uint8_t>(bufId);
        copyX.count = byteSize;
        CopyAsync(copyX);

        CopyParams copyY;
        copyY.src = gmY + offset * sizeof(float);
        copyY.dst = pipeY.GetBuf<uint8_t>(bufId);
        copyY.count = byteSize;
        CopyAsync(copyY);

        PipeSync();
    }

    __aicore__ inline void Compute(uint32_t step)
    {
        uint32_t bufId = step % BUFFER_NUM;
        auto xBuf = pipeX.GetBuf<float>(bufId);
        auto yBuf = pipeY.GetBuf<float>(bufId);
        auto zBuf = pipeZ.GetBuf<float>(bufId);

        //   Ԫ ؼ    z = x - y
        for (uint32_t i = 0; i < singleTileSize; i++) {
            zBuf[i] = xBuf[i] - yBuf[i];
        }
    }

    __aicore__ inline void CopyOut(uint32_t step)
    {
        uint32_t offset = step * singleTileSize;
        uint32_t bufId = step % BUFFER_NUM;
        uint32_t byteSize = singleTileSize * sizeof(float);

        CopyParams copyZ;
        copyZ.src = pipeZ.GetBuf<uint8_t>(bufId);
        copyZ.dst = gmZ + offset * sizeof(float);
        copyZ.count = byteSize;
        CopyAsync(copyZ);
        PipeSync();
    }

private:
    GM_ADDR gmX, gmY, gmZ;
    uint32_t totalNum;
    uint32_t tileCnt;
    uint32_t singleTileSize;
    Pipe<uint8_t> pipeX;
    Pipe<uint8_t> pipeY;
    Pipe<uint8_t> pipeZ;
};

__global__ __aicore__ void sub_custom_template(GM_ADDR x, GM_ADDR y, GM_ADDR z, SubCustomTilingData tiling)
{
    KernelSubCustomTemplate op;
    op.Init(x, y, z, tiling);
    op.Process();
} 