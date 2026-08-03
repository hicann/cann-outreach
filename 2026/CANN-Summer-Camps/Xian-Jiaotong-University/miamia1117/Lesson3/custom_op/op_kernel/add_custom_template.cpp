#include "kernel_operator.h"
#include "add_custom_template_tiling.h"

namespace config_const {
constexpr uint32_t QUEUE_MAX_DEPTH = 1;
constexpr uint32_t SUB_TILE_COUNT = 6;
static_assert(SUB_TILE_COUNT >= 1, "Sub tile number cannot be zero");
static_assert(QUEUE_MAX_DEPTH >= 1, "Queue depth must be greater than zero");
}

template <typename InX, typename InY, typename OutZ>
class VectorAddKernel {
public:
    __aicore__ inline VectorAddKernel() = default;

    __aicore__ inline void Init(GM_ADDR gmX, GM_ADDR gmY, GM_ADDR gmZ, uint32_t totalLength)
    {
        uint32_t blockCount = AscendC::GetBlockNum();
        uint32_t workBlockId = AscendC::GetBlockIdx();
        uint32_t avgBlockLen = totalLength / blockCount;
        uint32_t remainElem = totalLength % blockCount;

        // 计算当前Block有效长度与GM起始偏移
        if (workBlockId < blockCount - 1) {
            curBlockLen = avgBlockLen;
            gmStartOffset = workBlockId * avgBlockLen;
        } else {
            curBlockLen = avgBlockLen + remainElem;
            gmStartOffset = workBlockId * avgBlockLen;
        }

        singleTileLen = curBlockLen / config_const::SUB_TILE_COUNT;
        lastTileValidLen = singleTileLen + (curBlockLen % config_const::SUB_TILE_COUNT);

        // 绑定全局内存区间
        xGlobalBuf.SetGlobalBuffer((__gm__ InX *)gmX + gmStartOffset, curBlockLen);
        yGlobalBuf.SetGlobalBuffer((__gm__ InY *)gmY + gmStartOffset, curBlockLen);
        zGlobalBuf.SetGlobalBuffer((__gm__ OutZ *)gmZ + gmStartOffset, curBlockLen);

        // 初始化流水线队列
        pipeUnit.InitBuffer(xInputQue, config_const::QUEUE_MAX_DEPTH, singleTileLen * sizeof(InX));
        pipeUnit.InitBuffer(yInputQue, config_const::QUEUE_MAX_DEPTH, singleTileLen * sizeof(InY));
        pipeUnit.InitBuffer(zOutputQue, config_const::QUEUE_MAX_DEPTH, singleTileLen * sizeof(OutZ));
    }

    __aicore__ inline void Process()
    {
        if (curBlockLen == 0) {
            return;
        }
        // 处理前面完整tile
        for (uint32_t t = 0; t < config_const::SUB_TILE_COUNT - 1; t++) {
            uint32_t tileGmOff = t * singleTileLen;
            LoadData(tileGmOff, singleTileLen);
            ComputeData(singleTileLen);
            WriteBack(tileGmOff, singleTileLen);
        }
        // 处理尾部不完整tile
        uint32_t tailOffset = (config_const::SUB_TILE_COUNT - 1) * singleTileLen;
        LoadData(tailOffset, lastTileValidLen);
        ComputeData(lastTileValidLen);
        WriteBack(tailOffset, lastTileValidLen);
    }

private:
    __aicore__ inline void LoadData(uint32_t offset, uint32_t validSize)
    {
        AscendC::LocalTensor<InX> lmX = xInputQue.AllocTensor<InX>();
        AscendC::LocalTensor<InY> lmY = yInputQue.AllocTensor<InY>();
        AscendC::DataCopy(lmX, xGlobalBuf[offset], validSize);
        AscendC::DataCopy(lmY, yGlobalBuf[offset], validSize);
        xInputQue.EnQue(lmX);
        yInputQue.EnQue(lmY);
    }

    __aicore__ inline void ComputeData(uint32_t validSize)
    {
        AscendC::LocalTensor<InX> lmX = xInputQue.DeQue<InX>();
        AscendC::LocalTensor<InY> lmY = yInputQue.DeQue<InY>();
        AscendC::LocalTensor<OutZ> lmZ = zOutputQue.AllocTensor<OutZ>();

        AscendC::Add(lmZ, lmX, lmY, validSize);

        zOutputQue.EnQue<OutZ>(lmZ);
        xInputQue.FreeTensor(lmX);
        yInputQue.FreeTensor(lmY);
    }

    __aicore__ inline void WriteBack(uint32_t offset, uint32_t validSize)
    {
        AscendC::LocalTensor<OutZ> lmZ = zOutputQue.DeQue<OutZ>();
        AscendC::DataCopy(zGlobalBuf[offset], lmZ, validSize);
        zOutputQue.FreeTensor(lmZ);
    }

private:
    AscendC::TPipe pipeUnit;
    AscendC::TQue<AscendC::TPosition::VECIN, config_const::QUEUE_MAX_DEPTH> xInputQue;
    AscendC::TQue<AscendC::TPosition::VECIN, config_const::QUEUE_MAX_DEPTH> yInputQue;
    AscendC::TQue<AscendC::TPosition::VECOUT, config_const::QUEUE_MAX_DEPTH> zOutputQue;

    AscendC::GlobalTensor<InX> xGlobalBuf;
    AscendC::GlobalTensor<InY> yGlobalBuf;
    AscendC::GlobalTensor<OutZ> zGlobalBuf;

    uint32_t curBlockLen = 0;
    uint32_t gmStartOffset = 0;
    uint32_t singleTileLen = 0;
    uint32_t lastTileValidLen = 0;
};

extern "C" __global__ __aicore__ void add_custom_template(
    GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(AddCustomTemplateTilingData);
    GET_TILING_DATA_WITH_STRUCT(AddCustomTemplateTilingData, tileParam, tiling);
    VectorAddKernel<DTYPE_X, DTYPE_Y, DTYPE_Z> opKernel;
    opKernel.Init(x, y, z, tileParam.totalLength);
    opKernel.Process();
}