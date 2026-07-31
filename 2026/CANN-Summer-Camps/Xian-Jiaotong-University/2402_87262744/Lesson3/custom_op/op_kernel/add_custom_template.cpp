#include "kernel_operator.h"
#include "add_custom_template_tiling.h"

template <class dtypeX, class dtypeY, class dtypeZ>
class KernelAdd {
public:
    __aicore__ inline KernelAdd() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z,
                                uint32_t totalLength, uint32_t tileNum)
    {
        this->blockLength = totalLength / AscendC::GetBlockNum();
        this->tileNum = tileNum;
        this->tileLength = this->blockLength / tileNum;

        uint32_t offset = this->blockLength * AscendC::GetBlockIdx();
        xGm.SetGlobalBuffer((__gm__ dtypeX *)x + offset, this->blockLength);
        yGm.SetGlobalBuffer((__gm__ dtypeY *)y + offset, this->blockLength);
        zGm.SetGlobalBuffer((__gm__ dtypeZ *)z + offset, this->blockLength);

        pipe.InitBuffer(xBuf, this->tileLength * sizeof(dtypeX));
        pipe.InitBuffer(yBuf, this->tileLength * sizeof(dtypeY));
        pipe.InitBuffer(zBuf, this->tileLength * sizeof(dtypeZ));
    }

    __aicore__ inline void Process()
    {
        AscendC::LocalTensor<dtypeX> xLocal = xBuf.Get<dtypeX>();
        AscendC::LocalTensor<dtypeY> yLocal = yBuf.Get<dtypeY>();
        AscendC::LocalTensor<dtypeZ> zLocal = zBuf.Get<dtypeZ>();

        for (int32_t i = 0; i < this->tileNum; ++i) {
            AscendC::DataCopy(xLocal, xGm[i * this->tileLength], this->tileLength);
            AscendC::DataCopy(yLocal, yGm[i * this->tileLength], this->tileLength);
            AscendC::Add(zLocal, xLocal, yLocal, this->tileLength);
            AscendC::DataCopy(zGm[i * this->tileLength], zLocal, this->tileLength);
        }
    }

private:
    AscendC::TPipe pipe;
    AscendC::TBuf<> xBuf;  
    AscendC::TBuf<> yBuf;
    AscendC::TBuf<> zBuf;
    AscendC::GlobalTensor<dtypeX> xGm;
    AscendC::GlobalTensor<dtypeY> yGm;
    AscendC::GlobalTensor<dtypeZ> zGm;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
};

extern "C" __global__ __aicore__ void add_custom_template(
    GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(AddCustomTemplateTilingData);
    GET_TILING_DATA_WITH_STRUCT(AddCustomTemplateTilingData, tiling_data, tiling);
    KernelAdd<DTYPE_X, DTYPE_Y, DTYPE_Z> op;
    op.Init(x, y, z, tiling_data.totalLength, tiling_data.tileNum);
    op.Process();
}