#include "kernel_operator.h"
#include "sub_custom_template_tiling.h"

using namespace AscendC;

template <typename T>
class KernelSub {
public:
    __aicore__ inline KernelSub() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, SubCustomTemplateTilingData tiling) {
        int64_t baseLen = tiling.totalLength / GetBlockNum();
        int64_t offset = GetBlockIdx() * baseLen;
        if (GetBlockIdx() == GetBlockNum() - 1) {
            blockLen_ = static_cast<uint32_t>(baseLen + tiling.totalLength % GetBlockNum());
        } else {
            blockLen_ = static_cast<uint32_t>(baseLen);
        }

        gmX_.SetGlobalBuffer((__gm__ T*)x + offset, blockLen_);
        gmY_.SetGlobalBuffer((__gm__ T*)y + offset, blockLen_);
        gmZ_.SetGlobalBuffer((__gm__ T*)z + offset, blockLen_);

        pipe_.InitBuffer(qInX_, 1, blockLen_ * sizeof(T));
        pipe_.InitBuffer(qInY_, 1, blockLen_ * sizeof(T));
        pipe_.InitBuffer(qOutZ_, 1, blockLen_ * sizeof(T));
    }

    __aicore__ inline void Process() {
        if (blockLen_ == 0) return;
        // 一次搬运全部数据
        LocalTensor<T> xLocal = qInX_.AllocTensor<T>();
        LocalTensor<T> yLocal = qInY_.AllocTensor<T>();
        DataCopy(xLocal, gmX_, blockLen_);
        DataCopy(yLocal, gmY_, blockLen_);
        qInX_.EnQue(xLocal);
        qInY_.EnQue(yLocal);

        LocalTensor<T> xTile = qInX_.DeQue<T>();
        LocalTensor<T> yTile = qInY_.DeQue<T>();
        LocalTensor<T> zTile = qOutZ_.AllocTensor<T>();
        Sub(zTile, xTile, yTile, blockLen_);
        qOutZ_.EnQue<T>(zTile);
        qInX_.FreeTensor(xTile);
        qInY_.FreeTensor(yTile);

        LocalTensor<T> zLocal = qOutZ_.DeQue<T>();
        DataCopy(gmZ_, zLocal, blockLen_);
        qOutZ_.FreeTensor(zLocal);
    }

private:
    TPipe pipe_;
    TQue<QuePosition::VECIN, 1>  qInX_, qInY_;
    TQue<QuePosition::VECOUT, 1> qOutZ_;
    GlobalTensor<T> gmX_, gmY_, gmZ_;
    uint32_t blockLen_ = 0;
};

extern "C" __global__ __aicore__ void sub_custom_template(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace,
                                                          GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(SubCustomTemplateTilingData);
    GET_TILING_DATA(tilingData, tiling);
    (void)workspace;

    if (tilingData.dtype == 0) {
        KernelSub<float> op;
        op.Init(x, y, z, tilingData);
        op.Process();
    } else {
        KernelSub<half> op;
        op.Init(x, y, z, tilingData);
        op.Process();
    }
}