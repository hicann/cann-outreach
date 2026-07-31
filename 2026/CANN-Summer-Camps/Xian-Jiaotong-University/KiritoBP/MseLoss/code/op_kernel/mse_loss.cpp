// Kernel侧核函数实现 - MseLoss
// v9: ASCENDC_TPL_SEL回header + 直接指针读tiling + 模板参数T防宏污染
#include "kernel_operator.h"
// 不 include mse_loss_tiling.h — 避免与框架生成的 class MseLossTilingData 冲突
#include "tiling_key_mse_loss.h"

// DT_PREDICT 由编译器 -D 注入, 此处 fallback
// 注意: 模板参数名 T 必须与宏 DT_PREDICT 不同, 否则宏展开会污染模板参数
#ifndef DT_PREDICT
#define DT_PREDICT half
#endif

constexpr int32_t BUFFER_NUM = 2;
constexpr int32_t TILE_NUM   = 8;

template <class T>
class KernelMseLoss {
public:
    __aicore__ inline KernelMseLoss() {}
    __aicore__ inline void Init(GM_ADDR predict, GM_ADDR label, GM_ADDR y, uint32_t totalLength)
    {
        uint32_t coreNum = AscendC::GetBlockNum();
        uint32_t coreIdx = AscendC::GetBlockIdx();
        uint32_t basePerCore = totalLength / coreNum;
        uint32_t remainElem  = totalLength % coreNum;
        this->coreDataLen = basePerCore + (coreIdx < remainElem ? 1 : 0);
        if (this->coreDataLen == 0) return;

        uint32_t gmOffset = coreIdx * basePerCore + (coreIdx < remainElem ? coreIdx : remainElem);
        predGm.SetGlobalBuffer((__gm__ T *)predict + gmOffset, this->coreDataLen);
        lablGm.SetGlobalBuffer((__gm__ T *)label  + gmOffset, this->coreDataLen);
        yGm.SetGlobalBuffer((__gm__ T *)y       + gmOffset, this->coreDataLen);

        this->tileBaseLen   = this->coreDataLen / TILE_NUM;
        this->tileRemainLen = this->coreDataLen % TILE_NUM;
        // 修复: 最后tile长度 = tileBaseLen + tileRemainLen
        uint32_t maxTileLen = this->tileBaseLen + this->tileRemainLen;
        uint32_t bufByteSize = maxTileLen * sizeof(T);

        pipe.InitBuffer(inQx, BUFFER_NUM, bufByteSize);
        pipe.InitBuffer(inQl, BUFFER_NUM, bufByteSize);
        pipe.InitBuffer(outQ, BUFFER_NUM, bufByteSize);
    }

    __aicore__ inline void Process()
    {
        if (this->coreDataLen == 0) return;
        for (int32_t i = 0; i < TILE_NUM; ++i) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        uint32_t copyLen = GetTileLen(progress);
        if (copyLen == 0) return;

        uint32_t tileOffset = progress * tileBaseLen;

        AscendC::LocalTensor<T> xLocal = inQx.AllocTensor<T>();
        AscendC::LocalTensor<T> lLocal = inQl.AllocTensor<T>();
        AscendC::DataCopy(xLocal, predGm[tileOffset], copyLen);
        AscendC::DataCopy(lLocal, lablGm[tileOffset], copyLen);
        inQx.EnQue(xLocal);
        inQl.EnQue(lLocal);
    }

    __aicore__ inline void Compute(int32_t progress)
    {
        uint32_t computeLen = GetTileLen(progress);
        if (computeLen == 0) return;

        AscendC::LocalTensor<T> xLocal = inQx.DeQue<T>();
        AscendC::LocalTensor<T> lLocal = inQl.DeQue<T>();
        AscendC::LocalTensor<T> zLocal = outQ.AllocTensor<T>();

        // 严格按设计文档: Sub → in-place Mul
        AscendC::Sub(zLocal, xLocal, lLocal, computeLen);
        AscendC::Mul(zLocal, zLocal, zLocal, computeLen);

        outQ.EnQue<T>(zLocal);
        inQx.FreeTensor(xLocal);
        inQl.FreeTensor(lLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress)
    {
        uint32_t copyLen = GetTileLen(progress);
        if (copyLen == 0) return;

        uint32_t tileOffset = progress * tileBaseLen;

        AscendC::LocalTensor<T> zLocal = outQ.DeQue<T>();
        AscendC::DataCopy(yGm[tileOffset], zLocal, copyLen);
        outQ.FreeTensor(zLocal);
    }

    // 获取每个 tile 的实际处理长度
    // 修复: 最后 tile = tileBaseLen + tileRemainLen
    __aicore__ inline uint32_t GetTileLen(int32_t progress)
    {
        uint32_t len = tileBaseLen;
        if (progress == TILE_NUM - 1) {
            len += tileRemainLen;
        }
        return len;
    }

private:
    AscendC::TPipe pipe;
    AscendC::GlobalTensor<T> predGm, lablGm, yGm;
    AscendC::TQue<AscendC::QuePosition::VECIN,  BUFFER_NUM> inQx, inQl;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQ;
    uint32_t coreDataLen, tileBaseLen, tileRemainLen;
};

template <typename T>
__global__ __aicore__ void mse_loss(
    GM_ADDR predict, GM_ADDR label, GM_ADDR y,
    GM_ADDR workspace, GM_ADDR tiling)
{
    // 使用框架生成的 MseLossTilingData class (由 ASCENDC_TPL_SEL 触发)
    REGISTER_TILING_DEFAULT(MseLossTilingData);
    GET_TILING_DATA_WITH_STRUCT(MseLossTilingData, tiling_data, tiling);

    KernelMseLoss<T> op;
    op.Init(predict, label, y, tiling_data.length);
    op.Process();
}
