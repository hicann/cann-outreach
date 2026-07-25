#ifndef SUB_CUSTOM_TEMPLATE_VECTOR_H__
#define SUB_CUSTOM_TEMPLATE_VECTOR_H__

#include "kernel_operator.h"
#include "sub_custom_template_tiling.h"

using namespace AscendC;

template <typename T>
class SubCustomTemplateVector {
    const uint32_t BUFFER_NUM = 2;

public:
    __aicore__ SubCustomTemplateVector() {};
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, SubCustomTemplateTilingData *tiling, TPipe *pipe);
    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(uint64_t offset, uint32_t length);
    __aicore__ inline void Compute(uint32_t length);
    __aicore__ inline void CopyOut(uint64_t offset, uint32_t length);

private:
    GlobalTensor<T> xGm;
    GlobalTensor<T> yGm;
    GlobalTensor<T> zGm;
    TQue<TPosition::VECIN, 1> xQue;
    TQue<TPosition::VECIN, 1> yQue;
    TQue<TPosition::VECOUT, 1> zQue;
    TBuf<TPosition::VECCALC> xFloatBuf;
    TBuf<TPosition::VECCALC> yFloatBuf;
    TBuf<TPosition::VECCALC> zFloatBuf;
    LocalTensor<float> xFloat;
    LocalTensor<float> yFloat;
    LocalTensor<float> zFloat;
    uint32_t tileLength = 0;
    uint32_t lastTileLength = 0;
    uint32_t tileNum = 0;
    uint32_t blockIdx = 0;
    uint32_t coreNum = 0;
    uint64_t blockLength = 0;
    uint64_t offset = 0;
};

template <typename T>
__aicore__ inline void SubCustomTemplateVector<T>::Init(
    GM_ADDR x, GM_ADDR y, GM_ADDR z, SubCustomTemplateTilingData *tiling, TPipe *pipe)
{
    blockIdx = GetBlockIdx();
    coreNum = GetBlockNum();
    if (blockIdx >= coreNum) {
        return;
    }

    tileLength = tiling->tileLength;
    if (tileLength == 0) {
        return;
    }

    if (blockIdx < coreNum - 1) {
        lastTileLength = tiling->formerLastTileLength;
        tileNum = tiling->formerTileNum;
        offset = static_cast<uint64_t>(blockIdx) * tiling->formerLength;
        blockLength = tiling->formerLength;
    } else {
        lastTileLength = tiling->tailLastTileLength;
        tileNum = tiling->tailTileNum;
        offset = tiling->formerLength * static_cast<uint64_t>(coreNum - 1);
        blockLength = tiling->tailLength;
    }

    if (tileNum == 0 || blockLength == 0) {
        return;
    }

    xGm.SetGlobalBuffer((__gm__ T *)x + offset, blockLength);
    yGm.SetGlobalBuffer((__gm__ T *)y + offset, blockLength);
    zGm.SetGlobalBuffer((__gm__ T *)z + offset, blockLength);

    pipe->InitBuffer(xQue, BUFFER_NUM, tileLength * sizeof(T));
    pipe->InitBuffer(yQue, BUFFER_NUM, tileLength * sizeof(T));
    pipe->InitBuffer(zQue, BUFFER_NUM, tileLength * sizeof(T));
    if constexpr (!std::is_same<T, float>::value) {
        pipe->InitBuffer(xFloatBuf, tileLength * sizeof(float));
        pipe->InitBuffer(yFloatBuf, tileLength * sizeof(float));
        pipe->InitBuffer(zFloatBuf, tileLength * sizeof(float));
        xFloat = xFloatBuf.Get<float>();
        yFloat = yFloatBuf.Get<float>();
        zFloat = zFloatBuf.Get<float>();
    }
}

template <typename T>
__aicore__ inline void SubCustomTemplateVector<T>::Process()
{
    if (tileNum == 0 || tileLength == 0 || blockLength == 0) {
        return;
    }

    for (uint32_t i = 0; i < tileNum; ++i) {
        uint32_t length = (i == tileNum - 1) ? lastTileLength : tileLength;
        uint64_t tileOffset = static_cast<uint64_t>(i) * tileLength;
        CopyIn(tileOffset, length);
        Compute(length);
        CopyOut(tileOffset, length);
    }
}

template <typename T>
__aicore__ inline void SubCustomTemplateVector<T>::CopyIn(uint64_t offset, uint32_t length)
{
    LocalTensor<T> xLocal = xQue.AllocTensor<T>();
    LocalTensor<T> yLocal = yQue.AllocTensor<T>();
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(length * sizeof(T)), 0, 0, 0};
    DataCopyPadExtParams<T> padParams{false, 0, 0, static_cast<T>(0)};
    DataCopyPad(xLocal, xGm[offset], copyParams, padParams);
    DataCopyPad(yLocal, yGm[offset], copyParams, padParams);
    xQue.EnQue<T>(xLocal);
    yQue.EnQue<T>(yLocal);
}

template <typename T>
__aicore__ inline void SubCustomTemplateVector<T>::Compute(uint32_t length)
{
    LocalTensor<T> xLocal = xQue.DeQue<T>();
    LocalTensor<T> yLocal = yQue.DeQue<T>();
    LocalTensor<T> zLocal = zQue.AllocTensor<T>();

    if constexpr (std::is_same<T, float>::value) {
        Sub(zLocal, xLocal, yLocal, length);
    } else {
        Cast(xFloat, xLocal, RoundMode::CAST_NONE, length);
        Cast(yFloat, yLocal, RoundMode::CAST_NONE, length);
        Sub(zFloat, xFloat, yFloat, length);
        Cast(zLocal, zFloat, RoundMode::CAST_ROUND, length);
    }

    zQue.EnQue<T>(zLocal);
    xQue.FreeTensor(xLocal);
    yQue.FreeTensor(yLocal);
}

template <typename T>
__aicore__ inline void SubCustomTemplateVector<T>::CopyOut(uint64_t offset, uint32_t length)
{
    LocalTensor<T> zLocal = zQue.DeQue<T>();
    DataCopyExtParams copyParams{1, static_cast<uint32_t>(length * sizeof(T)), 0, 0, 0};
    DataCopyPad(zGm[offset], zLocal, copyParams);
    zQue.FreeTensor(zLocal);
}

#endif // SUB_CUSTOM_TEMPLATE_VECTOR_H__
