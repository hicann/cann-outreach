#include "kernel_operator.h"
#include "atanh_custom_tiling.h"

using namespace AscendC;

namespace {
constexpr uint32_t kBufferNum = 2;
constexpr uint32_t kBlockBytes = 32;

class KernelAtanhCustom {
 public:
  __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t total,
                              uint32_t tileLength, uint32_t tileNum,
                              uint32_t lastTileLength) {
    tileLength_ = tileLength;
    tileNum_ = tileNum;
    lastTileLength_ = lastTileLength;
    xGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(x), total);
    yGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(y), total);
    const uint32_t aligned =
        (tileLength * sizeof(half) + kBlockBytes - 1) / kBlockBytes *
        kBlockBytes / sizeof(half);
    pipe_.InitBuffer(xQueue_, kBufferNum, aligned * sizeof(half));
    pipe_.InitBuffer(yQueue_, kBufferNum, aligned * sizeof(half));
    pipe_.InitBuffer(tmpNumerator_, aligned * sizeof(half));
    pipe_.InitBuffer(tmpDenominator_, aligned * sizeof(half));
  }

  __aicore__ inline void Process() {
    for (uint32_t i = 0; i < tileNum_; ++i) {
      const uint32_t count = i + 1 == tileNum_ ? lastTileLength_ : tileLength_;
      CopyIn(i, count);
      Compute(count);
      CopyOut(i, count);
    }
  }

 private:
  __aicore__ inline void CopyIn(uint32_t tile, uint32_t count) {
    LocalTensor<half> x = xQueue_.AllocTensor<half>();
    DataCopyExtParams p{1, count * sizeof(half), 0, 0, 0};
    DataCopyPadExtParams<half> pad{true, 0, 0, 0};
    DataCopyPad(x, xGm_[tile * tileLength_], p, pad);
    xQueue_.EnQue(x);
  }

  __aicore__ inline void Compute(uint32_t count) {
    LocalTensor<half> x = xQueue_.DeQue<half>();
    LocalTensor<half> y = yQueue_.AllocTensor<half>();
    LocalTensor<half> numerator = tmpNumerator_.Get<half>();
    LocalTensor<half> denominator = tmpDenominator_.Get<half>();
    const half one = static_cast<half>(1.0f);
    Adds(numerator, x, one, count);
    Muls(denominator, x, static_cast<half>(-1.0f), count);
    Adds(denominator, denominator, one, count);
    Div(numerator, numerator, denominator, count);
    Ln(y, numerator, count);
    Muls(y, y, static_cast<half>(0.5f), count);
    yQueue_.EnQue(y);
    xQueue_.FreeTensor(x);
  }

  __aicore__ inline void CopyOut(uint32_t tile, uint32_t count) {
    LocalTensor<half> y = yQueue_.DeQue<half>();
    DataCopyExtParams p{1, count * sizeof(half), 0, 0, 0};
    DataCopyPad(yGm_[tile * tileLength_], y, p);
    yQueue_.FreeTensor(y);
  }

  TPipe pipe_;
  TQue<QuePosition::VECIN, kBufferNum> xQueue_;
  TQue<QuePosition::VECOUT, kBufferNum> yQueue_;
  TBuf<QuePosition::VECCALC> tmpNumerator_;
  TBuf<QuePosition::VECCALC> tmpDenominator_;
  GlobalTensor<half> xGm_;
  GlobalTensor<half> yGm_;
  uint32_t tileLength_ = 0;
  uint32_t tileNum_ = 0;
  uint32_t lastTileLength_ = 0;
};
}  // namespace

extern "C" __global__ __aicore__ void atanh_custom(
    GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
  GET_TILING_DATA(data, tiling);
  KernelAtanhCustom op;
  op.Init(x, y, data.totalLength, data.tileLength, data.tileNum,
          data.lastTileLength);
  op.Process();
}

#ifndef __CCE_KT_TEST__
void atanh_custom_do(uint32_t blockDim, void* l2ctrl, void* stream,
                     uint8_t* x, uint8_t* y, uint8_t* workspace,
                     uint8_t* tiling) {
  atanh_custom<<<blockDim, l2ctrl, stream>>>(x, y, workspace, tiling);
}
#endif
