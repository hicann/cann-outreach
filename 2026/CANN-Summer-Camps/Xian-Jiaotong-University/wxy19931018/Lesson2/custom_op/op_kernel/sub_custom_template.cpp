
#include "kernel_operator.h"
#include "sub_custom_template_tiling.h"
using namespace AscendC;

constexpr int32_t BUFFER_NUM = 1;

template<typename T>
class KernelSub {
  public:
      __aicore__ inline KernelSub() {}

      __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t totalLength) {
          xGm.SetGlobalBuffer((__gm__ T*)x);
          yGm.SetGlobalBuffer((__gm__ T*)y);
          zGm.SetGlobalBuffer((__gm__ T*)z);

          // 计算每个 block 处理的数据长度
          uint32_t blockNum = GetBlockNum();
          blockLength = totalLength / blockNum;

          // 初始化队列，只为当前 block 的数据分配内存
          pipe.InitBuffer(inQueueX, BUFFER_NUM, blockLength * sizeof(T));
          pipe.InitBuffer(inQueueY, BUFFER_NUM, blockLength * sizeof(T));
          pipe.InitBuffer(outQueueZ, BUFFER_NUM, blockLength * sizeof(T));
      }

      __aicore__ inline void Process() {
          // 计算当前 block 的起始位置
          uint32_t blockIdx = GetBlockIdx();
          uint32_t offset = blockIdx * blockLength;

          CopyIn(offset);
          Compute();
          CopyOut(offset);
      }

private:
      __aicore__ inline void CopyIn(uint32_t offset) {
          LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
          LocalTensor<T> yLocal = inQueueY.AllocTensor<T>();

          DataCopy(xLocal, xGm[offset], blockLength);
          DataCopy(yLocal, yGm[offset], blockLength);

          inQueueX.EnQue(xLocal);
          inQueueY.EnQue(yLocal);
      }

      __aicore__ inline void Compute() {
          LocalTensor<T> xLocal = inQueueX.DeQue<T>();
          LocalTensor<T> yLocal = inQueueY.DeQue<T>();
          LocalTensor<T> zLocal = outQueueZ.AllocTensor<T>();

          Sub(zLocal, xLocal, yLocal, blockLength);

          outQueueZ.EnQue<T>(zLocal);
          inQueueX.FreeTensor(xLocal);
          inQueueY.FreeTensor(yLocal);
      }

      __aicore__ inline void CopyOut(uint32_t offset) {
          LocalTensor<T> zLocal = outQueueZ.DeQue<T>();
          DataCopy(zGm[offset], zLocal, blockLength);
          outQueueZ.FreeTensor(zLocal);
      }
private:
      TPipe pipe;
      TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX, inQueueY;
      TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueZ;
      GlobalTensor<T> xGm, yGm, zGm;
      uint32_t blockLength;
};
extern "C" __global__ __aicore__ void sub_custom_template(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR
tiling) {
      REGISTER_TILING_DEFAULT(SubCustomTemplateTilingData);
      GET_TILING_DATA(tilingData, tiling);

      if (TILING_KEY_IS(1)) {
          KernelSub<half> op;
          op.Init(x, y, z, tilingData.totalLength);
          op.Process();
      } else {
          KernelSub<float> op;
          op.Init(x, y, z, tilingData.totalLength);
          op.Process();
      }
}