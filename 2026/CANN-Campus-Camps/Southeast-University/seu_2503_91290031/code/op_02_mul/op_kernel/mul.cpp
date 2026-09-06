// Kernel侧核函数实现
#include "kernel_operator.h"

#include "mul_tiling.h"
#include "tiling_key_mul.h"

constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t QUEUE_DEPTH = 2;

template <class DT_X>
class KernelMul {
public:
    __aicore__ inline KernelMul() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t length, uint32_t tileDataNum) {
        uint32_t coreNum  = AscendC::GetBlockNum();
        uint32_t coreIdx  = AscendC::GetBlockIdx();
        uint32_t baseLen  = length / coreNum;             // host 已保证整除
        uint32_t rem      = length % coreNum;
        this->coreStart   = coreIdx * baseLen;            // 整除 → 对齐
        this->coreLength  = baseLen;
        
        uint32_t elemAlign = 32 / static_cast<uint32_t>(sizeof(DT_X));
this->tileLength = tileDataNum;
if (this->tileLength > this->coreLength) this->tileLength = this->coreLength;
this->tileLength = (this->tileLength / elemAlign) * elemAlign;   // 只在这个边缘情况下一行兜底
if (this->tileLength == 0) this->tileLength = elemAlign;
this->loopCount = (this->coreLength + this->tileLength - 1) / this->tileLength;

        xGm.SetGlobalBuffer((__gm__ DT_X *)x + this->coreStart, this->coreLength);
        yGm.SetGlobalBuffer((__gm__ DT_X *)y + this->coreStart, this->coreLength);
        zGm.SetGlobalBuffer((__gm__ DT_X *)z + this->coreStart, this->coreLength);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(DT_X));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, this->tileLength * sizeof(DT_X));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileLength * sizeof(DT_X));
    }
    __aicore__ inline void Process() {
        for (int32_t i = 0; i < this->loopCount; i++) {
            CopyIn(i);    // 搬入第 i 块
            Compute(i);   // 计算第 i 块
            CopyOut(i);   // 搬出第 i 块
        }
    }
private:
    __aicore__ inline uint32_t curCount(int32_t progress) {
        uint32_t start  = progress * this->tileLength;
        uint32_t remain = this->coreLength - start;
        return remain < this->tileLength ? remain : this->tileLength;
    }
    // 向上取整到 32B 对齐（用于 DataCopy 搬运）
    __aicore__ inline uint32_t alignUp(uint32_t n) {
        uint32_t e = 32 / static_cast<uint32_t>(sizeof(DT_X));
        return (n + e - 1) / e * e;
    }
    __aicore__ inline void CopyIn(int32_t progress) {
        uint32_t cnt   = this->curCount(progress);
        uint32_t ccopy = this->alignUp(cnt);
        AscendC::LocalTensor<DT_X> xLocal = inQueueX.AllocTensor<DT_X>();
        AscendC::LocalTensor<DT_X> yLocal = inQueueY.AllocTensor<DT_X>();
        AscendC::DataCopy(xLocal, xGm[progress * this->tileLength], ccopy);
        AscendC::DataCopy(yLocal, yGm[progress * this->tileLength], ccopy);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }
    __aicore__ inline void Compute(int32_t progress) {
        uint32_t cnt = this->curCount(progress);           // 计算用真实长度
        AscendC::LocalTensor<DT_X> xLocal = inQueueX.DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> yLocal = inQueueY.DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> zLocal = outQueueZ.AllocTensor<DT_X>();
        AscendC::Mul(zLocal, xLocal, yLocal, cnt);         // 核心：z = x * y
        outQueueZ.EnQue<DT_X>(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress) {
        uint32_t cnt   = this->curCount(progress);
        uint32_t ccopy = this->alignUp(cnt);
        AscendC::LocalTensor<DT_X> zLocal = outQueueZ.DeQue<DT_X>();
        AscendC::DataCopy(zGm[progress * this->tileLength], zLocal, ccopy);
        outQueueZ.FreeTensor(zLocal);
    }
private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, QUEUE_DEPTH> inQueueX, inQueueY;
    AscendC::TQue<AscendC::TPosition::VECOUT, QUEUE_DEPTH> outQueueZ;
    AscendC::GlobalTensor<DT_X> xGm, yGm, zGm;
    uint32_t coreStart;
    uint32_t coreLength;
    uint32_t tileNum;
    uint32_t tileLength;
    uint32_t loopCount;
};

template <typename DT_X>
__global__ __aicore__ void mul(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(MulTilingData);
    GET_TILING_DATA_WITH_STRUCT(MulTilingData, tiling_data, tiling);
    KernelMul<DT_X> op;
    op.Init(x, y, z, tiling_data.length, tiling_data.tileDataNum);
    op.Process();
}