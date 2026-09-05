// Kernel侧核函数实现
#include "kernel_operator.h"

#include "mul_tiling.h"
#include "tiling_key_mul.h"

constexpr int32_t BUFFER_NUM = 1;

template <class DT_X>
class KernelMul {
public:
    __aicore__ inline KernelMul() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t totalLength,
                                uint32_t blockLength, uint32_t tileNum, uint32_t tileLength) {
        // 1. 保存tiling数据
        this->totalLength = totalLength;
        this->blockLength = blockLength;
        this->tileNum = tileNum;
        this->tileLength = tileLength;
        // 2. 计算当前核在GM上的起始偏移与需要处理的数据长度（尾核可能不足blockLength）
        uint32_t startIndex = this->blockLength * AscendC::GetBlockIdx();
        this->currentLength = 0;
        if (startIndex < this->totalLength) {
            uint32_t remainLength = this->totalLength - startIndex;
            this->currentLength = (remainLength < this->blockLength) ? remainLength : this->blockLength;
        }
        // 3. 计算循环次数：单核按tileLength切块，最后一块可能不满
        this->loopCount = 0;
        if (this->currentLength > 0 && this->tileLength > 0) {
            this->loopCount = (this->currentLength + this->tileLength - 1) / this->tileLength;
            if (static_cast<uint32_t>(this->loopCount) > this->tileNum) {
                this->loopCount = static_cast<int32_t>(this->tileNum);
            }
        }    
        // 4. 设置全局内存地址
        xGm.SetGlobalBuffer((__gm__ DT_X *)x + startIndex, this->currentLength);
        yGm.SetGlobalBuffer((__gm__ DT_X *)y + startIndex, this->currentLength);
        zGm.SetGlobalBuffer((__gm__ DT_X *)z + startIndex, this->currentLength);
        // 5. 为 x/y/z 在UB上各申请 BUFFER_NUM 份 tileLength 大小的 buffer
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(DT_X));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, this->tileLength * sizeof(DT_X));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileLength * sizeof(DT_X));
    }
    __aicore__ inline void Process() {
        if (this->loopCount == 0) {
            return;
        }
        // 双缓冲流水：CopyIn -> Compute -> CopyOut
        for (int32_t i = 0; i < this->loopCount; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    // 取第progress块的实际长度：只有最后一块可能不满tileLength
    __aicore__ inline void CopyIn(int32_t progress) {
        // 从GM搬运 x、y 的第 progress 块到UB
        uint32_t offset = static_cast<uint32_t>(progress) * this->tileLength;
        uint32_t length = this->tileLength; 
        AscendC::LocalTensor<DT_X> xLocal = inQueueX.AllocTensor<DT_X>();
        AscendC::LocalTensor<DT_X> yLocal = inQueueY.AllocTensor<DT_X>();
        AscendC::DataCopy(xLocal, xGm[offset], length);
        AscendC::DataCopy(yLocal, yGm[offset], length);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }
    __aicore__ inline void Compute(int32_t progress) {
        // 矢量逐元素乘法: z = x * y
        AscendC::LocalTensor<DT_X> xLocal = inQueueX.DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> yLocal = inQueueY.DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> zLocal = outQueueZ.AllocTensor<DT_X>();

        AscendC::Mul(zLocal, xLocal, yLocal, this->tileLength);

        outQueueZ.EnQue<DT_X>(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress) {
        // 将第 progress 块的计算结果搬回GM
        uint32_t offset = static_cast<uint32_t>(progress) * this->tileLength;
        AscendC::LocalTensor<DT_X> zLocal = outQueueZ.DeQue<DT_X>();
        AscendC::DataCopy(zGm[offset], zLocal, this->tileLength);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueY;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueZ;
    AscendC::GlobalTensor<DT_X> xGm;
    AscendC::GlobalTensor<DT_X> yGm;
    AscendC::GlobalTensor<DT_X> zGm;
    uint32_t totalLength;   // 总元素个数
    uint32_t blockLength;   // 每核分摊的元素个数
    uint32_t tileNum;       // 单核分块数
    uint32_t tileLength;    // 每块元素个数
    uint32_t currentLength; // 当前核实际处理的元素个数
    int32_t loopCount;      // 当前核的循环次数
};

template <typename DT_X>
 __global__ __aicore__ void mul(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(MulTilingData);
    GET_TILING_DATA_WITH_STRUCT(MulTilingData, tiling_data, tiling);
    KernelMul<DT_X> op;
    op.Init(x, y, z, tiling_data.totalLength, tiling_data.blockLength,tiling_data.tileNum, tiling_data.tileLength);
    op.Process();
}
