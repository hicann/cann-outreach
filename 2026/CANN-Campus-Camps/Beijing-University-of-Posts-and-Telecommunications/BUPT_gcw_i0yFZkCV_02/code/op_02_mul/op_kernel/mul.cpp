// Kernel侧核函数实现
#include "kernel_operator.h"

#include "mul_tiling.h"
#include "tiling_key_mul.h"

constexpr uint32_t BUFFER_NUM = 2;  // 每队列 2 份缓冲（双缓冲：搬第 i+1 轮与算第 i 轮重叠）

// z = x * y，elementwise Vector 算子：
//   数据流 = MTE2 搬入(GM→UB) → Vector 乘(UB 内) → MTE3 搬出(UB→GM)
//   核间切分：host 保证 blockDim 整除 totalLength 且每核块 32B 对齐，无尾块
//   核内切分：每核循环 tileNum * BUFFER_NUM 轮，每轮 tileLength 个元素
template <class DT_X>
class KernelMul {
public:
    __aicore__ inline KernelMul() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, const MulTilingData& tiling) {
        this->totalLength = tiling.totalLength;
        this->blockLength = tiling.blockLength;
        this->tileNum = tiling.tileNum;
        this->tileLength = tiling.tileLength;

        // 核间窗口绑定：本核只处理 [blockIdx*blockLength, +blockLength) 这一段
        uint32_t start = AscendC::GetBlockIdx() * this->blockLength;
        if (start >= this->totalLength) {
            // 防御：核数多于数据时该核空闲（本题切分保证整除，不会触发）
            this->loopCount = 0;
            return;
        }
        this->loopCount = this->tileNum * BUFFER_NUM;

        xGm.SetGlobalBuffer((__gm__ DT_X *)x + start, this->blockLength);
        yGm.SetGlobalBuffer((__gm__ DT_X *)y + start, this->blockLength);
        zGm.SetGlobalBuffer((__gm__ DT_X *)z + start, this->blockLength);

        // UB 切分方案：x/y 输入队列、z 输出队列各 BUFFER_NUM 份缓冲
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(DT_X));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, this->tileLength * sizeof(DT_X));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileLength * sizeof(DT_X));
    }

    __aicore__ inline void Process() {
        for (int32_t i = 0; i < this->loopCount; i++) {
            CopyIn(i);    // MTE2：HBM → UB
            Compute(i);   // Vector：UB 内计算
            CopyOut(i);   // MTE3：UB → HBM
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress) {
        AscendC::LocalTensor<DT_X> xLocal = inQueueX.AllocTensor<DT_X>();
        AscendC::LocalTensor<DT_X> yLocal = inQueueY.AllocTensor<DT_X>();
        AscendC::DataCopy(xLocal, xGm[progress * this->tileLength], this->tileLength);
        AscendC::DataCopy(yLocal, yGm[progress * this->tileLength], this->tileLength);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }

    __aicore__ inline void Compute(int32_t progress) {
        AscendC::LocalTensor<DT_X> xLocal = inQueueX.DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> yLocal = inQueueY.DeQue<DT_X>();
        AscendC::LocalTensor<DT_X> zLocal = outQueueZ.AllocTensor<DT_X>();
        // 参数顺序 (dst, src0, src1) = (z, x, y)，即 z = x * y
        AscendC::Mul(zLocal, xLocal, yLocal, this->tileLength);
        outQueueZ.EnQue<DT_X>(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress) {
        AscendC::LocalTensor<DT_X> zLocal = outQueueZ.DeQue<DT_X>();
        AscendC::DataCopy(zGm[progress * this->tileLength], zLocal, this->tileLength);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX, inQueueY;   // UB 输入区
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueZ;           // UB 输出区
    AscendC::GlobalTensor<DT_X> xGm, yGm, zGm;                                   // GM 窗口句柄

    uint32_t totalLength;  // 数据总长度
    uint32_t blockLength;  // 每核处理的元素数
    uint32_t tileNum;      // 核内记账组数
    uint32_t tileLength;   // 每轮进 UB 的元素数
    int32_t loopCount;     // 物理循环次数 = tileNum * BUFFER_NUM（空闲核为 0）
};

template <typename DT_X>
 __global__ __aicore__ void mul(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(MulTilingData);
    GET_TILING_DATA_WITH_STRUCT(MulTilingData, tiling_data, tiling);
    KernelMul<DT_X> op;
    op.Init(x, y, z, tiling_data);
    op.Process();
}
