// Kernel侧核函数实现
#include "kernel_operator.h"
#include "add_tiling.h"
#include "tiling_key_add.h"

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;  // 双缓冲

template <class DT_X> class KernelAdd {
public:
    __aicore__ inline KernelAdd() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z,
                                uint32_t length, uint32_t perCore) {
        // 计算本核处理的数据区间 [startIdx, startIdx + this->length)
        this->blockLength = perCore;
        uint32_t startIdx = GetBlockIdx() * blockLength;
        this->length = length - startIdx;
        if (this->length > blockLength) {
            this->length = blockLength;
        }

        // 设置全局内存
        xGm.SetGlobalBuffer((__gm__ DT_X *)x + startIdx, this->length);
        yGm.SetGlobalBuffer((__gm__ DT_X *)y + startIdx, this->length);
        zGm.SetGlobalBuffer((__gm__ DT_X *)z + startIdx, this->length);

        // 设置本地内存（UB）双缓冲队列
        pipe.InitBuffer(inQueueX, BUFFER_NUM, blockLength * sizeof(DT_X));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, blockLength * sizeof(DT_X));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, blockLength * sizeof(DT_X));
    }

    __aicore__ inline void Process() {
        if (this->length == 0) {  // 数据量不足时部分核无任务
            return;
        }
        CopyIn();
        Compute();
        CopyOut();
    }

private:
    // GM -> UB
    __aicore__ inline void CopyIn() {
        LocalTensor<DT_X> xLocal = inQueueX.AllocTensor<DT_X>();
        LocalTensor<DT_X> yLocal = inQueueY.AllocTensor<DT_X>();
        DataCopy(xLocal, xGm, this->length);
        DataCopy(yLocal, yGm, this->length);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }

    // UB内计算 z = x + y
    __aicore__ inline void Compute() {
        LocalTensor<DT_X> xLocal = inQueueX.DeQue<DT_X>();
        LocalTensor<DT_X> yLocal = inQueueY.DeQue<DT_X>();
        LocalTensor<DT_X> zLocal = outQueueZ.AllocTensor<DT_X>();

        AscendC::Add(zLocal, xLocal, yLocal, this->length);

        outQueueZ.EnQue<DT_X>(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }

    // UB -> GM
    __aicore__ inline void CopyOut() {
        LocalTensor<DT_X> zLocal = outQueueZ.DeQue<DT_X>();
        DataCopy(zGm, zLocal, this->length);
        outQueueZ.FreeTensor(zLocal);
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX, inQueueY;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueZ;
    GlobalTensor<DT_X> xGm, yGm, zGm;
    uint32_t length = 0;      // 本核实际处理元素个数
    uint32_t blockLength = 0; // 每核分配的元素个数
};

template <typename DT_X>
__global__ __aicore__ void add(GM_ADDR x, GM_ADDR y, GM_ADDR z,
                               GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(AddTilingData);
    GET_TILING_DATA_WITH_STRUCT(AddTilingData, tiling_data, tiling);
    KernelAdd<DT_X> op;
    op.Init(x, y, z, tiling_data.length, tiling_data.perCore);
    op.Process();
}

