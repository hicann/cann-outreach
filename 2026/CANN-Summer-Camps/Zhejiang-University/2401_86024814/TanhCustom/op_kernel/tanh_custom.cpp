#include "kernel_operator.h"
#include "tanh_custom_tiling.h"

constexpr int32_t BUFFER_NUM = 2; // tensor num for each queue

class KernelTanh {
public:
    __aicore__ inline KernelTanh() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength, uint32_t tileNum)
    {
        // tileNum 等于 BLOCK_DIM, 表示将数据切分为 tileNum 个分片
        // 每个 block 独立处理 1 个分片, tileLength 即本 block 负责的数据量
        this->tileNum = tileNum;
        this->tileLength = totalLength / tileNum;
        // 双缓冲: 将本 block 的数据再均分为 BUFFER_NUM 份, 每次迭代处理 blockLength 个元素
        this->blockLength = this->tileLength / BUFFER_NUM;

        uint32_t blockIdx = AscendC::GetBlockIdx();
        uint32_t offset = blockIdx * this->tileLength;

        xGm.SetGlobalBuffer((__gm__ DTYPE_X*)x + offset, this->tileLength);
        yGm.SetGlobalBuffer((__gm__ DTYPE_Y*)y + offset, this->tileLength);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->blockLength * sizeof(DTYPE_X));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->blockLength * sizeof(DTYPE_Y));
        pipe.InitBuffer(tmpBuf0, this->blockLength * sizeof(DTYPE_X));
        pipe.InitBuffer(tmpBuf1, this->blockLength * sizeof(DTYPE_X));
        pipe.InitBuffer(tmpBuf2, this->blockLength * sizeof(DTYPE_X));
    }
    __aicore__ inline void Process()
    {
        // 双缓冲迭代次数 = BUFFER_NUM, 每次处理 blockLength 个元素
        // 总处理量 = BUFFER_NUM * blockLength = tileLength, 恰好覆盖本 block 数据
        int32_t loopCount = BUFFER_NUM;
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        AscendC::LocalTensor<DTYPE_X> inLocal = inQueueX.AllocTensor<DTYPE_X>();
        AscendC::DataCopy(inLocal, xGm[progress * this->blockLength], this->blockLength);
        inQueueX.EnQue(inLocal);
    }
    __aicore__ inline void Compute(int32_t progress)
    {
        AscendC::LocalTensor<DTYPE_X> inLocal = inQueueX.DeQue<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_Y> outLocal = outQueueY.AllocTensor<DTYPE_Y>();

        // 从 TBuf 获取 LocalTensor，TBuf 不能直接传给计算 API
        AscendC::LocalTensor<DTYPE_X> tmpLocal0 = tmpBuf0.Get<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_X> tmpLocal1 = tmpBuf1.Get<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_X> tmpLocal2 = tmpBuf2.Get<DTYPE_X>();

        // tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
        // Step 1: tmpLocal0 = exp(x)
        AscendC::Exp(tmpLocal0, inLocal, this->blockLength);
        // Step 2: tmpLocal1 = -x, then exp(-x)
        AscendC::Muls(tmpLocal1, inLocal, (DTYPE_X)-1.0, this->blockLength);
        AscendC::Exp(tmpLocal1, tmpLocal1, this->blockLength);
        // Step 3: tmpLocal2 = exp(x) - exp(-x)  (numerator)
        AscendC::Sub(tmpLocal2, tmpLocal0, tmpLocal1, this->blockLength);
        // Step 4: tmpLocal0 = exp(x) + exp(-x)  (denominator)
        AscendC::Add(tmpLocal0, tmpLocal0, tmpLocal1, this->blockLength);
        // Step 5: outLocal = numerator / denominator
        AscendC::Div(outLocal, tmpLocal2, tmpLocal0, this->blockLength);

        outQueueY.EnQue(outLocal);
        inQueueX.FreeTensor(inLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<DTYPE_Y> outLocal = outQueueY.DeQue<DTYPE_Y>();
        AscendC::DataCopy(yGm[progress * this->blockLength], outLocal, this->blockLength);
        outQueueY.FreeTensor(outLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN, BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmpBuf0,tmpBuf1,tmpBuf2;
    AscendC::GlobalTensor<DTYPE_X> xGm;
    AscendC::GlobalTensor<DTYPE_Y> yGm;
    uint32_t blockLength;
    uint32_t tileNum;
    uint32_t tileLength;
};

extern "C" __global__ __aicore__ void tanh_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(TanhCustomTilingData);
    GET_TILING_DATA(tilingData, tiling);

    KernelTanh op;
    op.Init(x, y, tilingData.totalLength, tilingData.tileNum);
    op.Process();
}