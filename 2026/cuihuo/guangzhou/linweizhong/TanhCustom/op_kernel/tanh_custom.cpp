#include "kernel_operator.h"
#include "tanh_custom_tiling.h"

constexpr int32_t BUFFER_NUM = 2; // tensor num for each queue

// 算子输入/输出数据类型：FP16（考题要求支持 Float16 输入输出）
using DTYPE_X = half;
using DTYPE_Y = half;

class KernelTanh {
public:
    __aicore__ inline KernelTanh() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength, uint32_t tileNum)
    {
        // 1. 计算每个核负责的数据长度（多核均分）
        this->blockLength = totalLength / AscendC::GetBlockNum();
        // 2. 记录 tile 个数与单次搬运长度（注意双缓冲：每 tile 实际长度为 blockLength/tileNum/BUFFER_NUM）
        this->tileNum = tileNum;
        this->tileLength = this->blockLength / tileNum / BUFFER_NUM;

        // 3. 绑定全局内存（GlobalTensor），按当前核号偏移到本核负责的区间
        xGm.SetGlobalBuffer((__gm__ DTYPE_X *)x + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
        yGm.SetGlobalBuffer((__gm__ DTYPE_Y *)y + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);

        // 4. 申请双缓冲队列（输入/输出各 BUFFER_NUM 块，块大小为 tileLength 个元素）
        pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(DTYPE_X));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, this->tileLength * sizeof(DTYPE_Y));
        // 5. 申请临时计算 buffer（tanh 公式需要 e^x、e^-x、分子、分母共 4 路中间量，3 块 buffer 轮转复用即可）
        pipe.InitBuffer(tmpBuf0, this->tileLength * sizeof(DTYPE_X));
        pipe.InitBuffer(tmpBuf1, this->tileLength * sizeof(DTYPE_X));
        pipe.InitBuffer(tmpBuf2, this->tileLength * sizeof(DTYPE_X));
    }
    __aicore__ inline void Process()
    {
        int32_t loopCount = this->tileNum * BUFFER_NUM;
        for (int32_t i = 0; i < loopCount; i++) {
            CopyIn(i);
            Compute(i);
            CopyOut(i);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        // 从队列申请一块空闲 buffer，将本 tile 数据从全局内存拷入
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.AllocTensor<DTYPE_X>();
        AscendC::DataCopy(xLocal, xGm[progress * this->tileLength], this->tileLength);
        inQueueX.EnQue(xLocal);
    }
    __aicore__ inline void Compute(int32_t progress)
    {
        // 取输入 tile
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.DeQue<DTYPE_X>();
        // 申请输出 buffer 与 3 块临时 buffer
        AscendC::LocalTensor<DTYPE_Y> yLocal = outQueueY.AllocTensor<DTYPE_Y>();
        AscendC::LocalTensor<DTYPE_X> tmp0 = tmpBuf0.AllocTensor<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_X> tmp1 = tmpBuf1.AllocTensor<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_X> tmp2 = tmpBuf2.AllocTensor<DTYPE_X>();

        // tanh(x) = (exp(x) - exp(-x)) / (exp(x) + exp(-x))
        // tmp0 = exp(x)
        AscendC::Exp(tmp0, xLocal, this->tileLength);
        // tmp1 = -x
        AscendC::Muls(tmp1, xLocal, static_cast<DTYPE_X>(-1.0), this->tileLength);
        // tmp2 = exp(-x)
        AscendC::Exp(tmp2, tmp1, this->tileLength);
        // tmp1 = exp(x) - exp(-x)  （分子）
        AscendC::Sub(tmp1, tmp0, tmp2, this->tileLength);
        // tmp0 = exp(x) + exp(-x)  （分母，复用 tmp0）
        AscendC::Add(tmp0, tmp0, tmp2, this->tileLength);
        // y = 分子 / 分母 = tanh(x)
        AscendC::Div(yLocal, tmp1, tmp0, this->tileLength);

        // 释放/入队：结果入输出队列，其余归还
        outQueueY.EnQue(yLocal);
        inQueueX.FreeTensor(xLocal);
        tmpBuf0.FreeTensor(tmp0);
        tmpBuf1.FreeTensor(tmp1);
        tmpBuf2.FreeTensor(tmp2);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        // 取计算结果拷回全局内存
        AscendC::LocalTensor<DTYPE_Y> yLocal = outQueueY.DeQue<DTYPE_Y>();
        AscendC::DataCopy(yGm[progress * this->tileLength], yLocal, this->tileLength);
        outQueueY.FreeTensor(yLocal);
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
    // 创建算子对象并初始化（tiling 数据由 host 侧写入：totalLength 总长度、tileNum 每核 tile 数）
    KernelTanh op;
    op.Init(x, y, tilingData.totalLength, tilingData.tileNum);
    op.Process();
}
