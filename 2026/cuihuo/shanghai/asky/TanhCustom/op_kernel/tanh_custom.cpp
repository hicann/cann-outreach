#include "kernel_operator.h"
#include "tanh_custom_tiling.h"

constexpr int32_t BUFFER_NUM = 2; // tensor num for each queue

class KernelTanh {
public:
    __aicore__ inline KernelTanh() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength, uint32_t tileNum)
    {
        // 获取当前核的索引
        uint32_t blockIdx = AscendC::GetBlockIdx();
        // 获取参与计算的核数
        uint32_t usedCoreNum = AscendC::GetBlockNum();

        // 每核负责的数据量（均分，最后一核可能略少）
        uint32_t blockLength = (totalLength + usedCoreNum - 1) / usedCoreNum;
        // 对 32 字节（FP16 为 16 个元素）对齐
        constexpr uint32_t ALIGN_NUM = 32 / sizeof(DTYPE_X);  // FP16: 16
        blockLength = ((blockLength + ALIGN_NUM - 1) / ALIGN_NUM) * ALIGN_NUM;

        // 最后一核的实际长度，避免越界
        uint32_t actualBlockLength = blockLength;
        if ((blockIdx + 1) * blockLength > totalLength) {
            actualBlockLength = totalLength > blockIdx * blockLength
                                    ? totalLength - blockIdx * blockLength
                                    : 0;
        }

        uint32_t dataOffset = blockIdx * blockLength;
        this->blockLength = actualBlockLength;
        this->tileNum     = tileNum;

        // 每次 tile 的长度（需 32B 对齐）
        uint32_t tileLen = (actualBlockLength + tileNum - 1) / tileNum;
        tileLen = ((tileLen + ALIGN_NUM - 1) / ALIGN_NUM) * ALIGN_NUM;
        this->tileLength = tileLen;

        // 绑定全局内存
        xGm.SetGlobalBuffer((__gm__ DTYPE_X *)x + dataOffset, actualBlockLength);
        yGm.SetGlobalBuffer((__gm__ DTYPE_Y *)y + dataOffset, actualBlockLength);

        // 初始化 FP16 输入/输出队列（Double Buffer：BUFFER_NUM=2）
        // 每块 FP16 tileLength 个元素
        pipe.InitBuffer(inQueueX,  BUFFER_NUM, tileLength * sizeof(DTYPE_X));
        pipe.InitBuffer(outQueueY, BUFFER_NUM, tileLength * sizeof(DTYPE_Y));

        // FP32 升精度临时缓冲区（3 个：in_fp32、compute、out_fp32）
        // tmpBuf0：存放输入升精度后的 FP32 数据
        // tmpBuf1：Tanh 计算结果的 FP32 数据（复用 tmpBuf0 亦可，此处用两块保证安全）
        // tmpBuf2：预留（本算子仅 Tanh，tmpBuf1 已够用；保留与骨架声明一致）
        pipe.InitBuffer(tmpBuf0, tileLength * sizeof(float));
        pipe.InitBuffer(tmpBuf1, tileLength * sizeof(float));
        pipe.InitBuffer(tmpBuf2, tileLength * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        // loopCount = tileNum * BUFFER_NUM（Double Buffer 展开）
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
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.AllocTensor<DTYPE_X>();

        // 当前 tile 实际长度（最后一个 tile 可能不满整块）
        uint32_t curOffset     = progress * this->tileLength;
        uint32_t curTileLength = this->tileLength;
        if (curOffset + curTileLength > this->blockLength) {
            curTileLength = (curOffset < this->blockLength)
                                ? (this->blockLength - curOffset)
                                : 0;
            // 对齐到 32B（不足部分使用 pad 填充为 0）
            constexpr uint32_t ALIGN_NUM = 32 / sizeof(DTYPE_X);
            uint32_t alignedLen = ((curTileLength + ALIGN_NUM - 1) / ALIGN_NUM) * ALIGN_NUM;

            // 使用 DataCopyPad：isPad=true，rightPad=补齐元素数，padValue=0
            uint8_t rightPad = static_cast<uint8_t>(alignedLen - curTileLength);
            AscendC::DataCopyExtParams copyInParams{
                1,
                static_cast<uint32_t>(curTileLength * sizeof(DTYPE_X)),
                0, 0, 0
            };
            AscendC::DataCopyPadExtParams<DTYPE_X> padParams{
                true, 0, rightPad, static_cast<DTYPE_X>(0)
            };
            AscendC::DataCopyPad(xLocal, xGm[curOffset], copyInParams, padParams);
        } else {
            // 完整 tile，直接搬运
            AscendC::DataCopyExtParams copyInParams{
                1,
                static_cast<uint32_t>(curTileLength * sizeof(DTYPE_X)),
                0, 0, 0
            };
            AscendC::DataCopyPadExtParams<DTYPE_X> padParams{false, 0, 0, static_cast<DTYPE_X>(0)};
            AscendC::DataCopyPad(xLocal, xGm[curOffset], copyInParams, padParams);
        }

        inQueueX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(int32_t progress)
    {
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.DeQue<DTYPE_X>();
        AscendC::LocalTensor<DTYPE_Y> yLocal = outQueueY.AllocTensor<DTYPE_Y>();

        // 当前 tile 的对齐长度（计算使用对齐后的长度）
        uint32_t curOffset     = progress * this->tileLength;
        uint32_t curTileLength = this->tileLength;
        if (curOffset + curTileLength > this->blockLength) {
            uint32_t validLen = (curOffset < this->blockLength)
                                    ? (this->blockLength - curOffset)
                                    : 0;
            constexpr uint32_t ALIGN_NUM = 32 / sizeof(DTYPE_X);
            curTileLength = ((validLen + ALIGN_NUM - 1) / ALIGN_NUM) * ALIGN_NUM;
        }

        // FP16 → FP32 升精度
        AscendC::LocalTensor<float> xFp32 = tmpBuf0.Get<float>();
        AscendC::Cast(xFp32, xLocal, AscendC::RoundMode::CAST_NONE, curTileLength);

        // FP32 Tanh 计算
        AscendC::LocalTensor<float> yFp32 = tmpBuf1.Get<float>();
        AscendC::Tanh(yFp32, xFp32, curTileLength);

        // FP32 → FP16 降精度
        AscendC::Cast(yLocal, yFp32, AscendC::RoundMode::CAST_ROUND, curTileLength);

        outQueueY.EnQue<DTYPE_Y>(yLocal);
        inQueueX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<DTYPE_Y> yLocal = outQueueY.DeQue<DTYPE_Y>();

        uint32_t curOffset     = progress * this->tileLength;
        uint32_t curTileLength = this->tileLength;
        if (curOffset + curTileLength > this->blockLength) {
            curTileLength = (curOffset < this->blockLength)
                                ? (this->blockLength - curOffset)
                                : 0;
        }

        if (curTileLength > 0) {
            AscendC::DataCopyExtParams copyOutParams{
                1,
                static_cast<uint32_t>(curTileLength * sizeof(DTYPE_Y)),
                0, 0, 0
            };
            AscendC::DataCopyPad(yGm[curOffset], yLocal, copyOutParams);
        }

        outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::QuePosition::VECIN,  BUFFER_NUM> inQueueX;
    AscendC::TQue<AscendC::QuePosition::VECOUT, BUFFER_NUM> outQueueY;
    AscendC::TBuf<AscendC::QuePosition::VECCALC> tmpBuf0, tmpBuf1, tmpBuf2;
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
