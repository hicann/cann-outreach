// Kernel侧核函数实现 — 裸UB + 显式同步版本
// 优化点:
//   1. 用 TBuf 直接取 UB 地址(跳过 TQue 的 AllocTensor/EnQue/DeQue/FreeTensor 簿记)
//   2. 两条 DataCopy 搬入后合并为一次 MTE2_V 事件等待(原 TQue 需等 x 搬完 + 等 y 搬完两次)
//   3. 原地 Mul(xLocal, xLocal, yLocal, ...) 省掉独立输出缓冲(省 1/3 UB 与一次 InitBuffer)
//   4. 单遍搬入-计算-搬出(本形状下每核数据量 < ubFormer, 双缓冲 tile 循环退化为 1 次,
//      直接单遍处理, 消除循环开销)
#include "kernel_operator.h"

#include "mul_tiling.h"
#include "tiling_key_mul.h"

template <class DT_X>
class KernelMul {
public:
    __aicore__ inline KernelMul() {}

    // 绑定本核 GM 分片, 分配 UB 缓冲(x 和 y, 输出复用 x 的缓冲)
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, const MulTilingData &tilingData)
    {
        uint32_t blockIdx = static_cast<uint32_t>(AscendC::GetBlockIdx());
        bool isLastBlock = (blockIdx == tilingData.blockNum - 1);

        uint32_t offset = tilingData.blockFormer * blockIdx;
        uint32_t length = isLastBlock
            ? (tilingData.dim0 - tilingData.blockFormer * (tilingData.blockNum - 1))
            : tilingData.blockFormer;
        this->blockLength_ = length;

        if (length == 0) {
            return;
        }

        xGm_.SetGlobalBuffer((__gm__ DT_X *)x + offset, length);
        yGm_.SetGlobalBuffer((__gm__ DT_X *)y + offset, length);
        zGm_.SetGlobalBuffer((__gm__ DT_X *)z + offset, length);

        // 只分配 x 和 y 的 UB 缓冲; z 结果直接写回 xLocal, 无需独立输出缓冲
        pipe_.InitBuffer(bufX_, length * sizeof(DT_X));
        pipe_.InitBuffer(bufY_, length * sizeof(DT_X));
    }

    // 单遍: 搬入 → 等 MTE2 → 计算 → 等 V → 搬出
    __aicore__ inline void Process()
    {
        if (this->blockLength_ == 0) {
            return;
        }
        AscendC::LocalTensor<DT_X> xLocal = bufX_.Get<DT_X>();
        AscendC::LocalTensor<DT_X> yLocal = bufY_.Get<DT_X>();

        // 搬入: 两条 GM->UB 异步发射
        AscendC::DataCopy(xLocal, xGm_, this->blockLength_);
        AscendC::DataCopy(yLocal, yGm_, this->blockLength_);

        // 等搬入全部完成(合并为一次 MTE2 事件等待)
        int32_t evtMte2 = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE2_V));
        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(evtMte2);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(evtMte2);

        // 计算: z = x * y, 结果原地写回 xLocal(逐元素读-算-写, 同地址安全)
        AscendC::Mul(xLocal, xLocal, yLocal, this->blockLength_);

        // 等计算完成后再搬出
        int32_t evtV = static_cast<int32_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::V_MTE3));
        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(evtV);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(evtV);

        // 搬出: UB -> GM
        AscendC::DataCopy(zGm_, xLocal, this->blockLength_);
    }

private:
    AscendC::TPipe pipe_;                                     // 仅用于分配 UB 地址(不再走队列)
    AscendC::TBuf<AscendC::TPosition::VECIN> bufX_, bufY_;   // 输入 UB 缓冲(输出复用 bufX_)
    AscendC::GlobalTensor<DT_X> xGm_, yGm_, zGm_;
    uint32_t blockLength_;                                    // 本核元素个数
};

// 核函数入口: 保持 REGISTER_TILING_DEFAULT / GET_TILING_DATA 启动机制不变
template <typename DT_X>
__global__ __aicore__ void mul(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(MulTilingData);
    GET_TILING_DATA_WITH_STRUCT(MulTilingData, tiling_data, tiling);
    KernelMul<DT_X> op;
    op.Init(x, y, z, tiling_data);
    op.Process();
}