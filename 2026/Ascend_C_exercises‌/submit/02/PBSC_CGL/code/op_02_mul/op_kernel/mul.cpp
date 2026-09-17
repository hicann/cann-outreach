// Kernel侧核函数实现
#include "kernel_operator.h"

#include "mul_tiling.h"
#include "tiling_key_mul.h"

constexpr uint32_t EVENT_ID = 0;

// 额外空32B，用于错开不同Tensor的UB bank group
constexpr uint32_t BANK_PAD_BYTES = 32;

template <class DT_X>
class KernelMul {
public:
    __aicore__ inline KernelMul() {}

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR y,
        GM_ADDR z,
        uint32_t blockLength)
    {
        blockLength_ = blockLength;

        const uint32_t gmOffset =
            AscendC::GetBlockIdx() * blockLength_;

        xGm_.SetGlobalBuffer(
            (__gm__ DT_X *)x + gmOffset,
            blockLength_);

        yGm_.SetGlobalBuffer(
            (__gm__ DT_X *)y + gmOffset,
            blockLength_);

        zGm_.SetGlobalBuffer(
            (__gm__ DT_X *)z + gmOffset,
            blockLength_);
    }

    __aicore__ inline void Process()
    {
        /*
         * Static Tensor:
         *
         * x : 0
         *
         * y : blockBytes + 32B
         *
         * z : 2 * (blockBytes + 32B)
         *
         * 额外32B padding：
         * 避免x/y落到相同bank group，
         * 同时降低src/dst bank conflict。
         */

        const uint32_t blockBytes =
            blockLength_ * sizeof(DT_X);

        const uint32_t tensorStride =
            blockBytes + BANK_PAD_BYTES;

        AscendC::LocalTensor<DT_X> xLocal(
            AscendC::TPosition::VECCALC,
            0,
            blockLength_);

        AscendC::LocalTensor<DT_X> yLocal(
            AscendC::TPosition::VECCALC,
            tensorStride,
            blockLength_);

        AscendC::LocalTensor<DT_X> zLocal(
            AscendC::TPosition::VECCALC,
            tensorStride * 2,
            blockLength_);

        // ========================================
        // GM -> UB
        // 一次搬完2048个元素
        // ========================================

        AscendC::DataCopy(
            xLocal,
            xGm_,
            blockLength_);

        AscendC::DataCopy(
            yLocal,
            yGm_,
            blockLength_);

        /*
         * MTE2 -> Vector
         *
         * 等待x/y搬运完成后再进行Mul
         */
        AscendC::SetFlag<
            AscendC::HardEvent::MTE2_V>(
                EVENT_ID);

        AscendC::WaitFlag<
            AscendC::HardEvent::MTE2_V>(
                EVENT_ID);

        // ========================================
        // Vector Compute
        // z = x * y
        // ========================================

        AscendC::Mul(
            zLocal,
            xLocal,
            yLocal,
            blockLength_);

        /*
         * Vector -> MTE3
         *
         * 等待Mul完成后再写回GM
         */
        AscendC::SetFlag<
            AscendC::HardEvent::V_MTE3>(
                EVENT_ID);

        AscendC::WaitFlag<
            AscendC::HardEvent::V_MTE3>(
                EVENT_ID);

        // ========================================
        // UB -> GM
        // ========================================

        AscendC::DataCopy(
            zGm_,
            zLocal,
            blockLength_);
    }

private:
    AscendC::GlobalTensor<DT_X> xGm_;
    AscendC::GlobalTensor<DT_X> yGm_;
    AscendC::GlobalTensor<DT_X> zGm_;

    uint32_t blockLength_;
};


template <typename DT_X>
__global__ __aicore__ void mul(
    GM_ADDR x,
    GM_ADDR y,
    GM_ADDR z,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    /*
     * Static Tensor编程必须初始化Soc状态
     */
    AscendC::InitSocState();

    REGISTER_TILING_DEFAULT(
        MulTilingData);

    GET_TILING_DATA_WITH_STRUCT(
        MulTilingData,
        tiling_data,
        tiling);

    KernelMul<DT_X> op;

    op.Init(
        x,
        y,
        z,
        tiling_data.blockLength);

    op.Process();
}