#include "kernel_operator.h"
#include "mul_tiling.h"
#include "tiling_key_mul.h"

template <class DT_X>
class KernelMul {
public:
    __aicore__ inline KernelMul() {}

    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z) {
        constexpr uint32_t inputBytesPerCore =
            (sizeof(DT_X) == 2U)
                ? mul_config::FP16_INPUT_BYTES_PER_CORE
                : mul_config::FP32_INPUT_BYTES_PER_CORE;
        constexpr uint32_t blockLength =
            inputBytesPerCore / sizeof(DT_X);

        const uint32_t blockIdx =
            static_cast<uint32_t>(AscendC::GetBlockIdx());
        const uint32_t offset = blockIdx * blockLength;

        xGm.SetGlobalBuffer((__gm__ DT_X *)x + offset);
        yGm.SetGlobalBuffer((__gm__ DT_X *)y + offset);
        zGm.SetGlobalBuffer((__gm__ DT_X *)z + offset);
    }

    __aicore__ inline void Process() {
        constexpr uint32_t inputBytesPerCore =
            (sizeof(DT_X) == 2U)
                ? mul_config::FP16_INPUT_BYTES_PER_CORE
                : mul_config::FP32_INPUT_BYTES_PER_CORE;
        constexpr uint32_t blockLength =
            inputBytesPerCore / sizeof(DT_X);

        AscendC::LocalTensor<DT_X> xLocal(
            AscendC::TPosition::VECIN, 0, blockLength);
        AscendC::LocalTensor<DT_X> yLocal(
            AscendC::TPosition::VECIN,
            inputBytesPerCore + mul_config::UB_PAD_BYTES,
            blockLength);
        AscendC::LocalTensor<DT_X> zLocal(
            AscendC::TPosition::VECOUT,
            mul_config::Z_UB_ADDR,
            blockLength);

        CopyIn(xLocal, yLocal);
        Compute(xLocal, yLocal, zLocal);
        CopyOut(zLocal);
    }

private:
    __aicore__ inline void CopyIn(
        AscendC::LocalTensor<DT_X> xLocal,
        AscendC::LocalTensor<DT_X> yLocal) {
        constexpr uint32_t inputBytesPerCore =
            (sizeof(DT_X) == 2U)
                ? mul_config::FP16_INPUT_BYTES_PER_CORE
                : mul_config::FP32_INPUT_BYTES_PER_CORE;
        constexpr uint32_t blockLength =
            inputBytesPerCore / sizeof(DT_X);

        AscendC::DataCopy(xLocal, xGm, blockLength);
        AscendC::DataCopy(yLocal, yGm, blockLength);

        AscendC::SetFlag<AscendC::HardEvent::MTE2_V>(0);
        AscendC::WaitFlag<AscendC::HardEvent::MTE2_V>(0);
    }

    __aicore__ inline void Compute(
        AscendC::LocalTensor<DT_X> xLocal,
        AscendC::LocalTensor<DT_X> yLocal,
        AscendC::LocalTensor<DT_X> zLocal) {
        constexpr uint32_t inputBytesPerCore =
            (sizeof(DT_X) == 2U)
                ? mul_config::FP16_INPUT_BYTES_PER_CORE
                : mul_config::FP32_INPUT_BYTES_PER_CORE;
        constexpr uint64_t vectorMask = 256U / sizeof(DT_X);
        constexpr uint8_t repeatTimes =
            static_cast<uint8_t>(inputBytesPerCore / 256U);

        AscendC::Mul(
            zLocal, xLocal, yLocal,
            vectorMask, repeatTimes,
            {1, 1, 1, 8, 8, 8});

        AscendC::SetFlag<AscendC::HardEvent::V_MTE3>(0);
        AscendC::WaitFlag<AscendC::HardEvent::V_MTE3>(0);
    }

    __aicore__ inline void CopyOut(
        AscendC::LocalTensor<DT_X> zLocal) {
        constexpr uint32_t inputBytesPerCore =
            (sizeof(DT_X) == 2U)
                ? mul_config::FP16_INPUT_BYTES_PER_CORE
                : mul_config::FP32_INPUT_BYTES_PER_CORE;
        constexpr uint32_t blockLength =
            inputBytesPerCore / sizeof(DT_X);

        AscendC::DataCopy(zGm, zLocal, blockLength);
    }

private:
    AscendC::GlobalTensor<DT_X> xGm;
    AscendC::GlobalTensor<DT_X> yGm;
    AscendC::GlobalTensor<DT_X> zGm;
};

template <typename DT_X>
__global__ __aicore__ void mul(
    GM_ADDR x, GM_ADDR y, GM_ADDR z,
    GM_ADDR workspace, GM_ADDR tiling) {
    REGISTER_TILING_DEFAULT(MulTilingData);
    AscendC::InitSocState();

    KernelMul<DT_X> op;
    op.Init(x, y, z);
    op.Process();
}