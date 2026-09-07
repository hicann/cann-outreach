//v5
#include "kernel_operator.h"
#include "gelu_tiling.h"
#include "tiling_key_gelu.h"

constexpr int32_t BUFFER_NUM = 2;

template <typename T>
class KernelGelu {
public:
    __aicore__ inline void Init(GM_ADDR input_x, GM_ADDR output, const GeluTilingData &tiling)
    {
        // 必须保留 AscendC::，否则 cannjudge 的 INFERCHANNEL 编译阶段报错。
        const uint64_t coreIdx = AscendC::GetBlockIdx();
        const uint64_t start = coreIdx * tiling.blockFactor;
        blockLength_ = start < tiling.length ? tiling.length - start : 0;
        if (blockLength_ > tiling.blockFactor) {
            blockLength_ = tiling.blockFactor;
        }
        tileLength_ = tiling.ubFactor;
        inputGM_.SetGlobalBuffer((__gm__ T *)input_x + start, blockLength_);
        outputGM_.SetGlobalBuffer((__gm__ T *)output + start, blockLength_);
        pipe_.InitBuffer(inQueue_, BUFFER_NUM, tileLength_ * sizeof(T));
        pipe_.InitBuffer(outQueue_, BUFFER_NUM, tileLength_ * sizeof(T));
        pipe_.InitBuffer(floatInput_, tileLength_ * sizeof(float));
        pipe_.InitBuffer(erfcInput_, tileLength_ * sizeof(float));
        if constexpr (sizeof(T) != sizeof(float)) {
            pipe_.InitBuffer(result_, tileLength_ * sizeof(float));
        }
        pipe_.InitBuffer(erfcScratch_, tiling.erfcTmpBytes);
    }

    __aicore__ inline void Process()
    {
        if (blockLength_ == 0) {
            return;
        }
        CopyIn(0, static_cast<uint32_t>(blockLength_ < tileLength_ ? blockLength_ : tileLength_));
        for (uint64_t offset = 0; offset < blockLength_; offset += tileLength_) {
            // 在处理当前块前提交下一块搬运，输入队列最多持有两个 tile。
            const uint64_t next = offset + tileLength_;
            if (next < blockLength_) {
                const uint32_t nextCount = static_cast<uint32_t>(
                    blockLength_ - next < tileLength_ ? blockLength_ - next : tileLength_);
                CopyIn(next, nextCount);
            }
            const uint32_t count = static_cast<uint32_t>(
                blockLength_ - offset < tileLength_ ? blockLength_ - offset : tileLength_);
            Compute(count);
            CopyOut(offset, count);
        }
    }

private:
    __aicore__ inline void CopyIn(uint64_t offset, uint32_t count)
    {
        auto x = inQueue_.AllocTensor<T>();
        if (count % (32 / sizeof(T)) == 0) {
            AscendC::DataCopy(x, inputGM_[offset], count);
            inQueue_.EnQue(x);
            return;
        }
        const AscendC::DataCopyExtParams copy{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
        const uint32_t align = 32 / sizeof(T);
        const AscendC::DataCopyPadExtParams<T> pad{
            true, 0, static_cast<uint8_t>((align - count % align) % align), static_cast<T>(0)};
        AscendC::DataCopyPad(x, inputGM_[offset], copy, pad);
        inQueue_.EnQue(x);
    }

    __aicore__ inline void Compute(uint32_t count)
    {
        // 保留已验证的 64 元素计算粒度及尾块补零。
        const uint32_t calcCount = (count + 63) / 64 * 64;
        auto input = inQueue_.DeQue<T>();
        auto output = outQueue_.AllocTensor<T>();
        auto x = floatInput_.Get<float>();
        auto z = erfcInput_.Get<float>();
        AscendC::LocalTensor<float> y;
        if constexpr (sizeof(T) == sizeof(float)) {
            y = output.template ReinterpretCast<float>();
        } else {
            y = result_.Get<float>();
        }
        auto scratch = erfcScratch_.Get<uint8_t>();
        if (calcCount != count) {
            // DataCopyPad 仅补到 32B，不能直接读取到 256B 边界。
            // 先初始化整个 float 工作区，再覆盖真实输入；不读取 GM 尾部。
            AscendC::Duplicate(x, 0.0f, calcCount);
        }
        if constexpr (sizeof(T) == sizeof(float)) {
            if (calcCount == count) {
                x = input.template ReinterpretCast<float>();
            } else {
                AscendC::Muls(x, input.template ReinterpretCast<float>(), 1.0f, count);
            }
        } else {
            AscendC::Cast(x, input, AscendC::RoundMode::CAST_NONE, count);
        }

        // GELU(x) = max(x, 0) - |x| * Phi(-|x|)。无需 CompareScalar / Select。
        // 截断的只是尾项自变量，|x| >= 14 时尾项小于 1.1e-43。
        AscendC::Abs(z, x, calcCount);
        AscendC::Mins(z, z, 14.0f, calcCount);
        AscendC::Muls(z, z, 0.7071067811865475244f, calcCount);
        AscendC::Erfc<float, false>(y, z, scratch, calcCount);
        AscendC::Mul(y, y, z, calcCount);
        AscendC::Muls(y, y, 0.7071067811865475244f, calcCount);
        AscendC::Maxs(z, x, 0.0f, calcCount);
        AscendC::Sub(y, z, y, calcCount);
        if constexpr (sizeof(T) != sizeof(float)) {
            AscendC::Cast(output, y, AscendC::RoundMode::CAST_RINT, count);
        }
        outQueue_.EnQue(output);
        inQueue_.FreeTensor(input);
    }

    __aicore__ inline void CopyOut(uint64_t offset, uint32_t count)
    {
        auto output = outQueue_.DeQue<T>();
        if (count % (32 / sizeof(T)) == 0) {
            AscendC::DataCopy(outputGM_[offset], output, count);
            outQueue_.FreeTensor(output);
            return;
        }
        const AscendC::DataCopyExtParams copy{1, static_cast<uint32_t>(count * sizeof(T)), 0, 0, 0};
        // 只写真实字节数，UB 中的 padding 不写到 GM。
        AscendC::DataCopyPad(outputGM_[offset], output, copy);
        outQueue_.FreeTensor(output);
    }

    AscendC::TPipe pipe_;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueue_;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueue_;
    AscendC::GlobalTensor<T> inputGM_, outputGM_;
    AscendC::TBuf<AscendC::TPosition::VECCALC> floatInput_, erfcInput_, result_, erfcScratch_;
    uint64_t blockLength_ = 0;
    uint32_t tileLength_ = 0;
};

extern "C" __global__ __aicore__ void gelu(GM_ADDR input_x, GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(GeluTilingData);
    GET_TILING_DATA_WITH_STRUCT(GeluTilingData, tiling_data, tiling);
    // 保留字面量供编译工具识别；1/2 分别对应 GELU_TILING_FP16/FP32。
    if (TILING_KEY_IS(1)) {
        KernelGelu<half> op;
        op.Init(input_x, output, tiling_data);
        op.Process();
    } else if (TILING_KEY_IS(2)) {
        KernelGelu<float> op;
        op.Init(input_x, output, tiling_data);
        op.Process();
    }
}


