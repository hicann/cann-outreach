#include "kernel_operator.h"

constexpr uint32_t GCD123_NDIM = 4;

class KernelGcd123 {
public:
    __aicore__ inline KernelGcd123() {}

    __aicore__ inline void Init(GM_ADDR self, GM_ADDR other, GM_ADDR out,
                                int64_t selfLength, int64_t otherLength, int64_t totalLength,
                                const int64_t* selfShape, const int64_t* otherShape, const int64_t* outShape)
    {
        this->totalLength = totalLength;

        selfGm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(self), static_cast<uint64_t>(selfLength));
        otherGm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(other), static_cast<uint64_t>(otherLength));
        outGm.SetGlobalBuffer(reinterpret_cast<__gm__ half*>(out), static_cast<uint64_t>(totalLength));

        for (uint32_t i = 0; i < GCD123_NDIM; ++i) {
            this->selfShape[i] = selfShape[i];
            this->otherShape[i] = otherShape[i];
            this->outShape[i] = outShape[i];
        }

        ComputeStrides(this->selfShape, this->selfStride);
        ComputeStrides(this->otherShape, this->otherStride);
    }

    __aicore__ inline void Process()
    {
        if (totalLength <= 0) {
            return;
        }

        uint32_t blockNum = AscendC::GetBlockNum();
        uint32_t blockIdx = AscendC::GetBlockIdx();
        if (blockNum == 0) {
            return;
        }

        uint64_t total = static_cast<uint64_t>(totalLength);
        uint64_t perBlock = (total + blockNum - 1) / blockNum;
        uint64_t start = static_cast<uint64_t>(blockIdx) * perBlock;
        uint64_t end = start + perBlock;
        if (end > total) {
            end = total;
        }

        for (uint64_t i = start; i < end; ++i) {
            int64_t selfOffset = GetBroadcastOffset(static_cast<int64_t>(i), selfShape, selfStride);
            int64_t otherOffset = GetBroadcastOffset(static_cast<int64_t>(i), otherShape, otherStride);
            half result = GcdHalf(selfGm.GetValue(selfOffset), otherGm.GetValue(otherOffset));
            outGm.SetValue(i, result);
        }
    }

private:
    __aicore__ inline void ComputeStrides(const int64_t* shape, int64_t* stride)
    {
        stride[GCD123_NDIM - 1] = 1;
        for (int32_t i = GCD123_NDIM - 2; i >= 0; --i) {
            stride[i] = stride[i + 1] * shape[i + 1];
        }
    }

    __aicore__ inline int64_t GetBroadcastOffset(int64_t outIndex, const int64_t* shape, const int64_t* stride) const
    {
        int64_t remaining = outIndex;
        int64_t offset = 0;
        for (int32_t i = GCD123_NDIM - 1; i >= 0; --i) {
            int64_t coord = remaining % outShape[i];
            remaining /= outShape[i];
            offset += (coord % shape[i]) * stride[i];
        }
        return offset;
    }

    __aicore__ inline half GcdHalf(half lhs, half rhs) const
    {
        float l = static_cast<float>(lhs);
        float r = static_cast<float>(rhs);
        int64_t a = static_cast<int64_t>(l);
        int64_t b = static_cast<int64_t>(r);

        if (a < 0) {
            a = -a;
        }
        if (b < 0) {
            b = -b;
        }

        while (b != 0) {
            int64_t remainder = a % b;
            a = b;
            b = remainder;
        }
        return static_cast<half>(static_cast<float>(a));
    }

private:
    AscendC::GlobalTensor<half> selfGm;
    AscendC::GlobalTensor<half> otherGm;
    AscendC::GlobalTensor<half> outGm;

    int64_t selfShape[GCD123_NDIM];
    int64_t otherShape[GCD123_NDIM];
    int64_t outShape[GCD123_NDIM];
    int64_t selfStride[GCD123_NDIM];
    int64_t otherStride[GCD123_NDIM];
    int64_t totalLength;
};

extern "C" __global__ __aicore__ void gcd123(GM_ADDR self, GM_ADDR other, GM_ADDR out,
                                              GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);

    KernelGcd123 op;
    op.Init(self, other, out, tilingData.selfLength, tilingData.otherLength, tilingData.totalLength,
            tilingData.selfShape, tilingData.otherShape, tilingData.outShape);
    op.Process();
}
