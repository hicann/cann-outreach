// // Kernel侧核函数实现
// #include "kernel_operator.h"

// #include "add_tiling.h"
// #include "tiling_key_add.h"

// template <class DT_X>
// class KernelAdd {
// public:
//     __aicore__ inline KernelAdd() {}
//     __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t length) {

//     }
//     __aicore__ inline void Process() {

//     }
// private:

// };

// template <typename DT_X>
//  __global__ __aicore__ void add(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling) {
//     REGISTER_TILING_DEFAULT(AddTilingData);
//     GET_TILING_DATA_WITH_STRUCT(AddTilingData, tiling_data, tiling);
//     KernelAdd<DT_X> op;
//     op.Init(x, y, z, tiling_data.length);
//     op.Process();
// }
#include "kernel_operator.h"

#include "add_tiling.h"
#include "tiling_key_add.h"

namespace NsAdd {

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 1;

template <typename T>
class KernelAdd {
public:
    __aicore__ inline KernelAdd()
    {
    }

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR y,
        GM_ADDR z,
        const AddTilingData* tiling)
    {
        // 当前核在GM中的起始位置
        int64_t blockOffset =
            static_cast<int64_t>(
                tiling->blockFactor)
            * GetBlockIdx();

        int64_t remainderLength =
            static_cast<int64_t>(
                tiling->length)
            - blockOffset;

        // 处理最后一个核数据量可能不足的问题
        blockLength_ =
            remainderLength >
                    static_cast<int64_t>(
                        tiling->blockFactor)
                ? static_cast<int64_t>(
                      tiling->blockFactor)
                : remainderLength;

        if (blockLength_ < 0) {
            blockLength_ = 0;
        }

        ubLength_ =
            tiling->ubFactor > 0
                ? static_cast<int64_t>(
                      tiling->ubFactor)
                : blockLength_;

        if (ubLength_ <= 0) {
            ubLength_ = 1;
        }

        // 设置当前核处理的GM范围
        xGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ T*>(x)
                + blockOffset,
            blockLength_);

        yGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ T*>(y)
                + blockOffset,
            blockLength_);

        zGm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ T*>(z)
                + blockOffset,
            blockLength_);

        // 两个输入队列和一个输出队列
        pipe_.InitBuffer(
            inputQueueX_,
            BUFFER_NUM,
            ubLength_ * sizeof(T));

        pipe_.InitBuffer(
            inputQueueY_,
            BUFFER_NUM,
            ubLength_ * sizeof(T));

        pipe_.InitBuffer(
            outputQueueZ_,
            BUFFER_NUM,
            ubLength_ * sizeof(T));
    }

    __aicore__ inline void Process()
    {
        if (blockLength_ <= 0) {
            return;
        }

        int64_t loopCount =
            (blockLength_ + ubLength_ - 1)
            / ubLength_;

        for (int64_t progress = 0;
             progress < loopCount;
             ++progress) {
            int64_t currentNum =
                blockLength_
                - progress * ubLength_;

            if (currentNum > ubLength_) {
                currentNum = ubLength_;
            }

            CopyIn(progress, currentNum);
            Compute(currentNum);
            CopyOut(progress, currentNum);
        }
    }

private:
    __aicore__ inline void CopyIn(
        int64_t progress,
        int64_t currentNum)
    {
        LocalTensor<T> xLocal =
            inputQueueX_
                .template AllocTensor<T>();

        LocalTensor<T> yLocal =
            inputQueueY_
                .template AllocTensor<T>();

        int64_t offset =
            progress * ubLength_;

        // 数据满足32字节对齐时使用普通DataCopy
        if ((currentNum * sizeof(T)) % 32 == 0) {
            AscendC::DataCopy(
                xLocal,
                xGm_[offset],
                static_cast<uint32_t>(
                    currentNum));

            AscendC::DataCopy(
                yLocal,
                yGm_[offset],
                static_cast<uint32_t>(
                    currentNum));
        } else {
            // 尾块不足32字节对齐时使用DataCopyPad
            DataCopyParams copyParams {
                1,
                static_cast<uint16_t>(
                    currentNum * sizeof(T)),
                0,
                0
            };

            DataCopyPad(
                xLocal,
                xGm_[offset],
                copyParams,
                {false, 0, 0, 0});

            DataCopyPad(
                yLocal,
                yGm_[offset],
                copyParams,
                {false, 0, 0, 0});
        }

        inputQueueX_.EnQue(xLocal);
        inputQueueY_.EnQue(yLocal);
    }

    __aicore__ inline void Compute(
        int64_t currentNum)
    {
        LocalTensor<T> xLocal =
            inputQueueX_
                .template DeQue<T>();

        LocalTensor<T> yLocal =
            inputQueueY_
                .template DeQue<T>();

        LocalTensor<T> zLocal =
            outputQueueZ_
                .template AllocTensor<T>();

        uint32_t count =
            static_cast<uint32_t>(
                currentNum);

        // 核心计算：z = x + y
        AscendC::Add(
            zLocal,
            xLocal,
            yLocal,
            count);

        outputQueueZ_
            .template EnQue<T>(zLocal);

        inputQueueX_.FreeTensor(xLocal);
        inputQueueY_.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOut(
        int64_t progress,
        int64_t currentNum)
    {
        LocalTensor<T> zLocal =
            outputQueueZ_
                .template DeQue<T>();

        int64_t offset =
            progress * ubLength_;

        if ((currentNum * sizeof(T)) % 32 == 0) {
            AscendC::DataCopy(
                zGm_[offset],
                zLocal,
                static_cast<uint32_t>(
                    currentNum));
        } else {
            DataCopyParams copyParams {
                1,
                static_cast<uint16_t>(
                    currentNum * sizeof(T)),
                0,
                0
            };

            DataCopyPad(
                zGm_[offset],
                zLocal,
                copyParams);
        }

        outputQueueZ_.FreeTensor(zLocal);
    }

private:
    TPipe pipe_;

    TQue<QuePosition::VECIN, BUFFER_NUM>
        inputQueueX_;

    TQue<QuePosition::VECIN, BUFFER_NUM>
        inputQueueY_;

    TQue<QuePosition::VECOUT, BUFFER_NUM>
        outputQueueZ_;

    GlobalTensor<T> xGm_;
    GlobalTensor<T> yGm_;
    GlobalTensor<T> zGm_;

    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
};

}  // namespace NsAdd

template <typename DT_INPUT_X>
__global__ __aicore__ void add(
    GM_ADDR x,
    GM_ADDR y,
    GM_ADDR z,
    GM_ADDR workspace,
    GM_ADDR tiling)
{
    (void)workspace;

    REGISTER_TILING_DEFAULT(AddTilingData);

    GET_TILING_DATA_WITH_STRUCT(
        AddTilingData,
        tilingData,
        tiling);

    NsAdd::KernelAdd<DT_INPUT_X> op;

    op.Init(
        x,
        y,
        z,
        &tilingData);

    op.Process();
}