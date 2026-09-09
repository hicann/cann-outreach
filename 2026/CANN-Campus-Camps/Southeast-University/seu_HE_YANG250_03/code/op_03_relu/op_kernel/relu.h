/*!
 * \file relu.h
 * \brief High-performance Relu kernel
 */

#ifndef RELU_H
#define RELU_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "relu_tiling_data.h"
#include "relu_tiling_key.h"

namespace NsRelu {

using namespace AscendC;

/*
 * Static UB layout:
 *
 * input  : [0x00000, 0x0C000)
 * output : [0x0C000, 0x18000)
 *
 * Each region is at most 48KB.
 */
constexpr uint32_t INPUT_LOCAL_ADDR  = 0x00000;
constexpr uint32_t OUTPUT_LOCAL_ADDR = 0x0C000;


template <typename T>
class Relu {
public:
    __aicore__ inline Relu() {}

    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR y,
        const ReluTilingData* tilingData);

    __aicore__ inline void Process();

private:
    __aicore__ inline void CopyIn(
        int64_t progress,
        int64_t currentNum);

    __aicore__ inline void Compute(
        int64_t currentNum);

    __aicore__ inline void CopyOut(
        int64_t progress,
        int64_t currentNum);

private:
    GlobalTensor<T> inputGMX;
    GlobalTensor<T> outputGMY;

    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
};


/*
 * ================================================================
 * Init
 * ================================================================
 */
template <typename T>
__aicore__ inline void Relu<T>::Init(
    GM_ADDR x,
    GM_ADDR y,
    const ReluTilingData* tilingData)
{
    const int64_t blockOffset =
        tilingData->blockFactor *
        static_cast<int64_t>(
            AscendC::GetBlockIdx());

    const int64_t remain =
        tilingData->totalNum -
        blockOffset;

    if (remain <= 0) {
        blockLength_ = 0;
    } else if (remain >
               tilingData->blockFactor) {
        blockLength_ =
            tilingData->blockFactor;
    } else {
        blockLength_ =
            remain;
    }

    ubLength_ =
        tilingData->ubFactor;

    inputGMX.SetGlobalBuffer(
        reinterpret_cast<__gm__ T*>(x) +
        blockOffset);

    outputGMY.SetGlobalBuffer(
        reinterpret_cast<__gm__ T*>(y) +
        blockOffset);
}


/*
 * ================================================================
 * CopyIn
 *
 * GM -> VECIN
 * ================================================================
 */
template <typename T>
__aicore__ inline void Relu<T>::CopyIn(
    int64_t progress,
    int64_t currentNum)
{
    AscendC::LocalTensor<T> inputLocal(
        AscendC::TPosition::VECIN,
        INPUT_LOCAL_ADDR,
        static_cast<uint32_t>(
            ubLength_));

    const int64_t offset =
        progress *
        ubLength_;

    /*
     * Fast aligned path.
     *
     * FP32: 8 elements = 32B
     * FP16: 16 elements = 32B
     */
    constexpr int64_t ALIGN_ELEMENTS =
        32 / sizeof(T);

    if ((currentNum %
         ALIGN_ELEMENTS) == 0) {

        AscendC::DataCopy(
            inputLocal,
            inputGMX[offset],
            static_cast<uint32_t>(
                currentNum));

        return;
    }

    /*
     * Generic tail path.
     *
     * Does not affect the standard [8,2048] hot path.
     */
    AscendC::DataCopyExtParams copyParams;

    copyParams.blockCount = 1;
    copyParams.blockLen =
        static_cast<uint32_t>(
            currentNum *
            sizeof(T));
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    copyParams.rsv = 0;

    AscendC::DataCopyPadExtParams<T> padParams;

    padParams.isPad = false;
    padParams.leftPadding = 0;
    padParams.rightPadding = 0;
    padParams.paddingValue =
        static_cast<T>(0);

    AscendC::DataCopyPad(
        inputLocal,
        inputGMX[offset],
        copyParams,
        padParams);
}


/*
 * ================================================================
 * Compute
 *
 * y = max(0, x)
 * ================================================================
 */
template <typename T>
__aicore__ inline void Relu<T>::Compute(
    int64_t currentNum)
{
    AscendC::LocalTensor<T> inputLocal(
        AscendC::TPosition::VECIN,
        INPUT_LOCAL_ADDR,
        static_cast<uint32_t>(
            ubLength_));

    AscendC::LocalTensor<T> outputLocal(
        AscendC::TPosition::VECOUT,
        OUTPUT_LOCAL_ADDR,
        static_cast<uint32_t>(
            ubLength_));

    /*
     * One vector repeat = 256 bytes.
     *
     * FP32:
     *     64 elements/repeat
     *
     * FP16:
     *     128 elements/repeat
     */
    constexpr uint64_t PAR =
        256 / sizeof(T);

    /*
     * Fast path:
     *
     * [8,2048]
     *
     * FP32:
     *     2048 / 64 = 32 repeats
     *
     * FP16:
     *     2048 / 128 = 16 repeats
     */
    if ((currentNum %
         static_cast<int64_t>(PAR)) == 0) {

        const int64_t repeat64 =
            currentNum /
            static_cast<int64_t>(PAR);

        /*
         * repeatTimes is uint8_t.
         */
        if (repeat64 <= 255) {

            const uint8_t repeatTimes =
                static_cast<uint8_t>(
                    repeat64);

            AscendC::Relu(
                outputLocal,
                inputLocal,
                PAR,
                repeatTimes,
                {
                    1, 1,
                    8, 8
                });

            return;
        }
    }

    /*
     * Irregular tail / very large tile fallback.
     */
    AscendC::Relu(
        outputLocal,
        inputLocal,
        static_cast<int32_t>(
            currentNum));
}


/*
 * ================================================================
 * CopyOut
 *
 * VECOUT -> GM
 * ================================================================
 */
template <typename T>
__aicore__ inline void Relu<T>::CopyOut(
    int64_t progress,
    int64_t currentNum)
{
    AscendC::LocalTensor<T> outputLocal(
        AscendC::TPosition::VECOUT,
        OUTPUT_LOCAL_ADDR,
        static_cast<uint32_t>(
            ubLength_));

    const int64_t offset =
        progress *
        ubLength_;

    constexpr int64_t ALIGN_ELEMENTS =
        32 / sizeof(T);

    if ((currentNum %
         ALIGN_ELEMENTS) == 0) {

        AscendC::DataCopy(
            outputGMY[offset],
            outputLocal,
            static_cast<uint32_t>(
                currentNum));

        return;
    }

    AscendC::DataCopyExtParams copyParams;

    copyParams.blockCount = 1;
    copyParams.blockLen =
        static_cast<uint32_t>(
            currentNum *
            sizeof(T));
    copyParams.srcStride = 0;
    copyParams.dstStride = 0;
    copyParams.rsv = 0;

    AscendC::DataCopyPad(
        outputGMY[offset],
        outputLocal,
        copyParams);
}


/*
 * ================================================================
 * Process
 * ================================================================
 */
template <typename T>
__aicore__ inline void Relu<T>::Process()
{
    if (blockLength_ <= 0 ||
        ubLength_ <= 0) {
        return;
    }

    const int64_t loopCount =
        (blockLength_ +
         ubLength_ - 1) /
        ubLength_;

    for (int64_t i = 0;
         i < loopCount;
         ++i) {

        const int64_t remain =
            blockLength_ -
            i *
            ubLength_;

        const int64_t currentNum =
            (remain > ubLength_)
                ? ubLength_
                : remain;


        /*
         * Previous Vector must finish reading inputLocal
         * before MTE2 overwrites it.
         */
        if (i != 0) {
            AscendC::WaitFlag<
                AscendC::HardEvent::V_MTE2>(
                    EVENT_ID0);
        }


        /*
         * Copy input.
         */
        CopyIn(
            i,
            currentNum);


        /*
         * MTE2 -> Vector.
         */
        AscendC::SetFlag<
            AscendC::HardEvent::MTE2_V>(
                EVENT_ID0);

        AscendC::WaitFlag<
            AscendC::HardEvent::MTE2_V>(
                EVENT_ID0);


        /*
         * Previous MTE3 must finish reading outputLocal
         * before Vector overwrites it.
         */
        if (i != 0) {
            AscendC::WaitFlag<
                AscendC::HardEvent::MTE3_V>(
                    EVENT_ID0);
        }


        /*
         * ReLU.
         */
        Compute(
            currentNum);


        /*
         * If another tile exists, protect inputLocal
         * before next MTE2 overwrites it.
         */
        if (i + 1 <
            loopCount) {

            AscendC::SetFlag<
                AscendC::HardEvent::V_MTE2>(
                    EVENT_ID0);
        }


        /*
         * Vector -> MTE3.
         */
        AscendC::SetFlag<
            AscendC::HardEvent::V_MTE3>(
                EVENT_ID0);

        AscendC::WaitFlag<
            AscendC::HardEvent::V_MTE3>(
                EVENT_ID0);


        /*
         * Copy result.
         */
        CopyOut(
            i,
            currentNum);


        /*
         * Protect outputLocal before it is reused.
         */
        if (i + 1 <
            loopCount) {

            AscendC::SetFlag<
                AscendC::HardEvent::MTE3_V>(
                    EVENT_ID0);
        }
    }
}

} // namespace NsRelu

#endif // RELU_H
