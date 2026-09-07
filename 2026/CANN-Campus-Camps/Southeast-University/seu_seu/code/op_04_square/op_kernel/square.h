/*!
 * \file square.h
 * \brief High performance Square kernel
 */

#ifndef SQUARE_H
#define SQUARE_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "square_tiling_data.h"
#include "square_tiling_key.h"

namespace NsSquare {

using namespace AscendC;


template <typename T>
class Square {
public:
    __aicore__ inline Square()
    {
    }


    __aicore__ inline void Init(
        GM_ADDR input_x,
        GM_ADDR output,
        const SquareTilingData* tilingData);


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
 * Static UB address.
 *
 * input:
 *     0x00000
 *
 * output:
 *     0x10000
 *
 * 16KB tile means no overlap.
 */
constexpr uint32_t INPUT_LOCAL_ADDR =
    0x00000;

constexpr uint32_t OUTPUT_LOCAL_ADDR =
    0x10000;


/*
 * ================================================================
 * Init
 * ================================================================
 */
template <typename T>
__aicore__ inline void Square<T>::Init(
    GM_ADDR input_x,
    GM_ADDR output,
    const SquareTilingData* tilingData)
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

    } else if (
        remain >
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
        reinterpret_cast<
            __gm__ T*>(
                input_x) +
            blockOffset);


    outputGMY.SetGlobalBuffer(
        reinterpret_cast<
            __gm__ T*>(
                output) +
            blockOffset);
}


/*
 * ================================================================
 * CopyIn
 * ================================================================
 */
template <typename T>
__aicore__ inline void Square<T>::CopyIn(
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


    constexpr int64_t ALIGN_ELEMENTS =
        32 /
        sizeof(T);


    /*
     * Main aligned path.
     */
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
     * Tail only.
     *
     * DataCopyPad blockLen is measured in bytes.
     */
    AscendC::DataCopyExtParams
        copyParams;

    copyParams.blockCount =
        1;

    copyParams.blockLen =
        static_cast<uint32_t>(
            currentNum *
            sizeof(T));

    copyParams.srcStride =
        0;

    copyParams.dstStride =
        0;

    copyParams.rsv =
        0;


    AscendC::DataCopyPadExtParams<T>
        padParams;

    padParams.isPad =
        false;

    padParams.leftPadding =
        0;

    padParams.rightPadding =
        0;

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
 * ================================================================
 */
template <typename T>
__aicore__ inline void Square<T>::Compute(
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
     * Vector processing width.
     *
     * FP32:
     *
     *     PAR = 64 elements
     *
     * FP16:
     *
     *     PAR = 128 elements
     */
    constexpr uint64_t PAR =
        256 /
        sizeof(T);


    /*
     * Host aligns normal Core lengths to exactly 256 bytes.
     *
     * Therefore almost all cores enter this branch.
     *
     * With 16KB/core:
     *
     * FP32:
     *     repeat = 64
     *
     * FP16:
     *     repeat = 64
     *
     * repeatTimes is uint8_t, so this is comfortably below 255.
     */
    if ((currentNum %
         static_cast<int64_t>(PAR)) == 0) {

        const uint8_t repeatTimes =
            static_cast<uint8_t>(
                currentNum /
                static_cast<int64_t>(
                    PAR));


        AscendC::Mul(
            outputLocal,
            inputLocal,
            inputLocal,
            PAR,
            repeatTimes,
            {
                1, 1, 1,
                8, 8, 8
            });

        return;
    }


    /*
     * Only irregular final tail takes the generic API.
     */
    AscendC::Mul(
        outputLocal,
        inputLocal,
        inputLocal,
        static_cast<int32_t>(
            currentNum));
}


/*
 * ================================================================
 * CopyOut
 * ================================================================
 */
template <typename T>
__aicore__ inline void Square<T>::CopyOut(
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
        32 /
        sizeof(T);


    if ((currentNum %
         ALIGN_ELEMENTS) == 0) {

        AscendC::DataCopy(
            outputGMY[offset],
            outputLocal,
            static_cast<uint32_t>(
                currentNum));

        return;
    }


    AscendC::DataCopyExtParams
        copyParams;

    copyParams.blockCount =
        1;

    copyParams.blockLen =
        static_cast<uint32_t>(
            currentNum *
            sizeof(T));

    copyParams.srcStride =
        0;

    copyParams.dstStride =
        0;

    copyParams.rsv =
        0;


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
__aicore__ inline void Square<T>::Process()
{
    if (blockLength_ <= 0 ||
        ubLength_ <= 0) {
        return;
    }


    const int64_t loopCount =
        (blockLength_ +
         ubLength_ -
         1) /
        ubLength_;


    /*
     * Normally loopCount == 1.
     *
     * Only huge tensors where:
     *
     * total data >
     * maxAIV * 16KB
     *
     * enter the multi-tile fallback.
     */
    for (int64_t i = 0;
         i < loopCount;
         ++i) {

        const int64_t remain =
            blockLength_ -
            i *
            ubLength_;


        const int64_t currentNum =
            (remain >
             ubLength_)
                ? ubLength_
                : remain;


        /*
         * GM -> UB
         */
        CopyIn(
            i,
            currentNum);


        AscendC::SetFlag<
            AscendC::HardEvent::MTE2_V>(
                EVENT_ID0);

        AscendC::WaitFlag<
            AscendC::HardEvent::MTE2_V>(
                EVENT_ID0);


        /*
         * If the previous output is still being copied from this same
         * static output buffer, wait before overwriting it.
         *
         * Main one-tile path has no extra event here.
         */
        if (i != 0) {

            AscendC::WaitFlag<
                AscendC::HardEvent::MTE3_V>(
                    EVENT_ID1);
        }


        /*
         * x^2
         */
        Compute(
            currentNum);


        AscendC::SetFlag<
            AscendC::HardEvent::V_MTE3>(
                EVENT_ID0);

        AscendC::WaitFlag<
            AscendC::HardEvent::V_MTE3>(
                EVENT_ID0);


        /*
         * UB -> GM
         */
        CopyOut(
            i,
            currentNum);


        /*
         * Only establish backward dependency when another tile exists.
         *
         * This means the important one-tile fast path pays no
         * MTE3->V event overhead.
         */
        if (i + 1 <
            loopCount) {

            AscendC::SetFlag<
                AscendC::HardEvent::MTE3_V>(
                    EVENT_ID1);
        }
    }
}

} // namespace NsSquare

#endif

