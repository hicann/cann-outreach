/*!
 * \file square.h
 * \brief Square kernel implementation
 */

#ifndef SQUARE_H
#define SQUARE_H

#include "kernel_operator.h"
#include "square_tiling_data.h"

namespace NsSquare {

using namespace AscendC;


// 一个输入Queue，一个输出Queue。
// 这一版先以正确性和复杂Case为目标。
constexpr int32_t BUFFER_NUM = 1;


template <typename T>
class Square {
public:

    __aicore__ inline Square()
    {
    }


    __aicore__ inline void Init(
        GM_ADDR inputX,
        GM_ADDR output,
        const SquareTilingData* tilingData)
    {
        // -------------------------------------------------
        // 当前Core的起始元素位置
        // -------------------------------------------------

        int64_t blockOffset =
            tilingData->blockFactor *
            AscendC::GetBlockIdx();


        // -------------------------------------------------
        // 计算当前Core真正剩余多少数据
        // -------------------------------------------------

        int64_t remain =
            tilingData->totalNum -
            blockOffset;


        if (remain >
            tilingData->blockFactor) {

            blockLength_ =
                tilingData->blockFactor;

        } else {

            blockLength_ =
                remain;
        }


        ubFactor_ =
            tilingData->ubFactor;


        // -------------------------------------------------
        // 当前Core负责的GM区间
        // -------------------------------------------------

        inputGm_.SetGlobalBuffer(
            (__gm__ T*)inputX +
                blockOffset,
            blockLength_);


        outputGm_.SetGlobalBuffer(
            (__gm__ T*)output +
                blockOffset,
            blockLength_);


        // -------------------------------------------------
        // UB Buffer
        //
        // ubFactor = 1024
        // 对float16和float32都天然32B对齐
        // -------------------------------------------------

        pipe_.InitBuffer(
            inputQueue_,
            BUFFER_NUM,
            ubFactor_ *
                sizeof(T));


        pipe_.InitBuffer(
            outputQueue_,
            BUFFER_NUM,
            ubFactor_ *
                sizeof(T));
    }


    __aicore__ inline void Process()
    {
        // 当前核要循环多少次
        int64_t loopCount =
            (blockLength_ +
             ubFactor_ -
             1) /
            ubFactor_;


        for (int64_t i = 0;
             i < loopCount;
             ++i) {

            // 当前Tile剩余多少有效元素
            int64_t remain =
                blockLength_ -
                i *
                ubFactor_;


            int64_t currentNum =
                remain >
                ubFactor_
                    ? ubFactor_
                    : remain;


            CopyIn(
                i,
                currentNum);


            Compute(
                currentNum);


            CopyOut(
                i,
                currentNum);
        }
    }


private:

    // =====================================================
    // CopyIn
    //
    // 使用DataCopyPad解决：
    // N不是32Byte整数倍的问题
    // =====================================================

    __aicore__ inline void CopyIn(
        int64_t progress,
        int64_t currentNum)
    {
        LocalTensor<T> inputLocal =
            inputQueue_
                .AllocTensor<T>();


        DataCopyExtParams copyParams;

        copyParams.blockCount = 1;

        // 注意：
        // DataCopyPad的blockLen单位是Byte
        copyParams.blockLen =
            static_cast<uint32_t>(
                currentNum *
                sizeof(T));

        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        copyParams.rsv = 0;


        DataCopyPadExtParams<T>
            padParams;

        padParams.isPad = false;
        padParams.leftPadding = 0;
        padParams.rightPadding = 0;
        padParams.paddingValue =
            static_cast<T>(0);


        DataCopyPad(
            inputLocal,
            inputGm_[
                progress *
                ubFactor_],
            copyParams,
            padParams);


        inputQueue_
            .EnQue(
                inputLocal);
    }


    // =====================================================
    // Compute
    //
    // Square(x) = x * x
    // =====================================================

    __aicore__ inline void Compute(
        int64_t currentNum)
    {
        LocalTensor<T> inputLocal =
            inputQueue_
                .DeQue<T>();


        LocalTensor<T> outputLocal =
            outputQueue_
                .AllocTensor<T>();


        // 每一个元素：
        //
        // output[i]
        // =
        // input[i] * input[i]
        //
        AscendC::Mul(
            outputLocal,
            inputLocal,
            inputLocal,
            static_cast<uint32_t>(
                currentNum));


        outputQueue_
            .EnQue(
                outputLocal);


        inputQueue_
            .FreeTensor(
                inputLocal);
    }


    // =====================================================
    // CopyOut
    //
    // 同样使用DataCopyPad，
    // 只写回currentNum个真实有效元素。
    // =====================================================

    __aicore__ inline void CopyOut(
        int64_t progress,
        int64_t currentNum)
    {
        LocalTensor<T> outputLocal =
            outputQueue_
                .DeQue<T>();


        DataCopyExtParams copyParams;

        copyParams.blockCount = 1;

        copyParams.blockLen =
            static_cast<uint32_t>(
                currentNum *
                sizeof(T));

        copyParams.srcStride = 0;
        copyParams.dstStride = 0;
        copyParams.rsv = 0;


        DataCopyPad(
            outputGm_[
                progress *
                ubFactor_],
            outputLocal,
            copyParams);


        outputQueue_
            .FreeTensor(
                outputLocal);
    }


private:

    TPipe pipe_;


    TQue<
        QuePosition::VECIN,
        BUFFER_NUM>
        inputQueue_;


    TQue<
        QuePosition::VECOUT,
        BUFFER_NUM>
        outputQueue_;


    GlobalTensor<T>
        inputGm_;


    GlobalTensor<T>
        outputGm_;


    // 当前Core实际处理的数据长度
    int64_t blockLength_ = 0;


    // 一个Tile最大元素数量
    int64_t ubFactor_ = 0;
};


} // namespace NsSquare

#endif // SQUARE_H
