/*!
 * \file square.h
 * \brief Square kernel implementation
 */

#ifndef SQUARE_H
#define SQUARE_H

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"

#include "square_tiling_data.h"
#include "square_tiling_key.h"


namespace NsSquare {

using namespace AscendC;


/*
 * ============================================================
 * 竞速调整：
 *
 * Single Buffer
 *
 * 对于当前这种微秒级 ElementWise 算子，
 * 测试数据多数可以做到：
 *
 * 1 Core
 *   ↓
 * 1 Tile
 *   ↓
 * DataCopy
 *   ↓
 * Mul
 *   ↓
 * DataCopy
 *
 * 因此不再为 Double Buffer 支付额外管理成本。
 * ============================================================
 */
constexpr int32_t BUFFER_NUM = 1;

constexpr int64_t ALIGN_BYTES = 32;


template <typename T>
class Square {
public:

    __aicore__ inline Square() {}


    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR y,
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

    AscendC::TPipe pipe;


    /*
     * 输入 Single Buffer
     */
    AscendC::TQue<
        AscendC::QuePosition::VECIN,
        BUFFER_NUM>
        inputQueueX;


    /*
     * 输出 Single Buffer
     */
    AscendC::TQue<
        AscendC::QuePosition::VECOUT,
        BUFFER_NUM>
        outputQueueY;


    AscendC::GlobalTensor<T>
        inputGMX;

    AscendC::GlobalTensor<T>
        outputGMY;


    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
};


// ============================================================
// Init
// ============================================================

template <typename T>
__aicore__ inline void Square<T>::Init(
    GM_ADDR x,
    GM_ADDR y,
    const SquareTilingData* tilingData)
{
    const int64_t blockIdx =
        static_cast<int64_t>(
            AscendC::GetBlockIdx());


    /*
     * 当前 Core 从哪个元素开始。
     */
    const int64_t gmOffset =
        tilingData->blockFactor *
        blockIdx;


    /*
     * 当前 Core 实际还有多少有效数据。
     */
    int64_t remainLength =
        tilingData->totalNum -
        gmOffset;


    /*
     * 正常 Core：
     *
     * blockLength = blockFactor
     *
     * 最后一个 Core：
     *
     * blockLength = 剩余元素
     */
    blockLength_ =
        (remainLength >
         tilingData->blockFactor)
            ? tilingData->blockFactor
            : remainLength;


    ubLength_ =
        tilingData->ubFactor;


    /*
     * 当前 Core 对应的 GM。
     */
    inputGMX.SetGlobalBuffer(
        (__gm__ T*)x +
            gmOffset,
        blockLength_);


    outputGMY.SetGlobalBuffer(
        (__gm__ T*)y +
            gmOffset,
        blockLength_);


    /*
     * Single Buffer：
     *
     * input 1份
     * output 1份
     */
    pipe.InitBuffer(
        inputQueueX,
        BUFFER_NUM,
        ubLength_ *
            sizeof(T));


    pipe.InitBuffer(
        outputQueueY,
        BUFFER_NUM,
        ubLength_ *
            sizeof(T));
}


// ============================================================
// CopyIn
// ============================================================

template <typename T>
__aicore__ inline void Square<T>::CopyIn(
    int64_t progress,
    int64_t currentNum)
{
    AscendC::LocalTensor<T>
        xLocal =
            inputQueueX
                .AllocTensor<T>();


    const int64_t currentBytes =
        currentNum *
        sizeof(T);


    const int64_t gmOffset =
        progress *
        ubLength_;


    /*
     * ========================================================
     * Fast Path
     *
     * 当前数据量是 32B 整数倍：
     * 使用普通 DataCopy。
     * ========================================================
     */
    if ((currentBytes &
         (ALIGN_BYTES - 1)) == 0) {

        AscendC::DataCopy(
            xLocal,
            inputGMX[
                gmOffset],
            static_cast<uint32_t>(
                currentNum));

    } else {

        /*
         * ====================================================
         * Tail Path
         *
         * 非 32B 对齐的数据只会出现在尾块。
         *
         * 使用 DataCopyPad，
         * 避免 GM 越界读取。
         * ====================================================
         */
        AscendC::DataCopyParams
            copyParams;


        copyParams.blockCount =
            1;


        copyParams.blockLen =
            static_cast<uint16_t>(
                currentBytes);


        copyParams.srcStride =
            0;


        copyParams.dstStride =
            0;


        AscendC::DataCopyPad(
            xLocal,
            inputGMX[
                gmOffset],
            copyParams,
            {false, 0, 0, 0});
    }


    /*
     * EnQue 建立：
     *
     * MTE2 -> Vector
     *
     * 的同步关系。
     */
    inputQueueX.EnQue(
        xLocal);
}


// ============================================================
// Compute
// ============================================================

template <typename T>
__aicore__ inline void Square<T>::Compute(
    int64_t currentNum)
{
    AscendC::LocalTensor<T>
        xLocal =
            inputQueueX
                .DeQue<T>();


    AscendC::LocalTensor<T>
        yLocal =
            outputQueueY
                .AllocTensor<T>();


    /*
     * ========================================================
     * Square:
     *
     * y = x * x
     *
     * 这里直接传真实 currentNum。
     *
     * 不再为了 32B 对齐人为扩大计算量。
     * ========================================================
     */
    AscendC::Mul(
        yLocal,
        xLocal,
        xLocal,
        static_cast<int32_t>(
            currentNum));


    /*
     * EnQue 建立：
     *
     * Vector -> MTE3
     *
     * 的同步关系。
     */
    outputQueueY.EnQue(
        yLocal);


    inputQueueX.FreeTensor(
        xLocal);
}


// ============================================================
// CopyOut
// ============================================================

template <typename T>
__aicore__ inline void Square<T>::CopyOut(
    int64_t progress,
    int64_t currentNum)
{
    AscendC::LocalTensor<T>
        yLocal =
            outputQueueY
                .DeQue<T>();


    const int64_t currentBytes =
        currentNum *
        sizeof(T);


    const int64_t gmOffset =
        progress *
        ubLength_;


    /*
     * ========================================================
     * Fast Path
     * ========================================================
     */
    if ((currentBytes &
         (ALIGN_BYTES - 1)) == 0) {

        AscendC::DataCopy(
            outputGMY[
                gmOffset],
            yLocal,
            static_cast<uint32_t>(
                currentNum));

    } else {

        /*
         * ====================================================
         * 非对齐 Tail。
         *
         * 只写实际有效 Byte，
         * 防止写坏相邻 GM 数据。
         * ====================================================
         */
        AscendC::DataCopyParams
            copyParams;


        copyParams.blockCount =
            1;


        copyParams.blockLen =
            static_cast<uint16_t>(
                currentBytes);


        copyParams.srcStride =
            0;


        copyParams.dstStride =
            0;


        AscendC::DataCopyPad(
            outputGMY[
                gmOffset],
            yLocal,
            copyParams);
    }


    outputQueueY.FreeTensor(
        yLocal);
}


// ============================================================
// Process
// ============================================================

template <typename T>
__aicore__ inline void Square<T>::Process()
{
    /*
     * ========================================================
     * Fast Path
     *
     * 当前 Core 的数据一次能够放入 UB。
     *
     * 直接：
     *
     * CopyIn
     *   ↓
     * Mul
     *   ↓
     * CopyOut
     *
     * 完全不进入循环。
     * ========================================================
     */
    if (blockLength_ <=
        ubLength_) {

        CopyIn(
            0,
            blockLength_);


        Compute(
            blockLength_);


        CopyOut(
            0,
            blockLength_);


        return;
    }


    /*
     * ========================================================
     * 大 Tensor Path
     * ========================================================
     */

    const int64_t fullLoopCount =
        blockLength_ /
        ubLength_;


    const int64_t tailNum =
        blockLength_ -
        fullLoopCount *
        ubLength_;


    /*
     * 完整 Tile。
     *
     * 不在循环内部判断 Tail。
     */
    for (int64_t i = 0;
         i < fullLoopCount;
         ++i) {

        CopyIn(
            i,
            ubLength_);


        Compute(
            ubLength_);


        CopyOut(
            i,
            ubLength_);
    }


    /*
     * 最后一个 Tail。
     */
    if (tailNum > 0) {

        CopyIn(
            fullLoopCount,
            tailNum);


        Compute(
            tailNum);


        CopyOut(
            fullLoopCount,
            tailNum);
    }
}

} // namespace NsSquare

#endif // SQUARE_H