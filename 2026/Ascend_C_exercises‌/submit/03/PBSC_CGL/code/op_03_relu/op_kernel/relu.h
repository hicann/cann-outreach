/*!
 * \file relu.h
 * \brief Relu 算子 kernel 类定义
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
 * 题目固定Shape:
 *
 * (8, 2048)
 *
 * 8个Core，每Core固定2048元素。
 */
constexpr uint32_t BLOCK_LENGTH = 2048;

/*
 * 单Buffer。
 *
 * 这个小算子不做多Tile，
 * Double Buffer反而增加Queue和循环开销。
 */
constexpr int32_t BUFFER_NUM = 1;


template <typename DT_X>
class Relu {
public:

    __aicore__ inline Relu()
    {
    }


    __aicore__ inline void Init(
        GM_ADDR x,
        GM_ADDR y,
        const ReluTilingData* tilingData)
    {
        /*
         * 当前比赛shape固定，因此Kernel直接使用
         * 编译期BLOCK_LENGTH。
         *
         * 避免Kernel运行阶段执行：
         *
         * totalLength / GetBlockNum()
         * blockLength / tileNum
         *
         * 等Scalar计算。
         */
        (void)tilingData;

        const uint32_t offset =
            AscendC::GetBlockIdx() *
            BLOCK_LENGTH;

        /*
         * 当前Core独占连续2048元素。
         */
        xGm.SetGlobalBuffer(
            (__gm__ DT_X*)x + offset,
            BLOCK_LENGTH);

        yGm.SetGlobalBuffer(
            (__gm__ DT_X*)y + offset,
            BLOCK_LENGTH);

        /*
         * 一个输入Buffer
         * 一个输出Buffer
         *
         * FP32:
         * 2048 * 4 * 2 = 16KB
         *
         * FP16:
         * 2048 * 2 * 2 = 8KB
         */
        pipe.InitBuffer(
            inQueueX,
            BUFFER_NUM,
            BLOCK_LENGTH * sizeof(DT_X));

        pipe.InitBuffer(
            outQueueY,
            BUFFER_NUM,
            BLOCK_LENGTH * sizeof(DT_X));
    }


    __aicore__ inline void Process()
    {
        /*
         * 原模板：
         *
         * loopCount = tileNum * BUFFER_NUM
         * for (...)
         *
         * 由于本题固定tileNum=1，
         * 直接展开。
         *
         * 去掉Kernel Scalar循环控制。
         */
        CopyIn(0);
        Compute();
        CopyOut(0);
    }


private:

    __aicore__ inline void CopyIn(
        int32_t progress)
    {
        LocalTensor<DT_X> xLocal =
            inQueueX.AllocTensor<DT_X>();

        /*
         * progress恒为0。
         * 内联后编译器可直接消除地址计算。
         */
        AscendC::DataCopy(
            xLocal,
            xGm[
                progress *
                BLOCK_LENGTH],
            BLOCK_LENGTH);

        /*
         * EnQue同时负责建立
         * MTE2 -> Vector依赖关系。
         */
        inQueueX.EnQue(xLocal);
    }


    __aicore__ inline void Compute()
    {
        LocalTensor<DT_X> xLocal =
            inQueueX.DeQue<DT_X>();

        LocalTensor<DT_X> yLocal =
            outQueueY.AllocTensor<DT_X>();

        /*
         * ========================================
         * 核心Vector指令
         *
         * y = max(0, x)
         * ========================================
         *
         * 一次处理整个2048元素。
         */
        AscendC::Relu(
            yLocal,
            xLocal,
            BLOCK_LENGTH);

        /*
         * 建立Vector -> MTE3依赖。
         */
        outQueueY.EnQue(yLocal);

        inQueueX.FreeTensor(xLocal);
    }


    __aicore__ inline void CopyOut(
        int32_t progress)
    {
        LocalTensor<DT_X> yLocal =
            outQueueY.DeQue<DT_X>();

        /*
         * 一次写回整个2048元素。
         */
        AscendC::DataCopy(
            yGm[
                progress *
                BLOCK_LENGTH],
            yLocal,
            BLOCK_LENGTH);

        outQueueY.FreeTensor(yLocal);
    }


private:

    AscendC::TPipe pipe;

    AscendC::TQue<
        AscendC::QuePosition::VECIN,
        BUFFER_NUM>
        inQueueX;

    AscendC::TQue<
        AscendC::QuePosition::VECOUT,
        BUFFER_NUM>
        outQueueY;

    AscendC::GlobalTensor<DT_X> xGm;
    AscendC::GlobalTensor<DT_X> yGm;
};

} // namespace NsRelu

#endif // RELU_H