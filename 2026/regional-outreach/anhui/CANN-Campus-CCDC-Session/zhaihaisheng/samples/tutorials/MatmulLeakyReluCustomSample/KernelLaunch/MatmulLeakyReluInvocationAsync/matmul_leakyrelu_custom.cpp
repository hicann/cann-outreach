/**
 * @file matmul_leakyrelu_custom.cpp
 *
 * Copyright (C) 2023-2024. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#include "kernel_operator.h"
#include "lib/matmul_intf.h"

using namespace matmul;

__aicore__ inline uint32_t Ceiling(uint32_t a, uint32_t b)
{
    return (a + b - 1) / b;
}

/**
  * @brief  Copy tiling data to TCubeTiling ptr from tiling gm addr.
  * @param  tiling: TCubeTiling ptr which needs to copy tiling data.
  * @param  tilingGM: tiling gm addr.
  * @retval None
  */
__aicore__ inline void CopyTiling(TCubeTiling *tiling, GM_ADDR tilingGM)
{
    uint32_t *ptr = reinterpret_cast<uint32_t *>(tiling);
    auto tiling32 = reinterpret_cast<__gm__ uint32_t *>(tilingGM);

    for (uint32_t i = 0; i < sizeof(TCubeTiling) / sizeof(uint32_t); i++, ptr++) {
        *ptr = *(tiling32 + i);
    }
    return;
}

template <typename aType, typename bType, typename cType, typename biasType> class MatmulLeakyKernel {
public:
    __aicore__ inline MatmulLeakyKernel(){};
    __aicore__ inline void Init(GM_ADDR a, GM_ADDR b, GM_ADDR bias, GM_ADDR c, GM_ADDR workspace,
                                const TCubeTiling &tiling, AscendC::TPipe *pipe);
    __aicore__ inline void Process();

    __aicore__ inline void MatmulCompute();
    __aicore__ inline void LeakyReluCompute(uint32_t count);
    __aicore__ inline void CopyOut(uint32_t count);
    __aicore__ inline void CalcOffset(int32_t blockIdx, const TCubeTiling &tiling, int32_t &offsetA, int32_t &offsetB,
                                      int32_t &offsetC, int32_t &offsetBias);

    Matmul<MatmulType<AscendC::TPosition::GM, CubeFormat::ND, aType>, MatmulType<AscendC::TPosition::GM, CubeFormat::ND, bType>,
           MatmulType<AscendC::TPosition::VECIN, CubeFormat::ND, cType>, MatmulType<AscendC::TPosition::GM, CubeFormat::ND, biasType>>
        matmulObj;

    AscendC::GlobalTensor<aType> aGlobal;
    AscendC::GlobalTensor<bType> bGlobal;
    AscendC::GlobalTensor<cType> cGlobal;
    AscendC::GlobalTensor<biasType> biasGlobal;
    AscendC::GlobalTensor<cType> workspaceGlobal;
    AscendC::LocalTensor<cType> reluInLocal;
    TCubeTiling tiling;
    AscendC::TQue<AscendC::QuePosition::VECIN, 1> reluInQueue;
    AscendC::TQue<AscendC::QuePosition::VECOUT, 1> reluOutQueue;
    uint32_t splitRowNums = 0;
    uint32_t splitRowSize = 0;
};

/**
  * @brief  Set matmulLeaky input and output gm addr of current core.
  * @param  a: A matrix gm addr.
  * @param  b: B matrix gm addr.
  * @param  bias: Bias gm addr.
  * @param  c: C matrix gm addr.
  * @param  workspace: Temporary gm space addr required by matmul calc.
  * @param  tiling: matmul tiling data.
  * @param  pipe: Global memory and sync management TPipe object.
  * @retval None
  */
template <typename aType, typename bType, typename cType, typename biasType>
__aicore__ inline void MatmulLeakyKernel<aType, bType, cType, biasType>::Init(GM_ADDR a, GM_ADDR b, GM_ADDR bias,
                                                                              GM_ADDR c, GM_ADDR workspace,
                                                                              const TCubeTiling &tiling, AscendC::TPipe *pipe)
{
    this->tiling = tiling;
    splitRowNums = 4;
    splitRowSize = tiling.baseM / splitRowNums;
    aGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ aType *>(a), tiling.M * tiling.Ka);
    bGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ bType *>(b), tiling.Kb * tiling.N);
    cGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ cType *>(c), tiling.M * tiling.N);
    biasGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ biasType *>(bias), tiling.N);
    workspaceGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ cType *>(workspace), tiling.M * tiling.N);

    int32_t offsetA, offsetB, offsetC, offsetBias;
    CalcOffset(AscendC::GetBlockIdx(), tiling, offsetA, offsetB, offsetC, offsetBias); // Calculate the gm offset based on the blockidx.
    aGlobal = aGlobal[offsetA];
    bGlobal = bGlobal[offsetB];
    cGlobal = cGlobal[offsetC];
    biasGlobal = biasGlobal[offsetBias];
    workspaceGlobal = workspaceGlobal[GetBlockIdx() * tiling.singleCoreM * tiling.singleCoreN];
    pipe->InitBuffer(reluInQueue, 1, tiling.baseM * tiling.baseN * sizeof(cType)); // Init relu input queue.
    pipe->InitBuffer(reluOutQueue, 1, splitRowSize * tiling.baseN * sizeof(cType)); // Init relu output queue.
}

/**
  * @brief  Main process of matmul calculation
  * @retval None
  */
template <typename aType, typename bType, typename cType, typename biasType>
__aicore__ inline void MatmulLeakyKernel<aType, bType, cType, biasType>::Process()
{
    matmulObj.SetWorkspace(workspaceGlobal);
    matmulObj.SetTensorA(aGlobal);
    matmulObj.SetTensorB(bGlobal);
    matmulObj.SetBias(biasGlobal);
    matmulObj.template Iterate<false>(); // Sync is set false means async, this scene will run while(Iterate).
    for (int i = 0; i < tiling.singleCoreM * tiling.singleCoreN / (tiling.baseM * tiling.baseN); ++i) {
        MatmulCompute(); // Get matmul compute result.
        reluInLocal = reluInQueue.DeQue<cType>(); // wait matmul compute result finish.
        for (int j = 0; j < splitRowNums; ++j) {
            LeakyReluCompute(j); // Compute leakyRelu.
            CopyOut(i * splitRowNums + j); // Copy leakyRelu out result to GM.
        }
        reluInQueue.FreeTensor(reluInLocal);
    }
    matmulObj.End();
}

template <typename aType, typename bType, typename cType, typename biasType>
__aicore__ inline void MatmulLeakyKernel<aType, bType, cType, biasType>::MatmulCompute()
{
    reluInLocal = reluInQueue.AllocTensor<cType>();
    matmulObj.template GetTensorC<false>(reluInLocal, false, true);
    reluInQueue.EnQue(reluInLocal);
}

template <typename aType, typename bType, typename cType, typename biasType>
__aicore__ inline void MatmulLeakyKernel<aType, bType, cType, biasType>::LeakyReluCompute(uint32_t count)
{
    auto reluOutLocal = reluOutQueue.AllocTensor<cType>();
    LeakyRelu(reluOutLocal, reluInLocal[count * splitRowSize * tiling.baseN], (cType)0.001, splitRowSize * tiling.baseN);
    reluOutQueue.EnQue(reluOutLocal);
}

/**
  * @brief  Copy leakyRelu out result to GM.
  * @param  count: Iterate count.
  * @retval None
  */
template <typename aType, typename bType, typename cType, typename biasType>
__aicore__ inline void MatmulLeakyKernel<aType, bType, cType, biasType>::CopyOut(uint32_t count)
{
    auto reluOutLocal = reluOutQueue.DeQue<cType>(); // wait relu compute result finish.
    const uint32_t roundM = tiling.singleCoreM / splitRowSize;
    const uint32_t roundN = tiling.singleCoreN / tiling.baseN;
    uint32_t startOffset = (count % roundM * splitRowSize * tiling.N + count / roundM * tiling.baseN);
    AscendC::DataCopyParams copyParam = {(uint16_t)splitRowSize, (uint16_t)(tiling.baseN * sizeof(cType) / AscendC::DEFAULT_C0_SIZE), 0,
                                (uint16_t)((tiling.N - tiling.baseN) * sizeof(cType) / AscendC::DEFAULT_C0_SIZE)};
    DataCopy(cGlobal[startOffset], reluOutLocal, copyParam);
    reluOutQueue.FreeTensor(reluOutLocal);
}

/**
  * @brief  Calculate the gm offset based on the blockidx.
  * @param  blockIdx: Current Core blockidx.
  * @param  tiling: Matmul tiling data.
  * @param  offsetA: Gm offset of A matrix.
  * @param  offsetB: Gm offset of B matrix.
  * @param  offsetC: Gm offset of C matrix.
  * @param  offsetBias: Gm offset of Bias matrix.
  * @retval None
  */
template <typename aType, typename bType, typename cType, typename biasType>
__aicore__ inline void
MatmulLeakyKernel<aType, bType, cType, biasType>::CalcOffset(int32_t blockIdx, const TCubeTiling &tiling,
                                                             int32_t &offsetA, int32_t &offsetB, int32_t &offsetC,
                                                             int32_t &offsetBias)
{
    auto mSingleBlocks = Ceiling(tiling.M, tiling.singleCoreM);
    auto mCoreIndx = blockIdx % mSingleBlocks;
    auto nCoreIndx = blockIdx / mSingleBlocks;

    offsetA = mCoreIndx * tiling.Ka * tiling.singleCoreM;
    offsetB = nCoreIndx * tiling.singleCoreN;
    offsetC = mCoreIndx * tiling.N * tiling.singleCoreM + nCoreIndx * tiling.singleCoreN;
    offsetBias = nCoreIndx * tiling.singleCoreN;
}

/**
  * @brief  matmul_leakyrelu kernel function entry
  * @param  a: A matrix gm addr.
  * @param  b: B matrix gm addr.
  * @param  bias: Bias gm addr.
  * @param  c: Out gm addr.
  * @param  workspace: Temporary gm space addr required by matmul calc.
  * @param  tilingGm: Tiling data addr. 
  * @retval None
  */
extern "C" __global__ __aicore__ void matmul_leakyrelu_custom(GM_ADDR a, GM_ADDR b, GM_ADDR bias, GM_ADDR c,
                                                              GM_ADDR workspace, GM_ADDR tilingGm)
{
    AscendC::TPipe pipe;
    TCubeTiling tiling;
    CopyTiling(&tiling, tilingGm);

    MatmulLeakyKernel<half, half, float, float> matmulLeakyKernel;
    matmulLeakyKernel.Init(a, b, bias, c, workspace, tiling, &pipe);
    REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), matmulLeakyKernel.matmulObj, &matmulLeakyKernel.tiling);
    matmulLeakyKernel.Process();
}