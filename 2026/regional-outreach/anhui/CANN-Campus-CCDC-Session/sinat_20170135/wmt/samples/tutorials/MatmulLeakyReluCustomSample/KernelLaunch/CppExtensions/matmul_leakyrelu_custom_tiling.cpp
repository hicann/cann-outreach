/**
 * @file matmul_leakyrelu_custom_tiling.cpp
 *
 * Copyright (C) 2024. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#include <cassert>
#include <fstream>
#include <iostream>
#include <map>
#include <string>

#include "tiling/tiling_api.h"
using namespace matmul_tiling;
using namespace std;

uint8_t *GetTilingBuf(optiling::TCubeTiling *tilingData)
{
    uint32_t tilingSize = tilingData->GetDataSize();
    uint8_t *buf = (uint8_t *)malloc(tilingSize);
    tilingData->SaveToBuffer(buf, tilingSize);
    return buf;
}

uint8_t *GenerateTiling()
{
    int M = 1024;
    int N = 640;
    int K = 256;
    int baseM = 256;
    int baseN = 128;
    TPosition leftPos = TPosition::GM;
    CubeFormat leftFormat = CubeFormat::ND;
    DataType leftDtype = DataType::DT_FLOAT16;
    int transposeA = 0;

    TPosition rightPos = TPosition::GM;
    CubeFormat rightFormat = CubeFormat::ND;
    DataType rightDtype = DataType::DT_FLOAT16;
    int transposeB = 0;

    TPosition resPos = TPosition::GM;
    CubeFormat resFormat = CubeFormat::ND;
    DataType resDtype = DataType::DT_FLOAT;

    TPosition biasPos = TPosition::GM;
    CubeFormat biasFormat = CubeFormat::ND;
    DataType biasDtype = DataType::DT_FLOAT;
    int isBias = 1;
    int usedCoreNum = 2;
    optiling::TCubeTiling tilingData;
    tilingData.set_usedCoreNum(usedCoreNum);
    MultiCoreMatmulTiling tilingApi;
    tilingApi.SetDim(usedCoreNum);
    tilingApi.SetAType(leftPos, leftFormat, leftDtype, bool(transposeA));
    tilingApi.SetBType(rightPos, rightFormat, rightDtype, bool(transposeB));
    tilingApi.SetCType(resPos, resFormat, resDtype);
    tilingApi.SetBiasType(biasPos, biasFormat, biasDtype);

    tilingApi.SetOrgShape(M, N, K);
    tilingApi.SetShape(M, N, K);
    tilingApi.SetBias(bool(isBias));
    tilingApi.SetTraverse(MatrixTraverse::FIRSTM);
    tilingApi.SetFixSplit(baseM, baseN, -1);
    tilingApi.SetBufferSpace(-1, -1, -1);
    int64_t res = tilingApi.GetTiling(tilingData);
    tilingData.set_stepM(1);
    tilingData.set_stepN(1);
    if (res == -1) {
        std::cout << "gen tiling failed" << std::endl;
    }
    return GetTilingBuf(&tilingData);
}
