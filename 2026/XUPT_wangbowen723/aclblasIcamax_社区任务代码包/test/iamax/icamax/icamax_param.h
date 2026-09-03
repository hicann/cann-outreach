/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#pragma once
#include <string>
#include <vector>
#include "acl/acl.h"
#include "cann_ops_blas.h"
#include "csv_loader.h"
#include "fill.h"

// 1-D complex64 vector fill helper.
// The real part is generated with `seed`, the imaginary part with `seed + 1000`
// (mirrors makeBlasComplexMatrix): both parts share the same value range from
// the fill mode but use independent random streams to avoid correlation.
inline std::vector<aclblasComplex> makeBlasComplexArray(int64_t count, const BlasFillMode& fill, uint32_t seed)
{
    if (fill.method == BlasFillMode::M_NULLPTR || count <= 0) {
        return {};
    }
    std::vector<float> realPart = makeBlasArray(count, fill, seed);
    std::vector<float> imagPart = makeBlasArray(count, fill, seed + 1000U);
    std::vector<aclblasComplex> data(static_cast<size_t>(count));
    for (size_t i = 0; i < data.size(); i++) {
        data[i] = aclblasComplex{realPart[i], imagPart[i]};
    }
    return data;
}

inline std::vector<aclblasComplex> makeBlasComplexArray(int64_t count, const std::string& fillStr, uint32_t seed)
{
    return makeBlasComplexArray(count, BlasFillMode(fillStr), seed);
}

struct IcamaxParam : public BlasTestParamBase {
    int n = 0;
    int incx = 1;
    BlasFillMode x = parseFill("RANDOM");
    // result 列标记 result 空指针负向用例（"NULLPTR"）
    bool resultNullptr = false;
    IcamaxParam(const csv_map& map) : BlasTestParamBase(map)
    {
        n = parseInt(ReadMap(map, "n", "0"));
        incx = parseInt(ReadMap(map, "incx", "1"));
        x = parseFill(ReadMap(map, "x", "RANDOM"));
        resultNullptr = (ReadMap(map, "result", "") == "NULLPTR");
    }
};
