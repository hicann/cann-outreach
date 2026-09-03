/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <cmath>
#include <cstdlib>
#include <vector>
#include "verify.h"
#include "blas_test.h"
#include "csv_loader.h"
#include "fill.h"
#include "icamax_param.h"
#include "icamax_golden.h"
#include "icamax_npu_wrapper.h"
class IcamaxArch35Test : public BlasTest<IcamaxParam> {};
TEST_F(IcamaxArch35Test, NullHandle)
{
    int result = 0;
    aclblasStatus_t ret = aclblasIcamax_npu(nullptr, 5, nullptr, 1, &result);
    EXPECT_EQ(ret, ACLBLAS_STATUS_NOT_INITIALIZED);
}
// 直接验证 host API：handle 为 nullptr 返回 ACLBLAS_STATUS_HANDLE_IS_NULLPTR
TEST_F(IcamaxArch35Test, NullHandleDirect)
{
    int result = 0;
    aclblasStatus_t ret = aclblasIcamax(nullptr, 5, nullptr, 1, &result);
    EXPECT_EQ(ret, ACLBLAS_STATUS_HANDLE_IS_NULLPTR);
}
INSTANTIATE_TEST_SUITE_P(
    Icamax, IcamaxArch35Test, ::testing::ValuesIn(GetCasesFromCsv<IcamaxParam>(ReplaceFileExtension2Csv(__FILE__))),
    PrintCaseInfoString<IcamaxParam>);
TEST_P(IcamaxArch35Test, CsvDriven)
{
    const auto& p = GetParam();
    std::cout << "\n=== RUN  [" << p.caseName << "] n=" << p.n << " incx=" << p.incx
              << " x_method=" << static_cast<int>(p.x.method) << " desc=" << p.description << " ===" << std::endl;
    const int absIncx = std::abs(p.incx);
    const int64_t xLen =
        (p.n > 0 && absIncx > 0) ? static_cast<int64_t>(1) + static_cast<int64_t>(p.n - 1) * absIncx : 0;
    std::vector<aclblasComplex> xHost = makeBlasComplexArray(xLen, p.x, p.randomSeed);
    const aclblasComplex* xPtr = xHost.empty() ? nullptr : xHost.data();
    int result = 0;
    aclblasStatus_t ret;
    if (p.resultNullptr) {
        ret = aclblasIcamax_npu_null_result(IcamaxArch35Test::handle_, p.n, xPtr, p.incx);
    } else {
        ret = aclblasIcamax_npu(IcamaxArch35Test::handle_, p.n, xPtr, p.incx, &result);
    }
    EXPECT_EQ(static_cast<int>(ret), static_cast<int>(p.expectResult))
        << "[" << p.caseName << "] unexpected return code: got " << static_cast<int>(ret) << " expect "
        << static_cast<int>(p.expectResult);
    if (p.expectResult != ACLBLAS_STATUS_SUCCESS)
        return;
    int golden = 0;
    aclblasIcamax_cpu(IcamaxArch35Test::handle_, p.n, xHost.data(), p.incx, &golden);
    EXPECT_EQ(result, golden) << "[" << p.caseName << "] index mismatch: NPU=" << result << " golden=" << golden;
}
