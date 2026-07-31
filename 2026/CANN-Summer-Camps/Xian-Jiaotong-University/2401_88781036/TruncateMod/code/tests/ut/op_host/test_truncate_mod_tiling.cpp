/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>
#include <gtest/gtest.h>
#include "truncate_mod_tiling.h"
#include "../../../op_kernel/truncate_mod_tiling_data.h"
#include "../../../op_kernel/truncate_mod_tiling_key.h"
#include "tiling_context_faker.h"
#include "tiling_case_executor.h"

using namespace std;
using namespace optiling;

class TruncateModTiling : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "TruncateModTiling SetUp" << endl; }

    static void TearDownTestCase() { cout << "TruncateModTiling TearDown " << endl; }
};

TEST_F(TruncateModTiling, ascend910b_test_tiling_INT32_001)
{
    optiling::TruncateModCompileInfo compileInfo = {40, 196608, true};
    gert::TilingContextPara tilingContextPara("TruncateMod",
                                              {
                                                  {{{1024, 1024}, {1024, 1024}}, ge::DT_INT32, ge::FORMAT_ND},
                                                  {{{1024, 1024}, {1024, 1024}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              {
                                                  {{{1024, 1024}, {1024, 1024}}, ge::DT_INT32, ge::FORMAT_ND},
                                              },
                                              &compileInfo);
    uint64_t expectTilingKey = 0;
    string expectTilingData = "26216 26216 7 7 1640 1640 40 1048576 8192 3280 3280 1048576 1048576 "
                              "1 1 1 1 1 1 1024 1024 "
                              "1 1 1 1 1 1 1024 1024 "
                              "1 1 1 1 1 1 1024 1024 ";
    std::vector<size_t> expectWorkspaces = {0};
    ExecuteTestCase(tilingContextPara, ge::GRAPH_SUCCESS, expectTilingKey, expectTilingData, expectWorkspaces);
}
