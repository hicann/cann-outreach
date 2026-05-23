# 代码检视报告

## 检视概览
- 代码文件：/mnt/workspace/gitCode/cann/cann-outreach/2026/KADC/CANNBot/demo/code-review/tiling_common_demo.txt
- 代码侧别：Tiling侧
- 检视文档：cpp-secure.md, ascendc-topk.md
- 总条例数：10组（安全类条例优先检视）
- 检视时间：2026-05-22

## 检视统计

| 状态 | 条例数 | 占比 |
|-----|--------|------|
| PASS | ~20 | 67% |
| FAIL（发现问题） | 15 | 33% |
| SUSPICIOUS（需关注） | 1 | - |

## 发现问题（HIGH 置信度）

### [SEC-4.1 + TOPK-1 + TOPK-7] 空指针解引用风险

**问题描述**：多处指针返回值未判空直接使用，存在空指针解引用风险

**代码片段**（行 37-38）：
```cpp
auto attrs = context->GetAttrs();
const char *mode = attrs->GetAttrPointer<char>(CACHE_MODE_INDEX);

if (strcmp(mode, "PA_NZ") == 0) {
```

**修复建议**：
```cpp
auto attrs = context->GetAttrs();
if (attrs == nullptr) {
    OP_LOGE(context, "GetAttrs() returned nullptr");
    return false;
}
const char *mode = attrs->GetAttrPointer<char>(CACHE_MODE_INDEX);
if (mode == nullptr) {
    OP_LOGE(context, "mode is null");
    return false;
}
```

---

### [SEC-3.5 + TOPK-1] GetAttrPointer返回值未判空

**问题描述**：isSeqLensCumsum 指针未判空直接解引用

**代码片段**（行 76-84）：
```cpp
auto isSeqLensCumsum = attrs->GetAttrPointer<bool>(IS_SEQ_LENS_CUNSUM_INDEX);
tilingData.set_blockSize(blockSize);
tilingData.set_numTokens(numTokens);
tilingData.set_numblkTabCol(numblkTabCol);
tilingData.set_tokenSizeK(tokenSizeK);
tilingData.set_tokenSizeV(tokenSizeV);
tilingData.set_typeByte(typeByte);
tilingData.set_hasSeqStarts(hasSeqStarts);
tilingData.set_isSeqLensCumsum(*isSeqLensCumsum);
```

**修复建议**：
```cpp
auto isSeqLensCumsum = attrs->GetAttrPointer<bool>(IS_SEQ_LENS_CUNSUM_INDEX);
if (isSeqLensCumsum == nullptr) {
    OP_LOGE(context, "isSeqLensCumsum attribute is null");
    return false;
}
tilingData.set_isSeqLensCumsum(*isSeqLensCumsum);
```

---

### [SEC-2.1 + TOPK-8] 有符号整数溢出风险

**问题描述**：tokenSizeK/V 乘法运算可能溢出 int32_t 范围，违反"gm内存偏移必须用int64表示"

**代码片段**（行 53-56）：
```cpp
tokenSizeK = static_cast<int32_t>(
    kShape->GetStorageShape().GetDim(DIM_1) * kShape->GetStorageShape().GetDim(DIM_2));
tokenSizeV = static_cast<int32_t>(
    vShape->GetStorageShape().GetDim(DIM_1) * vShape->GetStorageShape().GetDim(DIM_2));
```

**风险分析**：GetDim 返回 int64_t，两值相乘可能超过 INT_MAX (2147483647)，强制转换为 int32_t 会截断。

**工具验证**：dim1=65536, dim2=32768 →乘积=2147483648 →截断为 -2147483648

**修复建议**：
```cpp
int64_t dim1 = kShape->GetStorageShape().GetDim(DIM_1);
int64_t dim2 = kShape->GetStorageShape().GetDim(DIM_2);
if (dim1 > 0 && dim2 > INT_MAX / dim1) {
    OP_LOGE(context, "tokenSizeK multiplication overflow");
    return false;
}
tokenSizeK = static_cast<int32_t>(dim1 * dim2);
```

---

### [TOPK-6] 校验条件逻辑错误

**问题描述**：numTokens 校验条件 `(numTokens <= 0 && numTokens > INT_MAX)` 永远为 false，校验失效

**代码片段**（行 67-68）：
```cpp
OP_CHECK_IF((numTokens <= 0 && numTokens > INT_MAX),
            OP_LOGE(context, "numTokens is invalid."), return false);
```

**修复建议**：
```cpp
OP_CHECK_IF((numTokens <= 0 || numTokens > INT_MAX),
            OP_LOGE(context, "numTokens is invalid."), return false);
```

---

### [SEC-4.1 + TOPK-7] 多个参数缺少边界校验

**问题描述**：tokenSizeK、tokenSizeV、numblkTabCol、typeByte 缺少边界校验

**代码片段**（行 40-44, 62-63）：
```cpp
int32_t blockSize;
int32_t tokenSizeK;
int32_t tokenSizeV;
auto inDtype = context->GetInputDesc(DIM_0)->GetDataType();
uint32_t typeByte = static_cast<uint32_t>(GetTensorElementSizes(inDtype));

int32_t numTokens = static_cast<int32_t>(blockTablesShape->GetStorageShape().GetDim(DIM_0));
int32_t numblkTabCol = static_cast<int32_t>(blockTablesShape->GetStorageShape().GetDim(DIM_1));
```

**修复建议**：
```cpp
OP_CHECK_IF((tokenSizeK <= 0), OP_LOGE(context, "tokenSizeK is invalid"), return false);
OP_CHECK_IF((tokenSizeV <= 0), OP_LOGE(context, "tokenSizeV is invalid"), return false);
OP_CHECK_IF((numblkTabCol <= 0), OP_LOGE(context, "numblkTabCol is invalid"), return false);
OP_CHECK_IF((typeByte == 0), OP_LOGE(context, "typeByte is invalid"), return false);
```

---

## 需关注（MED 置信度）

### [SEC-4.2] GetRawTilingData指针未判空

**问题描述**：SaveToBuffer 前未判空GetRawTilingData()，但框架 API 可能提供保护

**代码片段**（行 89-90）：
```cpp
tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(),
                        context->GetRawTilingData()->GetCapacity());
```

**修复建议**：
```cpp
auto rawTilingData = context->GetRawTilingData();
if (rawTilingData == nullptr) {
    OP_LOGE(context, "rawTilingData is null");
    return false;
}
tilingData.SaveToBuffer(rawTilingData->GetData(), rawTilingData->GetCapacity());
```

---

## 通过条例

- SEC-1.1：类型安全（部分PASS，缩窄转换需关注）
- SEC-1.2：内存安全（部分FAIL）
- SEC-1.3：未定义行为（部分FAIL）
- SEC-2.2：无符号整数回绕（PASS）
- SEC-2.3：除零保护（PASS，无除法运算）
- SEC-3.1：未初始化变量（PASS，有分支保护）
- SEC-3.2：资源释放后置空（PASS，无资源释放）
- SEC-3.3：数组索引校验（PASS，无外部数据索引）
- SEC-3.4：sizeof指针（PASS）
- SEC-3.6：字符串存储空间（PASS）
- SEC-5.1-5.4：资源管理（PASS，无资源申请）
- SEC-8.1-8.3：安全函数（PASS，无使用）
- SEC-11.1-11.4：LOG API安全（PASS）
- TOPK-2：GetInputDesc获取Dtype（PASS）
- TOPK-3：生命周期管理（PASS）
- TOPK-4：属性获取来源（PASS）
- TOPK-5：属性类型一致性（PASS）
- TOPK-10：整数计算优先（PASS）
- TOPK-12：宏变量命名（PASS）
- TOPK-13：thread_local禁用（PASS）

---

## 被检视代码

```cpp
/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * Licensed under CANN Open Software License Agreement Version 2.0.
 */

/*!
 * \file gather_pa_kv_cache_tiling.cpp
 * \brief core tiling logic for gather_pa_kv_cache
 */

#include <climits>
#include "log/log.h"
#include "platform/platform_info_def.h"
#include "tiling/tiling_api.h"
#include "gather_pa_kv_cache_tiling.h"

using namespace Ops::Transformer::OpTiling;
namespace optiling {

constexpr uint32_t DIM_0 = 0;
constexpr uint32_t DIM_1 = 1;
constexpr uint32_t DIM_2 = 2;
constexpr uint32_t DIM_3 = 3;
constexpr uint32_t DIM_6 = 6;
constexpr uint32_t CACHE_MODE_INDEX = 0;
constexpr uint32_t IS_SEQ_LENS_CUNSUM_INDEX = 1;
constexpr uint64_t ASCENDC_TOOLS_WORKSPACE = static_cast<uint64_t>(8) * 1024 * 1024;
static constexpr uint32_t TILING_KEY_NZ = 577;
static constexpr uint32_t TILING_KEY_ND = 617;

bool CommonGatherPaKvCacheTiling(gert::TilingContext *context)
{
    auto kCacheShape = context->GetInputShape(DIM_0);
    auto blockTablesShape = context->GetInputShape(DIM_2);
    auto kShape = context->GetOutputShape(DIM_0);
    auto vShape = context->GetOutputShape(DIM_1);
    auto attrs = context->GetAttrs();
    const char *mode = attrs->GetAttrPointer<char>(CACHE_MODE_INDEX);

    int32_t blockSize;
    int32_t tokenSizeK;
    int32_t tokenSizeV;
    auto inDtype = context->GetInputDesc(DIM_0)->GetDataType();
    uint32_t typeByte = static_cast<uint32_t>(GetTensorElementSizes(inDtype));

    if (strcmp(mode, "PA_NZ") == 0) {
        blockSize = static_cast<int32_t>(kCacheShape->GetStorageShape().GetDim(DIM_1));
        tokenSizeK = static_cast<int32_t>(kShape->GetStorageShape().GetDim(DIM_1));
        tokenSizeV = static_cast<int32_t>(vShape->GetStorageShape().GetDim(DIM_1));
        context->SetTilingKey(TILING_KEY_NZ);
    } else if (strcmp(mode, "Norm") == 0) {
        blockSize = static_cast<int32_t>(kCacheShape->GetStorageShape().GetDim(DIM_1));
        tokenSizeK = static_cast<int32_t>(
            kShape->GetStorageShape().GetDim(DIM_1) * kShape->GetStorageShape().GetDim(DIM_2));
        tokenSizeV = static_cast<int32_t>(
            vShape->GetStorageShape().GetDim(DIM_1) * vShape->GetStorageShape().GetDim(DIM_2));
        context->SetTilingKey(TILING_KEY_ND + typeByte);
    } else {
        return false;
    }

    int32_t numTokens = static_cast<int32_t>(blockTablesShape->GetStorageShape().GetDim(DIM_0));
    int32_t numblkTabCol = static_cast<int32_t>(blockTablesShape->GetStorageShape().GetDim(DIM_1));

    OP_CHECK_IF((blockSize <= 0 || blockSize > INT_MAX),
                OP_LOGE(context, "blockSize is invalid."), return false);
    OP_CHECK_IF((numTokens <= 0 && numTokens > INT_MAX),
                OP_LOGE(context, "numTokens is invalid."), return false);

    GatherPaKvCacheTilingData tilingData;

    int32_t hasSeqStarts;
    auto seqStartsTensor = context->GetOptionalInputTensor(DIM_6);
    seqStartsTensor == nullptr ? hasSeqStarts = 0 : hasSeqStarts = 1;

    auto isSeqLensCumsum = attrs->GetAttrPointer<bool>(IS_SEQ_LENS_CUNSUM_INDEX);
    tilingData.set_blockSize(blockSize);
    tilingData.set_numTokens(numTokens);
    tilingData.set_numblkTabCol(numblkTabCol);
    tilingData.set_tokenSizeK(tokenSizeK);
    tilingData.set_tokenSizeV(tokenSizeV);
    tilingData.set_typeByte(typeByte);
    tilingData.set_hasSeqStarts(hasSeqStarts);
    tilingData.set_isSeqLensCumsum(*isSeqLensCumsum);

    size_t *workspaceSize = context->GetWorkspaceSizes(1);
    *workspaceSize = ASCENDC_TOOLS_WORKSPACE;

    tilingData.SaveToBuffer(context->GetRawTilingData()->GetData(),
                            context->GetRawTilingData()->GetCapacity());
    return true;
}

} // namespace optiling
```
**代码行数**：94 行