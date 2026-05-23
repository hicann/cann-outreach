# 代码概要

算子: gather_pa_kv_cache | 功能: Paged Attention KV Cache 的 Tiling 参数计算与校验，支持 PA_NZ 和 Norm 两种缓存模式 | 侧别: Tiling

## 代码脉络

**入口**: `CommonGatherPaKvCacheTiling` → 被 Tiling 框架调用（注册为 TilingFunc） → 触发条件: 算子计算前由 GE 框架自动调用

**数据流**:
输入 Tensor Shape（kCache、blockTables） + Attrs（mode、isSeqLensCumsum） → 模式分支判断 → 参数计算（blockSize、tokenSizeK/V、TilingKey） → 参数校验 → TilingData 结构填充 → 保存到 TilingContext buffer

**计算核心**: 无（Tiling 侧不执行实际计算，仅做参数计算和校验）

**分支覆盖**:
| 分支条件 | 位置(文件:行) | 触发场景 | 处理逻辑 | 涉及 API |
|---------|-------------|---------|---------|----------|
| mode == "PA_NZ" | tiling_common_demo.txt:46-50 | 使用 Paged Attention NZ 格式缓存 | blockSize=shape[1], tokenSizeK/V=shape[1], TilingKey=577 | GetStorageShape, SetTilingKey |
| mode == "Norm" | tiling_common_demo.txt:51-57 | 使用标准格式缓存 | blockSize=shape[1], tokenSizeK/V=shape[1]*shape[2], TilingKey=617/618 | GetStorageShape, SetTilingKey |
| 其他 | tiling_common_demo.txt:58-59 | 不支持的模式 | 返回 false | - |
| seqStartsTensor == nullptr | tiling_common_demo.txt:73-74 | 无 seqStarts 可选输入 | hasSeqStarts=0 | GetOptionalInputTensor |
| seqStartsTensor != nullptr | tiling_common_demo.txt:73-74 | 有 seqStarts 可选输入 | hasSeqStarts=1 | GetOptionalInputTensor |

**关键变量流转**:
| 变量 | 来源 | 用途 | 流转路径 |
|------|------|------|----------|
| blockSize | Shape[1] | KV Cache 块大小 | kCacheShape → tilingData.set_blockSize() |
| numTokens | blockTables Shape[0] | 待处理的 token 数量 | blockTablesShape → tilingData.set_numTokens() |
| tokenSizeK | Shape（分支计算） | K tensor 每个 token 的维度 | kShape → 分支计算 → tilingData.set_tokenSizeK() |
| tokenSizeV | Shape（分支计算） | V tensor 每个 token 的维度 | vShape → 分支计算 → tilingData.set_tokenSizeV() |
| TilingKey | mode + typeByte | Kernel 分支选择标识 | mode 分支 → SetTilingKey() → Kernel if TILING_KEY_IS |
| hasSeqStarts | 可选输入存在性 | 是否使用 seqStarts | GetOptionalInputTensor → 判断 → tilingData.set_hasSeqStarts() |

**核心 API**: 
- Tiling 框架 API: `gert::TilingContext`, `GetInputShape`, `GetOutputShape`, `GetAttrs`, `SetTilingKey`, `GetOptionalInputTensor`, `SaveToBuffer`
- 校验 API: `OP_CHECK_IF`, `OP_LOGE`
- 辅助 API: `GetTensorElementSizes`, `GetStorageShape`, `GetDim`

**输出**: TilingData 结构 → 保存到 `context->GetRawTilingData()` buffer → Kernel 侧通过 `GET_TILING_DATA_WITH_STRUCT` 获取

## 变量溯源

| 变量 | 声明(文件:行) | 初始化(文件:行) | 校验(文件:行) | 来源类型 |
|------|-------------|----------------|-------------|---------|
| blockSize | tiling_common_demo.txt:40 | tiling_common_demo.txt:47/52（分支内） | tiling_common_demo.txt:65（`<=0` 或 `>INT_MAX`） | 外部输入（Shape） |
| numTokens | tiling_common_demo.txt:62 | tiling_common_demo.txt:62（从shape获取） | tiling_common_demo.txt:67（`<=0 && >INT_MAX`，条件有误） | 外部输入（Shape） |
| numblkTabCol | tiling_common_demo.txt:63 | tiling_common_demo.txt:63（从shape获取） | 无 | 外部输入（Shape） |
| tokenSizeK | tiling_common_demo.txt:41 | tiling_common_demo.txt:48/53-54（分支内） | 无 | 外部输入（Shape） |
| tokenSizeV | tiling_common_demo.txt:42 | tiling_common_demo.txt:49/55-56（分支内） | 无 | 外部输入（Shape） |
| typeByte | tiling_common_demo.txt:44 | tiling_common_demo.txt:44（从dtype计算） | 无 | 外部输入（dtype） |
| hasSeqStarts | tiling_common_demo.txt:72 | tiling_common_demo.txt:74（条件判断） | 无 | 外部输入（可选tensor） |
| isSeqLensCumsum | - | tiling_common_demo.txt:84（从attrs获取） | 无 | 外部输入（Attrs） |
| mode | - | tiling_common_demo.txt:38（从attrs获取） | tiling_common_demo.txt:46/51（字符串比较） | 外部输入（Attrs） |

> 来源类型判定：
> - 外部输入（Shape/dtype/Attrs/可选tensor）：来自用户输入，需校验
> - TilingData 传递：已校验，无需重复
> - 硬件配置：硬件保证
> - 编译期常量：编译期固定

**⚠️ 校验问题**:
- `numTokens` 校验条件逻辑错误（第67行）：`numTokens <= 0 && numTokens > INT_MAX` 永远为 false，应为 `||`
- `blockSize` 校验条件冗余：`blockSize > INT_MAX` 在 `blockSize` 为 int32_t 类型时永远不会触发（int32_t 最大值即 INT_MAX）

## 代码关联

**上游文件**:
| 文件路径 | 关联方式 | 依据 |
|----------|----------|------|
| gather_pa_kv_cache_tiling.h（即 tiling_data_def_demo.txt） | include | 代码中 `#include "gather_pa_kv_cache_tiling.h"` |
| tiling/tiling_api.h | include | 代码中 `#include "tiling/tiling_api.h"` |
| register/op_impl_registry.h | include | TilingData 定义文件引用 |
| register/tilingdata_base.h | include | TilingData 定义文件引用 |

**下游文件**:
| 文件路径/API | 关联方式 | 依据 |
|----------|----------|------|
| kernel_dispatch_demo.txt（Kernel 入口） | TilingData 传递 | Kernel 通过 `GET_TILING_DATA_WITH_STRUCT(GatherPaKvCacheTilingData, tilingData, tiling)` 获取 Tiling 参数 |
| Tiling 框架 | 框架依赖 | `gert::TilingContext` 等框架 API |
| OP_CHECK_IF/OP_LOGE | 校验 API | 参数校验宏 |

**跨文件数据流**:
```
Tiling 侧（tiling_common_demo.txt）
  ↓ GatherPaKvCacheTilingData
  ↓ blockSize, numTokens, numblkTabCol, tokenSizeK/V, typeByte, hasSeqStarts, isSeqLensCumsum
  ↓ SaveToBuffer
  ↓
Kernel 侧（kernel_dispatch_demo.txt）
  ↓ GET_TILING_DATA_WITH_STRUCT
  ↓ 根据 TilingKey（577/617/618）选择 Kernel 实现
  ↓ GatherPaKvCacheNd<half/int8_t> 或 GatherPaKvCacheNz<int8_t>
```

## Tiling 侧设计分析

**参数计算策略**:
| 参数 | 计算方式 | 依赖条件 |
|------|----------|----------|
| blockSize | kCacheShape.GetStorageShape().GetDim(DIM_1) | 两种模式相同 |
| tokenSizeK（PA_NZ） | kShape.GetStorageShape().GetDim(DIM_1) | 仅 shape[1] |
| tokenSizeK（Norm） | kShape.GetStorageShape().GetDim(DIM_1) * GetDim(DIM_2) | shape[1] * shape[2] |
| tokenSizeV（PA_NZ） | vShape.GetStorageShape().GetDim(DIM_1) | 仅 shape[1] |
| tokenSizeV（Norm） | vShape.GetStorageShape().GetDim(DIM_1) * GetDim(DIM_2) | shape[1] * shape[2] |
| TilingKey（PA_NZ） | 固定值 577 | - |
| TilingKey（Norm） | 617 + typeByte | 617=FP16, 618=INT8 |

**模式切换设计**:
- 通过 `mode` 属性区分 PA_NZ 和 Norm 两种缓存格式
- 通过 `typeByte` 区分数据类型（FP16 vs INT8）
- TilingKey 编码规则：高位表示模式，低位表示数据类型
- Kernel 侧通过 `TILING_KEY_IS` 宏分支选择对应实现

**可选输入处理**:
- `seqStarts` 为可选输入，通过 `GetOptionalInputTensor(DIM_6)` 检测
- `isSeqLensCumsum` 为属性参数，从 `attrs->GetAttrPointer<bool>` 获取
- 两者均传递给 Kernel 侧用于分支判断

**Workspace 设计**:
- 固定大小：8MB (`ASCENDC_TOOLS_WORKSPACE = 8 * 1024 * 1024`)
- 无动态计算逻辑

**校验覆盖度分析**:
| 参数 | 是否校验 | 校验条件 | 问题 |
|------|---------|---------|------|
| blockSize | ✅ | `<=0 \|\| >INT_MAX` | `>INT_MAX` 冗余 |
| numTokens | ✅ | `<=0 && >INT_MAX` | 条件逻辑错误 |
| numblkTabCol | ❌ | 无 | 缺失校验 |
| tokenSizeK | ❌ | 无 | 缺失校验 |
| tokenSizeV | ❌ | 无 | 缺失校验 |
| mode | ✅ | 字符串比较 | 不匹配时返回 false |
| 其他 | ❌ | 无 | - |

**设计亮点**:
1. 支持 PA_NZ 和 Norm 两种缓存格式，通过 TilingKey 编码实现 Kernel 分支
2. 可选输入 seqStarts 通过 nullptr 检测，避免空指针访问
3. 参数通过 TilingData 结构统一传递，降低 Kernel 侧复杂度

**设计缺陷**:
1. numTokens 校验条件逻辑错误（应为 `||` 而非 `&&`），导致校验失效
2. 多个关键参数（numblkTabCol、tokenSizeK/V）缺少边界校验
3. 模式匹配失败时，部分变量（blockSize、tokenSizeK/V）未初始化就返回，虽有返回值保护但逻辑不清晰