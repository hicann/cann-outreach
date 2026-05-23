# online_safe_softmax 算子设计与实施文档

---

## 0. 概述

### 0.0 需求类型判断

**判断标准**：用户未明确指定 shape 和 dtype，且描述为"online safe softmax"通用算法。

**结论**：**通用**需求，需支持任意 shape（2D 及以上可合轴为 [A1, R]）和 FP32 数据类型。

### 0.1 基本信息

| 项目 | 内容 |
|-----|------|
| 算子名称 | online_safe_softmax |
| 算子类别 | Reduction（归约类，AR 模式） |
| 需求类型 | 通用 |
| 支持数据类型 | FP32（输入/输出/中间计算均为 FP32） |
| 支持芯片 | 910B3（Ascend910B3, DAV_2201） |
| 特殊约束 | Subs/Divs 在 DAV_2201 不支持，需使用 Adds/Muls 替代 |

### 0.2 方案决策

| 决策项 | 结论 | 理由 |
|-------|------|------|
| 目标架构 | DAV_2201 | 910B3 属于 Ascend910B 系列，映射到 DAV_2201（非 DAV_3510） |
| 技术路线 | 通用 SIMD/MemBase | DAV_2201 不支持 RegBase，走传统 SIMD/MemBase 路线 |
| 算子类型 | Reduction (AR 模式) | softmax 沿最内维归约，合轴后为 (A1, R)，A0=1 → AR 模式 |
| 精度策略 | FP32 全链路 | ReduceSum Level 2 API 在 A2 仅支持 float；softmax 精度敏感，FP32 保证精度 |

---

## 1. 算子设计

### 1.1 数学公式

```
输入: x — shape=[A1, R], dtype=float32
输出: y — shape=[A1, R], dtype=float32

// Online Safe Softmax 算法（逐行计算）
对于每一行 i (i = 0, 1, ..., A1-1):

  // === FullLoad 场景（整行驻留 UB）===
  m_i   = max(x[i, 0], x[i, 1], ..., x[i, R-1])       // 行最大值
  y[i,j] = exp(x[i,j] - m_i)  for all j               // 安全移位 + 指数
  s_i   = sum(y[i, 0], y[i, 1], ..., y[i, R-1])       // 指数求和
  out[i,j] = y[i,j] / s_i  for all j                  // 归一化

  // === ColSplit Online 场景（行分 chunk 处理）===
  初始化: running_max = -INF, running_sum = 0

  Phase 1 — 累积 running_max 和 running_sum:
    For chunk c = 0, 1, ..., numChunks-1:
      chunk_max = max(x[i, c*chunkLen : (c+1)*chunkLen])
      new_max   = max(running_max, chunk_max)
      correction = exp(running_max - new_max)           // 修正因子
      running_sum = running_sum * correction + sum(exp(x[i,chunk] - new_max))
      running_max = new_max

  Phase 2 — 计算最终输出:
    For chunk c = 0, 1, ..., numChunks-1:
      out[i,chunk] = exp(x[i,chunk] - final_max) / final_sum
```

### 1.2 API 映射

| 数学操作 | 对应 API | 关键参数 | 数据布局 | 官方文档 |
|---------|---------|---------|---------|---------|
| 行最大值 | `ReduceMax<float, Pattern::Reduce::AR>` (Level 2) | `(dst, src, tmpBuf, srcShape={1,R_align}, true)` | 连续 2D，32B 内轴对齐 | [ReduceMax-91](../../../asc-devkit/docs/api/SIMD-API/高阶API/归约操作/ReduceMax接口/ReduceMax-91.md) |
| 行最大值(chunk) | `ReduceMax<float>` (Level 1, count) | `(dst, src, tmpBuf, chunkLen, false)` | 连续 1D | [ReduceMax](../../../asc-devkit/docs/api/SIMD-API/基础API/Memory矢量计算/归约计算/ReduceMax.md) |
| 行求和 | `ReduceSum<float, Pattern::Reduce::AR>` (Level 2) | `(dst, src, tmpBuf, srcShape={1,R_align}, true)` | 连续 2D，32B 内轴对齐 | [ReduceSum-90](../../../asc-devkit/docs/api/SIMD-API/高阶API/归约操作/ReduceSum接口/ReduceSum-90.md) |
| 行求和(chunk) | `ReduceSum<float>` (Level 1, count) | `(dst, src, tmpBuf, chunkLen)` | 连续 1D | [ReduceSum](../../../asc-devkit/docs/api/SIMD-API/基础API/Memory矢量计算/归约计算/ReduceSum.md) |
| 减去最大值 | `Adds<float>` (vector + scalar) | `(dst, src, -max_val, R)` | 连续 1D | [Adds](../../../asc-devkit/docs/api/SIMD-API/基础API/Memory矢量计算/基础算术/Adds.md) |
| 除以 sum | `Muls<float>` (vector × scalar) | `(dst, src, 1.0/sum_val, R)` | 连续 1D | [Muls](../../../asc-devkit/docs/api/SIMD-API/基础API/Memory矢量计算/基础算术/Muls.md) |
| 指数运算 | `Exp<float>` | `(dst, src, R)` | 连续 1D | [Exp](../../../asc-devkit/docs/api/SIMD-API/基础API/Memory矢量计算/基础算术/Exp.md) |
| GM→UB 搬入 | `DataCopyPad<float>` | `(dst, src, copyParams, padParams)` | GM 连续→UB 对齐 | DataCopyPad (asc-devkit) |
| UB→GM 搬出 | `DataCopy<float>` | `(dst, src, count)` | UB→GM 连续 | [DataCopy](../../../asc-devkit/docs/api/SIMD-API/基础API/Memory数据搬运/DataCopy/基础数据搬运.md) |

#### 1.2.1 API 语义验证

| API | 数据布局 | 功能需求 | API选择 | 限制条件 | 匹配 | 文档 |
|-----|---------|---------|---------|---------|-----|------|
| ReduceMax (Level 2) | 2D 连续，内轴 32B 对齐 | 行归约最大值 | `ReduceMax<float, Pattern::Reduce::AR>` | A2 仅支持 half/float；srcInnerPad=true 时内轴须 32B 对齐 | ✅ | ReduceMax-91.md |
| ReduceMax (Level 1) | 1D 连续 | chunk 归约最大值 | `ReduceMax<float>(dst, src, tmpBuf, count, false)` | A2 支持 half/float；dst 起始 8B 对齐 | ✅ | ReduceMax.md |
| ReduceSum (Level 2) | 2D 连续，内轴 32B 对齐 | 行归约求和 | `ReduceSum<float, Pattern::Reduce::AR>` | **A2 仅支持 float**（不支持 half！） | ✅ | ReduceSum-90.md |
| ReduceSum (Level 1) | 1D 连续 | chunk 归约求和 | `ReduceSum<float>(dst, src, tmpBuf, count)` | A2 支持 half/float | ✅ | ReduceSum.md |
| Adds | 1D 连续 | vector + (-max_scalar) = vector - max | `Adds<float>(dst, src, -max_val, count)` | A2 支持 half/float/int16_t/int32_t；**Subs 在 A2 不支持，用 Adds 传负值替代** | ✅ | Adds.md |
| Muls | 1D 连续 | vector × (1/sum_scalar) = vector / sum | `Muls<float>(dst, src, 1.0/sum, count)` | A2 支持 half/float/int16_t/int32_t；**Divs 在 A2 不支持，用 Muls 传倒数替代** | ✅ | Muls.md |
| Exp | 1D 连续 | 自然指数 | `Exp<float>(dst, src, count)` | A2 支持 half/float；dst/src 32B 对齐 | ✅ | Exp.md |
| DataCopyPad | GM→UB | 搬入 + 非 32B 对齐填充 | `DataCopyPad(dst, src, copyParams, padParams)` | 无特殊限制 | ✅ | asc-devkit |

**验证清单**：
- [x] 1. 数据布局确认：行数据连续排列，非对齐时用 DataCopyPad 填充
- [x] 2. 功能需求明确：行级 max/sum 归约 + 逐元素 sub/exp/div
- [x] 3. 已查阅官方文档（所有 API 文档路径已记录）
- [x] 4. 匹配验证：Subs/Divs 不可用 → Adds(-val)/Muls(1/val) 替代方案可行
- [x] 5. 已记录验证过程

**⚠️ 关键 API 替代决策**：
- **Subs 不可用**（DAV_2201 不支持）→ 使用 `Adds(dst, src, -scalar, count)` 实现矢量减标量
- **Divs 不可用**（DAV_2201 不支持）→ 使用 `Muls(dst, src, 1.0/scalar, count)` 实现矢量除标量
- **ReduceSum Level 2 仅支持 float**（DAV_2201 不支持 half）→ 统一使用 FP32 计算

### 1.3 数据流

#### FullLoad 场景（R ≤ 全载阈值）

```
GM row[i] (R elements)
    ↓ DataCopyPad
UB xLocal (R_align elements, 32B padding)
    ↓ ReduceMax<float, Pattern::Reduce::AR>(dst, src, tmpBuf, {1,R_align}, true)
UB maxLocal (1 scalar)
    ↓ GetValue(0) → max_val
    ↓ Adds<float>(xLocal, xLocal, -max_val, R)
UB xLocal (shifted: x - max)
    ↓ Exp<float>(xLocal, xLocal, R)   [in-place, src=dst]
UB xLocal (exp values)
    ↓ ReduceSum<float, Pattern::Reduce::AR>(sumLocal, xLocal, tmpBuf, {1,R_align}, true)
UB sumLocal (1 scalar)
    ↓ GetValue(0) → sum_val; inv_sum = 1.0 / sum_val
    ↓ Muls<float>(xLocal, xLocal, inv_sum, R)
UB xLocal (final softmax output)
    ↓ DataCopy
GM out[i] (R elements)
```

#### ColSplit Online 场景（R > 全载阈值）

```
Phase 1: 累积 running_max / running_sum (遍历 chunks)
  For each chunk c:
    GM row[i][chunk_offset] (chunkLen elements)
        ↓ DataCopyPad
    UB chunkLocal (chunkLen_align)
        ↓ ReduceMax<float>(maxLocal, chunkLocal, tmpBuf, chunkLen, false)
    UB maxLocal → chunk_max_val
        ↓ scalar compare: new_max = max(running_max, chunk_max_val)
        ↓ correction = exp(running_max - new_max)
        ↓ running_sum = running_sum * correction
        ↓ Adds<float>(chunkLocal, chunkLocal, -new_max, chunkLen)
    UB chunkLocal (shifted)
        ↓ Exp<float>(chunkLocal, chunkLocal, chunkLen)
    UB chunkLocal (exp values)
        ↓ ReduceSum<float>(sumLocal, chunkLocal, tmpBuf, chunkLen)
    UB sumLocal → partial_sum
        ↓ running_sum += partial_sum
        ↓ running_max = new_max

Phase 2: 计算最终输出 (遍历 chunks)
  For each chunk c:
    GM row[i][chunk_offset] (chunkLen elements)
        ↓ DataCopyPad
    UB chunkLocal
        ↓ Adds<float>(chunkLocal, chunkLocal, -final_max, chunkLen)
        ↓ Exp<float>(chunkLocal, chunkLocal, chunkLen)
        ↓ Muls<float>(chunkLocal, chunkLocal, 1.0/final_sum, chunkLen)
    UB chunkLocal (final output for this chunk)
        ↓ DataCopy
    GM out[i][chunk_offset] (chunkLen elements)
```

### 1.4 核心计算步骤

**核心计算步骤**：
```
1. 搬入行数据 → UB
2. 求行最大值（ReduceMax）
3. 安全移位：减去最大值（Adds 传负值）
4. 指数运算（Exp）
5. 指数求和（ReduceSum）
6. 归一化：除以求和值（Muls 传倒数）
7. 搬出结果 → GM
```

**分支差异对比**：

| 操作 | FullLoad | ColSplit Online |
|------|---------|-----------------|
| 数据搬入 | 整行一次搬入 | 分 chunk 两次遍历搬入（Phase1 + Phase2） |
| ReduceMax | Level 2 Pattern::AR | Level 1 count-based（每 chunk） |
| ReduceSum | Level 2 Pattern::AR | Level 1 count-based（每 chunk） |
| running_max/sum | 不需要（整行在 UB） | 需要维护标量状态，跨 chunk 累积 |
| 精度风险 | 低（整行驻留 UB） | 需注意 correction 因子精度 |

**关键设计要点**：
1. **Buffer 使用**: xLocal 可 in-place 复用（减→指数→归一化逐步变换），节省 UB 空间
2. **API 选择**: Subs/Divs 不可用 → Adds(-val)/Muls(1/val) 替代，性能无损（Muls 比 Div 更快）
3. **参数含义**: ReduceMax/ReduceSum count 参数用 `rLength`（有效长度），Buffer 大小用 `rLengthAlign`（对齐长度）
4. **Online 修正**: ColSplit 中 running_sum 乘 exp(running_max - new_max) 修正先前累积值

**参数使用规则**：
| 参数位置 | 用有效长度 (rLength) | 用对齐长度 (rLengthAlign) |
|---------|:---:|:---:|
| DataCopyPad blockLen / 计算 API count | ✓ | ✗ |
| UB 数据偏移 / Buffer 大小分配 | ✗ | ✓ |

### 1.5 内存管理(Buffer 规划)

#### FullLoad 场景 Buffer 规划

| Buffer 名称 | 用途 | 大小计算 | TPosition |
|------------|------|---------|-----------|
| inQueueX | 输入数据 + 中间计算复用 | 2 × R_align × sizeof(float) | VECIN |
| outQueueY | 最终输出 | 2 × R_align × sizeof(float) | VECOUT |
| tmpBuf | ReduceMax/ReduceSum 临时空间 | ComputeReduceBufSize(R_align, 4) | VECCALC |
| maxBuf | ReduceMax 结果（1 标量） | 32 字节（1 block） | VECCALC |
| sumBuf | ReduceSum 结果（1 标量） | 32 字节（1 block） | VECCALC |

**总 UB 使用量**: `4 × R_align × 4 + tmpBufSize + 64` 字节

**全载阈值计算**: `R_align_max = (UB_SIZE - tmpBufSize - 64) / 16`
- UB_SIZE = 192KB = 196608
- tmpBufSize ≈ max(4KB, R_align 相关计算)
- 保守估算 R_max ≈ 11000 元素（FP32）

#### ColSplit Online 场景 Buffer 规划

| Buffer 名称 | 用途 | 大小计算 | TPosition |
|------------|------|---------|-----------|
| chunkQueueX | chunk 输入 + 中间计算复用 | 2 × chunkLen_align × sizeof(float) | VECIN |
| chunkQueueY | chunk 输出 | 2 × chunkLen_align × sizeof(float) | VECOUT |
| tmpBuf | ReduceMax/ReduceSum 临时空间 | ComputeReduceBufSize(chunkLen_align, 4) | VECCALC |
| maxBuf | ReduceMax chunk 结果 | 32 字节 | VECCALC |
| sumBuf | ReduceSum chunk 结果 | 32 字节 | VECCALC |

**chunkLen 选择**: 需满足 `4 × chunkLen_align × 4 + tmpBufSize + 64 ≤ UB_SIZE`
- chunkLen_align = FP32 下 8 元素对齐（32B / 4B = 8）
- 典型 chunkLen = 2048~4096 元素

---

## 2. 架构设计

### 2.1 多核切分策略

| 项目 | 说明 |
|-----|------|
| 切分维度 | 按 A1（行数）切分，每核处理若干行 |
| 单核任务量 | `rowsPerCore = ceil(A1 / blockDim)` |
| 使用的核数 | **强制动态计算**: `usedCoreNum = ceil(A1 / rowsPerCore)`，由 PlatformAscendC.GetCoreNumAiv() 获取可用核数 |
| 负载均衡方式 | 尾核可能少一行，通过 Tiling 传递 `tailCoreRows` |

```cpp
// Host 侧 Tiling 计算
uint32_t totalCoreNum = platform.GetCoreNumAiv();
uint32_t rowsPerCore = (A1 + totalCoreNum - 1) / totalCoreNum;
uint32_t usedCoreNum = (A1 + rowsPerCore - 1) / rowsPerCore;
uint32_t tailCoreRows = A1 - rowsPerCore * (usedCoreNum - 1);
```

### 2.2 UB 切分策略

| 项目 | 说明 |
|-----|------|
| UB 容量 | 192KB (DAV_2201) |
| 单次处理数据量 | FullLoad: R 元素/行; ColSplit: chunkLen 元素/chunk |
| 是否需要分 chunk | R > 全载阈值时需要 |
| chunk 大小计算公式 | `chunkLen = min(R, floor((UB_SIZE - overhead) / (4 × typeSize)))` |

**全载判定公式**:
```
全载条件: 4 × R_align × sizeof(float) + tmpBufSize + 64 ≤ UB_SIZE
R_align = ceil(R * sizeof(float) / 32) * 32 / sizeof(float)
tmpBufSize = ComputeReduceBufSize(R_align, sizeof(float))
```

### 2.3 分支场景覆盖

| 分支条件 | 处理策略 |
|---------|---------|
| 数据类型 | FP32 全链路；若未来支持 FP16 输入 → Cast FP16→FP32 计算 → Cast FP32→FP16 输出 |
| 大 shape (R > 全载阈值) | ColSplit Online: Phase1 累积 max/sum + Phase2 计算输出 |
| 小 shape (R ≤ 全载阈值) | FullLoad: 整行驻留 UB，顺序计算 |
| 对齐 (R * 4 % 32 == 0) | 直接 DataCopy，ReduceMax/ReduceSum Level 2 |
| 非对齐 | DataCopyPad 填充对齐，ReduceMax/ReduceSum Level 2 srcInnerPad=true |
| 边界情况 (R ≤ 8) | 最小行长度，需确保 ReduceMax/ReduceSum count ≥ 1 |
| 极大 R (单核 R chunk 过多) | chunkLen 合理设置，Phase2 遍历 chunk 数可控 |

### 2.4 类别特有设计

#### 2.4.1 FullLoad 分支（R ≤ 全载阈值）

**适用场景**: 整行数据可在 UB 中驻留，一次性完成全部计算

**Compute 核心流程伪代码**:

```cpp
// FullLoad: 逐行处理，每行整行驻留 UB
for (uint32_t row = 0; row < rowsThisCore; row++) {
    uint32_t rowOffset = row * rLengthAlign;  // UB 内偏移用对齐长度

    // 1. CopyIn: 搬入整行数据
    DataCopyPad(xLocal, xGm[gmOffset], copyParams, padParams);

    // 2. ReduceMax: 求行最大值
    uint32_t srcShape[] = {1, rLengthAlign};
    ReduceMax<float, Pattern::Reduce::AR>(maxLocal, xLocal[rowOffset],
        tmpLocal, srcShape, true);
    float maxVal = maxLocal.GetValue(0);

    // 3. Safe shift: 减去最大值
    Adds<float>(xLocal[rowOffset], xLocal[rowOffset], -maxVal, rLength);

    // 4. Exp: 指数运算
    Exp<float>(xLocal[rowOffset], xLocal[rowOffset], rLength);

    // 5. ReduceSum: 指数求和
    ReduceSum<float, Pattern::Reduce::AR>(sumLocal, xLocal[rowOffset],
        tmpLocal, srcShape, true);
    float sumVal = sumLocal.GetValue(0);
    float invSum = 1.0f / sumVal;

    // 6. Normalize: 除以求和值
    Muls<float>(xLocal[rowOffset], xLocal[rowOffset], invSum, rLength);

    // 7. CopyOut: 搬出结果
    DataCopy(yGm[gmOutOffset], xLocal[rowOffset], rLengthAlign);
}
```

**Buffer 需求**:

| Buffer 名称 | 用途 | 大小计算 |
|------------|------|---------|
| inQueueX | 行数据 + 中间计算 | 2 × rLengthAlign × 4 |
| outQueueY | 最终输出 | 2 × rLengthAlign × 4 |
| tmpBuf | Reduce 临时空间 | ComputeReduceBufSize(rLengthAlign, 4) |
| maxBuf | ReduceMax 结果 | 32 |
| sumBuf | ReduceSum 结果 | 32 |

#### 2.4.2 ColSplit Online 分支（R > 全载阈值）

**适用场景**: 行长度超出 UB 全载能力，需分 chunk 处理

**Compute 核心流程伪代码**:

```cpp
// ColSplit Online: 逐行分 chunk 处理
for (uint32_t row = 0; row < rowsThisCore; row++) {
    float runningMax = -3.4e38f;  // -FLT_MAX as initial
    float runningSum = 0.0f;

    // ===== Phase 1: 累积 running_max 和 running_sum =====
    for (uint32_t chunk = 0; chunk < numChunks; chunk++) {
        uint32_t chunkOffset = chunk * chunkLen;
        uint32_t curChunkLen = min(chunkLen, rLength - chunkOffset);

        // 1. CopyIn chunk
        DataCopyPad(chunkLocal, xGm[rowOffset + chunkOffset],
            chunkCopyParams, chunkPadParams);

        // 2. ReduceMax chunk
        ReduceMax<float>(maxLocal, chunkLocal, tmpLocal, curChunkLen, false);
        float chunkMax = maxLocal.GetValue(0);

        // 3. Update running_max/sum (online correction)
        float newMax = (chunkMax > runningMax) ? chunkMax : runningMax;
        float correction = expf(runningMax - newMax);
        runningSum = runningSum * correction;

        // 4. Safe shift chunk
        Adds<float>(chunkLocal, chunkLocal, -newMax, curChunkLen);

        // 5. Exp chunk
        Exp<float>(chunkLocal, chunkLocal, curChunkLen);

        // 6. ReduceSum exp values
        ReduceSum<float>(sumLocal, chunkLocal, tmpLocal, curChunkLen);
        float partialSum = sumLocal.GetValue(0);
        runningSum += partialSum;
        runningMax = newMax;
    }

    float finalMax = runningMax;
    float invFinalSum = 1.0f / runningSum;

    // ===== Phase 2: 计算最终输出 =====
    for (uint32_t chunk = 0; chunk < numChunks; chunk++) {
        uint32_t chunkOffset = chunk * chunkLen;
        uint32_t curChunkLen = min(chunkLen, rLength - chunkOffset);

        // 1. CopyIn chunk (再次读取原始数据)
        DataCopyPad(chunkLocal, xGm[rowOffset + chunkOffset],
            chunkCopyParams, chunkPadParams);

        // 2. Safe shift with final max
        Adds<float>(chunkLocal, chunkLocal, -finalMax, curChunkLen);

        // 3. Exp
        Exp<float>(chunkLocal, chunkLocal, curChunkLen);

        // 4. Normalize with final sum
        Muls<float>(chunkLocal, chunkLocal, invFinalSum, curChunkLen);

        // 5. CopyOut chunk
        DataCopy(yGm[outRowOffset + chunkOffset], chunkLocal, curChunkLenAlign);
    }
}
```

**Buffer 需求**:

| Buffer 名称 | 用途 | 大小计算 |
|------------|------|---------|
| chunkQueueX | chunk 数据 + 中间计算 | 2 × chunkLenAlign × 4 |
| chunkQueueY | chunk 输出 | 2 × chunkLenAlign × 4 |
| tmpBuf | Reduce 临时空间 | ComputeReduceBufSize(chunkLenAlign, 4) |
| maxBuf | chunk ReduceMax 结果 | 32 |
| sumBuf | chunk ReduceSum 结果 | 32 |

---

## 3. 确认清单

- [x] 多核切分策略已确定（按 A1 行切分，动态核数）
- [x] UB 切分策略已确定（FullLoad 整行 / ColSplit 分 chunk）
- [x] Buffer 规划已完成（Double Buffer + tmpBuf + 结果 Buf）
- [x] 分支场景已覆盖（FullLoad / ColSplit Online / 对齐 / 非对齐 / 小 shape）
- [x] 类别特有设计已完成（两分支伪代码 + Buffer 需求）
- [x] API 替代方案已确认（Adds 替代 Subs, Muls 替代 Divs）
- [x] 精度策略已确定（FP32 全链路，社区标准 threshold=2^-13）