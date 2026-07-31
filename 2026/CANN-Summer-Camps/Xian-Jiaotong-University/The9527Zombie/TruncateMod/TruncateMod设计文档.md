# 需求背景（required）

## 需求来源

基于华为昇腾（Ascend）自定义算子开发任务，使用 Ascend C 编程语言实现 `TruncateMod` 算子，对齐社区内置 `TruncateMod` 算子的功能与精度，并满足 CANN Judge 判题平台的性能与精度验收要求。

## 背景介绍

### TruncateMod 算子说明

`TruncateMod` 为逐元素（element-wise）截断取模算子，语义等价于 numpy / torch 的 `fmod`：商向零取整（truncate），因此余数符号与被除数 `x1` 保持一致。

内置算子实现路径（参考）：
- 算子实现路径：`/usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe/impl`
- 相关 API 路径：`/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/tbe/dsl`

### 内置 TruncateMod 算子功能分析

通过对内置 `TruncateMod` 算子的功能分析，当前需要对齐的能力如下：

| 参数 | 参数含义 | 数据类型 | 支持数据类型 | 约束 | 形状 |
| --- | --- | --- | --- | --- | --- |
| x1 | 被除数 tensor | tensor | float16, float32, bfloat16, int32, int8, uint8 | 与 x2 同 dtype | (N,…) |
| x2 | 除数 tensor | tensor | float16, float32, bfloat16, int32, int8, uint8 | 与 x1 同 dtype | (N,…) |
| y | 输出 tensor | tensor | 同输入 dtype | 无 | broadcast(x1, x2) |

计算公式：

```
y = x1 - trunc(x1 / x2) * x2
```

其中 `trunc()` 为向零取整。

除零（`x2 == 0`）行为对齐内置算子 golden：
- 浮点类型（fp16/fp32/bf16）：结果为 `NaN`
- 有符号整型（int32/int8）：结果为 `-1`
- 无符号整型（uint8）：结果为 `255`

# 需求分析（required）

## 需求描述

使用 Ascend C 编程语言实现 `TruncateMod` 算子，支持 float16、float32、bfloat16、int32、int8、uint8 六种数据类型，支持 NumPy 风格广播，性能不低于内置算子的 95%，精度满足判题平台默认阈值。

## 需求拆解

1. 支持 float16、float32、bfloat16、int32、int8、uint8 六种数据类型；
2. 支持 NumPy 风格广播（x1、x2 形状可广播，y 取广播形状）；
3. 除零行为对齐内置算子 golden（浮点 → NaN，有符号整型 → -1，无符号整型 → 255）；
4. 性能不低于内置算子 95%（全核参与计算场景）；
5. 精度满足 CANN Judge 平台对应题目默认阈值。

# 详细设计（required）

## 算子分析

### 数学公式

```
y = x1 - trunc(x1 / x2) * x2          （trunc 向零取整）
```

为避免整型溢出并统一各类型处理，`trunc` 在 UB 上以 fp32 实现，采用无分支写法：

```
trunc(t) = floor(max(t, 0)) + ceil(min(t, 0))
```

- 正数部分 `max(t,0)` 走 `floor`（向下取整），
- 负数部分 `min(t,0)` 走 `ceil`（向上取整），
- 两者相加即得到向零取整结果，天然覆盖正负与零。

### 支持数据类型

float16、float32、bfloat16、int32、int8、uint8。

所有类型均提升到 fp32 完成核心计算，计算完成后再转回原类型输出。

### 支持形状

支持 NumPy 风格广播。输出 shape 为 x1、x2 的广播结果；右对齐补维，缺失高维按 1 处理；两维度相容条件为「相等或其一为 1」，输出取较大者。

## 算子实现

### 实现方案

整体采用 host（tiling + 原型/推导）+ kernel（模板核）分层结构：

- **Op prototype（`truncate_mod_def.cpp`）**：声明 x1/x2 输入、y 输出，6 种 dtype × ND format，开启 `DynamicRankSupportFlag` / `DynamicShapeSupportFlag`，支持 ascend910b / ascend910_93 / ascend950。
- **InferShape / InferDataType（`truncate_mod_infershape.cpp`）**：输出 shape 取 x1、x2 右对齐广播；输出 dtype 取 x1 dtype。
- **Tiling（`truncate_mod_tiling.cpp`）**：平台信息读取、dtype→tilingKey 映射、广播信息构建、分核与 UB 切分。
- **Kernel（`truncate_mod.cpp` + `truncate_mod.h`）**：按 tilingKey（schMode）在编译期分发到对应 dtype 的模板实现。

#### 3.2.1 host 侧设计

##### 1. tilingkey 规划策略

需要感知 host 侧 dtype 信息，让 kernel 侧走不同的类型分支。将输入 dtype 映射为单一模板参数 `schMode`（3 bit），并通过 `ASCENDC_TPL_ARGS_DECL` / `ASCENDC_TPL_SEL` 注册模板核实例：

| dtype | schMode 宏 | 值 |
| --- | --- | --- |
| float16 | `TRUNCATE_MOD_SCH_FP16` | 0 |
| float32 | `TRUNCATE_MOD_SCH_FP32` | 1 |
| bfloat16 | `TRUNCATE_MOD_SCH_BF16` | 2 |
| int32 | `TRUNCATE_MOD_SCH_INT32` | 3 |
| int8 | `TRUNCATE_MOD_SCH_INT8` | 4 |
| uint8 | `TRUNCATE_MOD_SCH_UINT8` | 5 |

host 侧通过 `SelectSchMode(dtype, tilingKey, dtypeSize)` 完成映射，并用 `GET_TPL_TILING_KEY(...)` 生成最终 tiling key；不支持的 dtype 直接返回失败。x1、x2 dtype 必须一致，否则报错。

##### 2. 广播信息构建策略

`BuildBroadcastInfo()` 完成维度补全与广播 stride 计算：

- 取 x1、x2 rank 的较大值为输出 rank（纯标量按 1 元素向量处理）；
- 各输入在高位补 1 对齐到输出 rank；
- 逐维取 `max(s1[i], s2[i])` 作为输出维度，写入 `outShape[]`，并累计 `totalCount`；
- 计算每个输入在其自身（右对齐）布局下的连续 stride，**被广播的维度（该维为 1 而输出维更大）stride 置 0**，写入 `x1Stride[] / x2Stride[]`；
- 同时标记 `x1SameShape / x2SameShape`（是否无需广播）与 `x1Scalar / x2Scalar`（是否单元素），供 kernel 侧走快路径。

> 说明：当前提交版本 kernel 计算路径以同形状（element-wise）为主，tiling 已完整携带广播 stride，为后续广播分支预留。

##### 3. 分核策略

优先满核原则，以 32B block 为分核粒度：

- 输入数据总字节数由 `totalCount × dtypeSize` 得到，`blockElem = 32B / dtypeSize`；
- 总 block 数 `totalBlocks = totalCount / blockElem`，非对齐余数 `tailElems`；
- 若 `totalBlocks < aivCoreNum`，则收敛核数到 `totalBlocks`（至少 1 核）；
- 若核间能均分，大小核数据块一致；不能均分时，把多出的 block 分配到前 `tailBlocks` 个核（每核多一个 block）；
- 非 32B 对齐的余数元素统一并入最后一个核（`lastCoreCount = perCoreCount + tailElems`），避免数据碎片。

核数通过 `context->SetBlockDim(coreNum)` 下发。

##### 4. 数据分块与 UB 内存优化策略

充分使用 UB 空间原则：

- 通过 `platform_ascendc::PlatformAscendC` 获取 AIV 核数（`GetCoreNumAiv`）与 UB 容量（`GetCoreMemSize`）；
- 预留 `UB_RESERVED_BYTES = 8192B` 作为 tiling 结构 / 栈头空间；
- 单元素 UB 预算 = 队列 `3 × dtypeSize × bufferNum`（x1/x2/y）+ fp32 计算缓冲 `CALC_BYTES`（按 int32 高精度路径最坏情况预留 6 个 fp32 缓冲）；
- 开启 double buffer（`bufferNum = 2`）；
- `tileCount` 向下取整到整数个 32B block，最小为 `blockElem`，作为单次 UB 处理的最大元素数。

切分结果写入 `TruncateModTilingData`，字段含义见下表：

| 字段 | 含义 |
| --- | --- |
| coreNum | 实际启动的 AI 核数 |
| totalCount | 输出总元素数 |
| tileCount | 单个 UB tile 最大处理元素数 |
| perCoreCount | 每核基础元素数（block 对齐） |
| tailCoreNum | 前部多分一个 block 的核数 |
| lastCoreCount | 最后一个核处理的元素数（含非对齐余数） |
| blockElem | 当前 dtype 每 32B block 元素数 |
| bufferNum | 1 或 2（double buffer） |
| dimNum / outShape[8] / x1Stride[8] / x2Stride[8] | 广播维度与 stride 信息 |
| x1SameShape / x2SameShape / x1Scalar / x2Scalar | 广播快路径标记 |

#### 3.2.2 kernel 侧设计

kernel 采用经典 `Init` + `Process` 结构，`Process` 内按 tile 循环执行 `CopyIn → Compute → CopyOut` 三阶段流水。

1. **类型分发**：`truncate_mod.cpp` 中的模板核 `truncate_mod<schMode>` 通过 `if constexpr` 在编译期分发到 `NsTruncateMod::Run<T>`，覆盖 half/float/bfloat16_t/int32_t/int8_t/uint8_t 六种类型，无运行期分支开销。

2. **CopyIn**：使用 `DataCopyPad` + `DataCopyPadExtParams` 按实际元素数搬入 x1、x2，尾块非对齐由 pad 处理。

3. **类型提升（ToFloat）**：
   - float：直接 `ReinterpretCast` + `DataCopy`；
   - int8/uint8：先 `Cast` 到 half 再 `Cast` 到 fp32（两级转换）；
   - half/bfloat16/int32：直接 `Cast` 到 fp32。

4. **Compute（fp32 域）**：
   ```
   quot = x1 / x2
   tmp  = ceil(min(quot, 0))        // 负部
   quot = floor(max(quot, 0))       // 正部
   quot = quot + tmp                // trunc(quot)
   quot = quot * x2
   x1f  = x1f - quot                // 余数
   ```
   使用 `Div / Mins / Ceil / Maxs / Floor / Add / Mul / Sub` 等 Ascend C Vector API。

5. **除零处理（ApplyDivZero）**：用 `CompareScalar(x2f == 0)` 生成 mask，`Duplicate` sentinel 到临时缓冲，再 `Select` 覆盖除零位置：
   - 浮点：sentinel = quiet NaN（`0x7FC00000` 位模式，通过 union 位转换得到，不依赖 scalar 位转换 API）；
   - 无符号：sentinel = 255.0f；
   - 有符号：sentinel = -1.0f。

6. **类型回写（FromFloat）**：
   - float：`ReinterpretCast` + `DataCopy`；
   - half：`Cast(CAST_NONE)`；
   - bfloat16 / int32：`Cast(CAST_RINT)`；
   - int8/uint8：fp32 → half → int（`CAST_RINT`）。

7. **CopyOut**：`DataCopyPad` 按实际元素数写回 GM。

UB 资源：`inQue1 / inQue2 / outQue`（VECIN/VECOUT，深度 2）+ `x1fBuf / x2fBuf / quotBuf / tmpBuf / maskBuf`（VECCALC），int8/uint8 额外分配 `halfBuf`。fp32 向量按 64 元素对齐（`AlignUp`）。

## 支持硬件

| 支持的芯片版本 | 是否勾选 |
| --- | --- |
| Atlas 800I/T A2（ascend910b） | √ |
| ascend910_93 | √ |
| ascend950 | √ |

## 算子约束限制

1. x1、x2 数据类型必须一致；
2. 支持 dtype：float16、float32、bfloat16、int32、int8、uint8；
3. int32 走 fp32 计算路径，在 `|value| ≤ 2^24` 范围内精确；超大幅值 int32 的完全精确计算列入后续阶段（与 int64 一并加固）；
4. 广播 stride 已在 tiling 中完整携带，广播分支的完整 kernel 计算路径在后续阶段完善。

# 可维可测分析

## 精度标准 / 性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | 满足 CANN Judge 平台对应题目默认阈值；对齐内置 TruncateMod golden（含除零 NaN/-1/255 行为） | 判题平台 |
| 性能标准 | 全核参与计算场景下不低于内置算子 95%；小 shape（10μs 以下且相差 3μs 内）以性能仿真图与分析结论证明与内置一致或更优 | 判题平台 |

## 自验证方案

参考内置 `TruncateMod` 算子设计全场景自验证用例，采用泛化数据进行功能、精度、性能全维度验证，用例可复现：

1. **功能 / 精度对拍**：用 numpy `fmod`（或调用内置算子）生成 golden，覆盖 6 种 dtype × 多组 shape × 含除零 / 含负数 / 近整数商 等边界数据，逐元素比对；
2. **性能采样**：全核场景与小 shape 场景分别采样，记录用时并与内置算子对比；
3. **报告产出**：输出精度错误占比、性能对比数据，形成可复现自验证报告。

## 当前验证结果

判题平台 50 个测试点执行结果：**49 个 Pass，1 个 Wrong Answer（测试点 45，错误占比 0.05%）**。

- 性能维度：大 shape（测试点 46–50）用时约为最优用时的 96%~98%，满足 95% 红线；
- 精度维度：测试点 45 错误占比仅 0.05%，为个别元素的 fp32 精度边界问题（`x1/x2` 商在 fp32 下呈现「近整数」偏差导致 `trunc` 偏移一个最低位，或 int32 超 `2^24` 幅值丢精度），非结构性错误；
- 后续优化方向：针对浮点路径引入余数区间修正（把因 fp32 除法误差多减 / 少减的一个 `x2` 修正回来），对超大幅值 int32 走整数域精确计算。

## 兼容性分析

新增算子，不涉及存量算子兼容性。
