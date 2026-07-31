# SoftshrinkGrad 算子设计文档

## 需求背景（required）

### 需求来源

CANN 训练营 2026 暑期季西安交通大学专场社区任务：
[SoftshrinkGrad 算子开发任务书](https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/docs/202607/SoftshrinkGrad_task_doc.md)。

任务要求参考 CANN 内置 `aclnnSoftshrinkBackward` 的 TBE 实现，在昇腾 NPU 上基于 Ascend C 实现功能一致的 SoftshrinkGrad 算子，并完成设计、开发、测试与验收流程。算子最终目标合入 `ops-nn` 仓 `experimental/activation` 目录。

### 背景介绍

Softshrink 是常用激活函数，按阈值 `lambd` 对输入进行软收缩：

```text
y = x - lambd,  x > lambd
y = x + lambd,  x < -lambd
y = 0,          otherwise
```

SoftshrinkGrad 是 Softshrink 反向梯度算子。由于 Softshrink 在区间 `[-lambd, lambd]` 内输出为 0，在该区间外对输入的一阶导数为 1，因此反向梯度可表达为：

```text
dx = dy, x > lambd or x < -lambd
dx = 0,  otherwise
```

其中 `dy` 为上游梯度，`x` 为 Softshrink 前向输入，`dx` 为对前向输入的梯度。

### SoftshrinkGrad 算子 TBE 实现现状分析

根据任务书要求，TBE 参考实现路径如下：

- kernel 实现：`/usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe/impl/dynamic/`
- 算子原型：`/usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_proto/inc/`
- 算子信息库：`/usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe/config/ascend910b`

SoftshrinkGrad 的核心能力分析如下：

| 参数 | 参数含义 | 参数类型 | 支持数据类型 | 约束 | 形状 |
| --- | --- | --- | --- | --- | --- |
| grad_output | 上游梯度 tensor | tensor | float16, float32, bfloat16 | 与 self 类型一致 | ND，动态 shape |
| self | Softshrink 前向输入 tensor | tensor | float16, float32, bfloat16 | 与 grad_output 类型一致 | 与 grad_output 相同 |
| lambd | 阈值属性 | float attr | float | 默认 0.5，建议取非负值 | 标量属性 |
| output | 输入梯度 tensor | tensor | float16, float32, bfloat16 | 与输入类型一致 | 与输入相同 |

计算公式：

```text
output_i = grad_output_i, self_i > lambd or self_i < -lambd
output_i = 0,             -lambd <= self_i <= lambd
```

### SoftshrinkGrad 算子功能分析

SoftshrinkGrad 属于逐元素二输入一输出算子。计算过程不改变 shape，不涉及 reduce、axis 或复杂广播。设计重点如下：

1. 对齐内置 TBE backward 行为，比较逻辑采用数值逻辑比较，不依赖二进制结果比较。
2. 支持 float16、float32、bfloat16 三类浮点数据类型。
3. 支持 ND 格式、动态 shape、动态 rank。
4. 对输入 tensor 做同 shape、同 dtype 校验。
5. 通过分核与 UB tiling 保证泛化 shape 下的并行度和搬运效率。

## 需求分析（required）

### 需求描述

使用 Ascend C 实现 SoftshrinkGrad 算子，功能与 CANN 内置 TBE 版本 `aclnnSoftshrinkBackward` 保持一致。算子输入为上游梯度 `grad_output` 与前向输入 `self`，属性为阈值 `lambd`，输出为前向输入梯度 `output`。

### 需求拆解

1. 支持 `float16`、`float32`、`bfloat16` 数据类型。
2. 支持 ND 数据格式、动态 shape、动态 rank。
3. 支持空 tensor 场景，输出 shape 与输入保持一致。
4. 校验 `grad_output` 与 `self` 的 dtype、shape 一致。
5. 实现 host 侧 shape 推导、tiling 计算、tiling key 选择。
6. 实现 kernel 侧数据搬入、阈值比较、Select 写回。
7. 所有核参与计算场景性能不低于 TBE 版本 95%。
8. 精度满足 CANN Judge 平台默认阈值。

## 详细设计（required）

### 算子分析

#### 数学公式

Softshrink 前向：

```text
softshrink(x, lambd) =
  x - lambd, x > lambd
  x + lambd, x < -lambd
  0,         otherwise
```

SoftshrinkGrad 反向：

```text
output_i = grad_output_i * mask_i
mask_i = 1, self_i > lambd or self_i < -lambd
mask_i = 0, otherwise
```

等价实现：

```text
output_i = Select(self_i > lambd || self_i < -lambd, grad_output_i, 0)
```

边界点处理：

- `self_i == lambd` 时输出 0。
- `self_i == -lambd` 时输出 0。
- 比较条件必须使用严格大于和严格小于。

#### 支持数据类型

| 数据类型 | 计算策略 |
| --- | --- |
| float32 | 直接比较并 Select 输出 |
| float16 | 可直接计算；若目标架构比较指令存在精度或布尔选择限制，则 cast 到 float32 完成比较后 cast 回 float16 |
| bfloat16 | 建议 cast 到 float32 完成比较与选择，再 cast 回 bfloat16 |

#### 支持形状

- 支持 ND 格式。
- 支持动态 shape 与动态 rank。
- `grad_output`、`self`、`output` shape 完全相同。
- 不设计广播能力；若后续 TBE 明确支持广播，再扩展 host 侧 broadcast shape 推导。

### 算子实现

#### host 侧设计

host 侧负责参数校验、shape 推导、tiling 参数计算和 kernel 分支选择。

##### 参数校验

1. `grad_output`、`self`、`output` 不能为空。
2. `grad_output` 与 `self` 的 dtype 必须一致。
3. `output` dtype 必须与输入一致。
4. `grad_output` 与 `self` 的 shape 必须一致。
5. `output` shape 由输入 shape 推导得到，与输入 shape 相同。
6. `lambd` 使用 float 属性，默认值为 0.5。建议限制为非负；若 TBE 对负值有兼容行为，则以 TBE 行为为准。

##### shape 推导

SoftshrinkGrad 不改变 shape：

```text
output.shape = grad_output.shape = self.shape
```

总元素数：

```text
totalLength = NumElements(grad_output)
```

当 `totalLength == 0` 时，host 侧仍完成输出 shape 推导，kernel 侧可直接返回。

##### tiling 策略

设计目标：

- 尽量使用全部可用 AICore。
- 单核内充分利用 UB。
- 保证尾块处理正确。
- 对小 shape 减少无效分核和搬运开销。

基础参数：

```text
dataSize       = sizeof(dtype)
blockBytes     = 32
blockElements  = blockBytes / dataSize
ubSize         = platform.GetCoreMemSize()
coreNumLimit   = platform.GetCoreNumAic()
bufferNum      = 1 or 2
```

UB 空间规划：

- `grad_output` 输入队列。
- `self` 输入队列。
- `output` 输出队列。
- `mask` 或临时比较结果。
- float16/bfloat16 cast 场景需要额外 float32 临时空间。

单次 tile 元素数按 UB 可用空间向下对齐到 `blockElements`：

```text
tileLength = AlignDown(usableUbBytes / bytesPerElementGroup, blockElements)
minElementsPerCore = max(tileLength, blockElements)
```

`minElementsPerCore` 表示每个有效核至少处理一个 tile，并保证最小处理长度满足 32B 搬运对齐要求。

分核策略：

```text
usedCoreNum = min(coreNumLimit, CeilDiv(totalLength, minElementsPerCore))
blockLength = CeilDiv(totalLength, usedCoreNum)
```

当 `totalLength` 较小、不足以让所有核有效工作时，减少 `usedCoreNum`，避免调度开销超过计算收益。

每个 core 的数据区间：

```text
start = coreId * blockLength
end   = min(start + blockLength, totalLength)
```

单核内部循环：

```text
for offset in [start, end) step tileLength:
    curLength = min(tileLength, end - offset)
```

##### tiling key 规划

tiling key 需要区分 dtype 和 buffer 模式：

| tiling key 维度 | 取值 | 说明 |
| --- | --- | --- |
| dtype | fp32 / fp16 / bf16 | 区分计算路径 |
| buffer 模式 | single / double | 小 shape 可使用 single buffer，大 shape 使用 double buffer |
| cast 模式 | direct / cast_fp32 | fp16/bf16 可走 cast_fp32 路径 |

建议规划：

```text
key = Encode(dtype, bufferMode, computeMode)
```

首版可优先实现：

1. fp32 direct single/double buffer。
2. fp16 cast_fp32 single/double buffer。
3. bf16 cast_fp32 single/double buffer。

#### kernel 侧设计

kernel 侧分为 `Init` 和 `Process` 两阶段。`Process` 包含 `CopyIn`、`Compute`、`CopyOut`。

##### Init

`Init` 根据 tiling data 初始化：

- GM 输入输出地址。
- 当前 core 的起始 offset。
- 当前 core 处理长度。
- tile 长度。
- `lambd` 和 `-lambd` 常量。
- 输入、输出、临时队列。

##### CopyIn

从 GM 搬入当前 tile 的 `grad_output` 和 `self`：

```text
DataCopy(gradLocal, gradGm[offset], curLength)
DataCopy(selfLocal, selfGm[offset], curLength)
```

尾块按 `curLength` 处理。若 DataCopy 需要 32B 对齐，host 侧 tiling 与 kernel 侧 CopyOut 需要共同处理尾块 pad 或按支持非对齐尾块的接口实现。

##### Compute

fp32 direct 路径：

```text
maskUpper = selfLocal > lambd
maskLower = selfLocal < -lambd
mask = maskUpper || maskLower
zero = Duplicate(0)
outLocal = Select(mask, gradLocal, zero)
```

fp16/bf16 cast_fp32 路径：

```text
selfFp32 = Cast(selfLocal, fp32)
gradFp32 = Cast(gradLocal, fp32)
maskUpper = selfFp32 > lambd
maskLower = selfFp32 < -lambd
mask = maskUpper || maskLower
outFp32 = Select(mask, gradFp32, 0.0)
outLocal = Cast(outFp32, dtype)
```

如果 Ascend C 当前架构支持 fp16/bf16 原生比较并满足精度要求，可将 fp16 direct 作为性能优化分支；bf16 保持 fp32 比较更稳妥。

##### CopyOut

将当前 tile 的结果搬回 GM：

```text
DataCopy(outputGm[offset], outLocal, curLength)
```

##### 异常和边界处理

1. `totalLength == 0` 时不做 GM 访问。
2. `lambd == 0` 时公式退化为 `self != 0 ? grad_output : 0`，仍使用严格比较：`self > 0 || self < 0`。
3. `NaN` 输入按比较结果处理。由于 `NaN > lambd` 和 `NaN < -lambd` 均为 false，输出 0；若 TBE 存在特殊 NaN 行为，最终以 TBE 对齐结果为准。
4. `Inf` 输入按普通比较处理。

### 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas A2 训练系列产品 | √ |
| Atlas A3 系列产品 | √ |

### 算子约束限制

1. `grad_output` 和 `self` 必须同 dtype、同 shape。
2. 当前设计不支持广播。
3. 当前设计仅支持 ND 格式。
4. `lambd` 为 float 属性，默认 0.5。
5. 输出 dtype 与输入 dtype 一致。

## 可维可测分析

### 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 功能标准 | 与 CANN 内置 `aclnnSoftshrinkBackward` TBE 实现功能一致 | 任务书 |
| 精度标准 | 满足 CANN Judge 平台对应题目默认精度阈值 | 任务书 |
| 性能标准 | 所有核参与计算场景性能不低于 TBE 版本 95% | 任务书 |
| 泛化标准 | 支持合法 dtype、动态 shape、边界值和随机输入场景 | 任务书 |

### 自验证用例设计

| 用例类型 | 覆盖内容 |
| --- | --- |
| 基础功能 | `self` 全部大于 `lambd`、全部小于 `-lambd`、全部位于闭区间内 |
| 边界值 | `self == lambd`、`self == -lambd`、`self == 0`、`lambd == 0` |
| 混合分布 | 随机输入同时覆盖三段分支 |
| 数据类型 | float16、float32、bfloat16 |
| shape 泛化 | 标量、空 tensor、小 shape、32B 对齐 shape、非 32B 对齐尾块、大 shape、多维 ND |
| 特殊值 | `Inf`、`-Inf`、`NaN`，以 TBE 行为作为对齐标准 |
| 性能 | 小 shape、单核 shape、满核 shape、大 shape |

参考 Python 期望值：

```python
expected = np.where((self > lambd) | (self < -lambd), grad_output, 0)
```

### 性能分析方法

1. 与内置 TBE 版本在相同输入 shape、dtype、运行环境下对比。
2. 记录每个 CANN Judge 测试点耗时。
3. 对满核场景确认 `usedCoreNum` 达到硬件可用核数。
4. 对小 shape 若出现 10us 以下且差距 3us 内的场景，补充性能仿真图和分析结论。
5. 对 fp16/bf16 分支对比 direct 与 cast_fp32 的收益，最终选择精度与性能均满足要求的路径。

### 兼容性分析

SoftshrinkGrad 是新增 Ascend C 实现，用于替代或补齐内置 TBE backward 能力。接口语义、输入输出、属性默认值、shape 推导、边界比较规则均与 TBE 版本保持一致，不改变上层框架调用语义。
