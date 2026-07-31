# 需求背景（required）

## 需求来源

基于昇腾CANN训练营2026暑期季——西安交通大学专场任务书，参考昇腾版本内置TruncateMod算子的TBE实现，在昇腾NPU上基于Ascend C编程语言实现功能一致的算子。

## 背景介绍

### TruncateMod算子实现优化

基于TruncateMod算子历史TBE版本使用Ascend C编程语言进行优化。

TruncateMod算子（TBE）实现路径和相关API路径：

TruncateMod算子实现路径为：`/usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe/impl/dynamic/`

TruncateMod算子实现中的API路径：`/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/tbe/dsl`

### TruncateMod算子TBE实现现状分析

通过对TruncateMod算子TBE版本的功能分析，当前支持的能力如下：

| 参数 | 参数含义 | 数据类型 | 约束 | 形状 |
| --- | --- | --- | --- | --- |
| x1 | 输入tensor | float16, float32, bfloat16 | 无 | (N,...) |
| x2 | 输入tensor | float16, float32, bfloat16 | 无 | (N,...) |
| y | 输出tensor | float16, float32, bfloat16 | 无 | (N,...) |

计算公式：`y = x1 - trunc(x1 / x2) * x2`

其中 `trunc` 表示向零取整（截断）。

### TruncateMod算子功能分析

TruncateMod算子功能：`y = x1 - trunc(x1 / x2) * x2`

输入：x1、x2

输出：y

支持数据类型：float16、float32、bfloat16

支持广播：支持（通过broadcast_shapes实现NumPy风格广播）

TBE实现中，在Ascend910B平台上使用 `cast_to(float32 → int32 → float32)` 实现截断操作；在Ascend950（v200）平台上使用原生 `tbe.dsl.trunc` API实现。

# 需求分析（required）

## 需求描述

使用Ascend C编程语言实现TruncateMod算子，支持float16、float32、bfloat16数据类型，支持广播功能，功能与精度对齐TBE内置算子。

## 需求拆解

1. 支持float16、float32、bfloat16数据类型
2. 支持NumPy风格广播功能
3. 精度满足CANN Judge平台对应题目精度默认阈值
4. 所有核参与计算场景下，性能不低于原算子的95%

# 详细设计（required）

## 算子分析

### 数学公式

$$y = x_1 - \text{trunc}\left(\frac{x_1}{x_2}\right) \cdot x_2$$

其中截断函数 $\text{trunc}(q)$ 定义为向零取整：

$$\text{trunc}(q) = \begin{cases} \lfloor q \rfloor & q \geq 0 \\ \lceil q \rceil & q < 0 \end{cases}$$

### 支持数据类型

float16、float32、bfloat16

### 支持形状

支持NumPy风格广播（Broadcast），两个输入可以具有不同的形状，输出形状为广播后的结果形状。

## 算子实现

### 实现方案

#### host侧设计

**Tiling策略：**

在host侧将数据视为一维向量进行tiling切分。通过输出shape获取总元素数totalNum，结合硬件UB大小和核心数确定分块策略。

**数据分块策略：**

1. **Block切分**：根据 `MIN_SPLIT_THRESHOLD`（默认2048）将总元素数分配到各核心。每个核心处理的元素数不超过阈值，优先使用满核。

2. **UB Tile切分**：每个核心内进一步将数据切分为UB Tile。ubFactor计算公式基于实际UB缓冲区占用：
   ```
   per_element_UB = 6 × sizeof(T) + 38 + safety_margin
   ubFactor = (ubSize - 2048) / per_element_UB
   ```
   其中6×sizeof(T)为3个双缓冲队列（inputQueueX, inputQueueY, outputQueue），38字节为9个float TBuf + 2个uint8 TBuf，safety_margin为82字节（含对齐和临时空间）。

3. **广播参数传递**：host侧通过TilingData将x1Total、x2Total、x1LastDim、x2LastDim、outLastDim等广播参数传递给kernel侧。

**TilingData结构体：**

| 字段 | 含义 |
| --- | --- |
| totalNum | 输出总元素数 |
| blockFactor | 每个核处理的元素数 |
| ubFactor | 每次UB循环处理的元素数 |
| x1Total | x1总元素数 |
| x2Total | x2总元素数 |
| x1LastDim | x1最后一维大小 |
| x2LastDim | x2最后一维大小 |
| outLastDim | 输出最后一维大小 |

**TilingKey规划：**

根据输入数据类型设置不同的TilingKey，分发到不同的kernel模板实例：
- `TRUNCATEMOD_TPL_SCH_MODE_FP16 (0)`：float16
- `TRUNCATEMOD_TPL_SCH_MODE_FP32 (1)`：float32
- `TRUNCATEMOD_TPL_SCH_MODE_BF16 (2)`：bfloat16

#### kernel侧设计

kernel侧实现Init和Process两个阶段，Process包括数据搬入（CopyIn）、计算（Compute）、搬出（CopyOut）三个阶段，使用双缓冲流水线实现IO与计算的重叠。

**整体流程：**

1. **Init阶段**：接收GM地址和TilingData，初始化GlobalTensor和UB Buffer，计算tileNum和lastLen。
2. **Process阶段**：流水线调度——CopyIn(tile 0) → 循环{Compute(tile i), CopyOut(tile i), CopyIn(tile i+1)} → Compute(last), CopyOut(last)。

**Compute设计：**

```
1. 输入类型转换：
   - fp32: ReinterpretCast（无损，直接复用）
   - fp16/bf16: Cast到fp32（CAST_NONE，无损提升精度）

2. 截断计算（分类型策略）：
   - bfloat16: Cast(float32→int32, CAST_TRUNC) → Cast(int32→float32, CAST_RINT)
     与TBE 910B的cast_to(int32)路径对齐，硬件直接截断
   - float16/float32: Floor + Muls + Compare + Select
     等效实现 trunc(q) = q≥0 ? floor(q) : ceil(q)

3. 余数计算：
   tq = trunc(x1 / x2)
   y  = x1 - tq × x2

4. 输出类型转换：
   - fp32: Add(y, r, 0) 保持float32输出
   - bfloat16: Cast(float32→bfloat16, CAST_RINT)
   - float16: Cast(float32→float16, CAST_TRUNC)
```

**UB缓冲区设计：**

| 缓冲区 | 类型 | 大小(每元素) | 用途 |
| --- | --- | --- | --- |
| inputQueueX | TQue<T>(×2) | 2×sizeof(T) | x1数据双缓冲 |
| inputQueueY | TQue<T>(×2) | 2×sizeof(T) | x2数据双缓冲 |
| outputQueue | TQue<T>(×2) | 2×sizeof(T) | y数据双缓冲 |
| x1F32Buf | TBuf<float> | 4 | x1的float32临时存储 |
| x2F32Buf | TBuf<float> | 4 | x2的float32临时存储 |
| floorBuf | TBuf<float> | 4 | floor(q)结果 |
| ceilBuf | TBuf<float> | 4 | ceil(q)结果/bf16的int32临时 |
| negBuf | TBuf<float> | 4 | -q中间值 |
| zeroBuf | TBuf<float> | 4 | 零向量（Compare基准+Add基准） |
| tqBuf | TBuf<float> | 4 | 截断后商 |
| prodBuf | TBuf<float> | 4 | 乘积tq×x2 |
| remBuf | TBuf<float> | 4 | 余数结果 |
| scratchBuf | TBuf<uint8_t> | 1 | Floor操作临时空间 |
| maskBuf | TBuf<uint8_t> | 1 | Compare结果掩码 |

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas A2 训练系列（Ascend910B） | √ |
| Atlas A3 系列（Ascend950） | √ |

## 算子约束限制

1. x1和x2的数据类型必须一致
2. x2中不能包含零值（避免除零异常）
3. 支持广播，广播规则遵循NumPy广播语义

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
| --- | --- | --- |
| 精度标准 | 满足CANN Judge平台对应题目精度默认阈值（fp16: MARE<2⁻¹⁰, bf16: MARE<2⁻⁷, fp32: MARE<2⁻¹³） | CANN Judge平台 |
| 性能标准 | 所有核参与计算场景下，性能不低于原算子的95% | 任务书要求 |

## 兼容性分析

新算子，基于TBE内置算子参考实现，在Ascend C框架下重新开发，不涉及历史版本兼容性。
