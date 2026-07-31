# TruncateMod算子Ascend C实现设计文档

# 需求背景（required）

## 需求来源
CANN训练营2026暑期季 — 西安交通大学专场算子开发任务。
## 背景介绍

### TruncateMod算子实现优化

基于TruncateMod算子历史TBE版本使用Ascend C编程语言进行优化。

TruncateMod算子实现中的API路径：/usr/local/Ascend/ascend-toolkit/latest/python/site-packages/tbe/dsl

TruncateMod算子（TBE）实现路径：/usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe/impl/ops_legacy/（dynamic模式，support_fusion=True, support_bfp16=True）

TruncateMod算子原型路径：/usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_graph/inc/elewise_calculation_ops.h（第2490-2499行）

### TruncateMod算子TBE实现现状分析

通过对TruncateMod算子TBE版本的功能分析，当前支持的能力如下：

| 参数 | 参数含义 | 参数类型 | 支持数据类型 | 约束 | 形状 |
| --- | --- | --- | --- | --- | --- |
| x1 | 被除数输入tensor | tensor | bfloat16, float16, float32, int8, uint8, int32 | 无 | (D1,D2,...,Dn)，D1*D2*...*Dn ≤ 1000000，n ≤ 8 |
| x2 | 除数输入tensor | tensor | bfloat16, float16, float32, int8, uint8, int32 | 除数不能为0 | (D1,D2,...,Dn)，D1*D2*...*Dn ≤ 1000000，n ≤ 8 |
| y | 输出tensor | tensor | bfloat16, float16, float32, int8, uint8, int32 | 无 | (D1,D2,...,Dn)，D1*D2*...*Dn ≤ 1000000，n ≤ 8 |

> **说明**：TBE实现中check_list仅包含6种数据类型（bfloat16, float16, float32, int8, uint8, int32），但算子原型定义支持8种（额外包含float64、int64）。Ascend C实现优先支持TBE实际支持的6种类型，如环境允许可扩展支持float64和int64。

计算公式：y = x1 - trunc(x1 / x2) * x2

其中 `trunc` 表示向零方向取整（fix），即舍弃小数部分。TruncateMod 运算结果符号与被除数 x1 一致，区别于 FloorMod（结果符号与除数 x2 一致）。

**约束说明：**
- 输入数据总元素数不超过 1,000,000
- 输入数据维度不超过 8 维
- 由于不同架构差异，NPU与CPU计算结果可能存在不一致

### TruncateMod算子功能分析

TruncateMod算子功能：y = x1 - trunc(x1 / x2) * x2

输入：x1（被除数）、x2（除数）

输出：y（取模结果）

支持数据类型：bfloat16、float16、float32、int8、uint8、int32（TBE实际支持）；原型定义额外支持float64、int64

支持数据格式：ND（以实际TBE版本op_info配置为准）

支持广播：支持

---

# 需求分析（required）

## 需求描述

使用Ascend C编程语言实现TruncateMod算子，支持原TBE版本对应的所有数据类型（bfloat16、float16、float32、int8、uint8、int32），支持广播功能，性能不低于TBE版本。

## 需求拆解

1. 支持6种数据类型：bfloat16、float16、float32、int8、uint8、int32（原型定义额外支持float64、int64，如环境允许可扩展）
2. 支持原TBE版本对应的所有数据格式（如ND等）
3. 支持广播功能
4. 性能不低于TBE版本（所有核参与计算场景下性能不低于原算子95%；小shape 10us以下场景允许相差3us，但需提供性能仿真图和分析结论）
5. 支持泛化输入（任意合法shape，元素总数≤1000000，维度≤8）

---

# 详细设计（required）

## 算子分析

### 数学公式

TruncateMod 为向零截断取模运算，数学表达式为：

```
y = x1 - trunc(x1 / x2) * x2
```

其中：
- `trunc(z)` 表示向零方向取整，即 `fix(z)`。例如：trunc(1.7)=1，trunc(-1.7)=-1。
- 结果 `y` 的符号与被除数 `x1` 一致。
- 当 `x1` 与 `x2` 同号时，TruncateMod 与 FloorMod 结果相同；当异号时，两者结果不同。

**示例：**

| x1 | x2 | x1 / x2 | trunc(x1 / x2) | y = x1 - trunc(x1/x2)*x2 |
|----|----|---------|----------------|--------------------------|
| 7 | 3 | 2.333 | 2 | 7 - 2*3 = 1 |
| -7 | 3 | -2.333 | -2 | -7 - (-2)*3 = -1 |
| 7 | -3 | -2.333 | -2 | 7 - (-2)*(-3) = 1 |
| -7 | -3 | 2.333 | 2 | -7 - 2*(-3) = -1 |
| 5 | 2 | 2.5 | 2 | 5 - 2*2 = 1 |
| -5 | 2 | -2.5 | -2 | -5 - (-2)*2 = -1 |

### 支持数据类型

bfloat16、float16、float32、int8、uint8、int32（TBE实际支持）

### 支持形状

- 支持任意维度tensor，维度数 n ≤ 8
- 总元素数 D1*D2*...*Dn ≤ 1,000,000
- 支持广播

---

## 算子实现

### 实现方案

#### 3.2.1 host侧设计：

**tiling策略：**

当不需要广播的情况下，算子计算过程不涉及数据的维度信息，故在host侧将数据视为一维向量，仅考虑数据个数，不考虑数据维度信息。

在广播的情况下，在host侧获取x1、x2等输入的相应shape大小以及各自的维度dim信息，首先，通过函数实现输入的shape维度补全和统一，遍历所有输入对应的维度大小得到其中最大的维度数，维度缺失的输入在首部维度补1实现维度扩充。再根据对应的广播规则，遍历各输入对应的shape，纵向比较求得最大的维度数以此得到最终的广播形状，根据最终广播形状通过指定接口获得total_length。需要将host侧获取的输入形状、最终广播形状以及total_length变量传到kernel侧。

任务均分：coreNum 根据输入长度和块大小动态调整，确保每个核心处理的数据块数均匀。

批量搬运：tileBlockNum 和 tileDataNum 计算单次搬运的数据量，通过 finalSmallTileNum 和 finalBigTileNum 确定小核/大核的搬运次数，将多次搬运合并为批量操作，减少冗余开销。尾块的处理逻辑确保不完整块也能被合并到计算流程中，避免数据碎片。


##### 1. 分核策略：

优先使用满核的原则。

如果核间能均分，可视作无大小核区分，大核小核数据块一致；

如果核间不能均分，需要将余出的数据块分配到前几个核上。

输入数据大小计算：通过GetInputShape和GetDataTypeLength函数获取输入数据的大小和类型长度，计算出输入数据的总字节数。

UB内存大小和核心数量获取：通过平台信息获取UB内存大小和核心数量，并根据这些信息调整核心数量。

##### 2. 数据分块和内存优化策略：
充分使用UB空间的原则。

需要考虑不同硬件的UB大小不同、是否开启double buffer、kernel侧API实现过程中是否需要临时数据的储存，综合考虑单核内切分的大小。

UB内存大小获取：通过GetCoreMemSize函数获取UB内存的大小，用于后续的数据切分计算。

Tile块计算：根据UB内存大小和预定义的BLOCK_SIZE及BUFFER_NUM，计算出每个Tile块的数据数量。

数据切分：将输入数据按照计算出的Tile块大小进行切分，计算出每个core需要处理的数据块数量和最后一个block的剩余数据量。

设置切分参数：将计算出的切分参数（如每个core的数据量、Tile块大小等）设置到TruncateModTilingData对象中。

这些策略确保了数据在多个核心之间的均匀分布，并且在单个核心内进行了合理的切分，以提高并行处理的效率。

##### 3. tilingkey规划策略：

需要tilingkey的情况：需要感知host侧信息对kernel侧走不同分支。在host侧获取输入的数据类型和是否涉及广播，通过不同tilingkey区分不同执行分支：

| tilingkey | 场景 | 计算路径 |
|-----------|------|---------|
| 0 | 不广播，整数类型（int8/uint8/int32） | 直接整数取余运算（C++ `%` 运算符，向零截断） |
| 1 | 不广播，浮点类型（bfloat16/float16/float32） | 转float32 → 除法 → trunc → 乘减 → 转回原类型 |
| 2 | 广播，整数类型（int8/uint8/int32） | 广播索引映射 + 整数取余运算 |
| 3 | 广播，浮点类型（bfloat16/float16/float32） | 广播索引映射 + 浮点trunc计算 |

数据检测：
- 检测除数x2是否为0。TBE实现中未显式处理除零，由底层指令行为决定（浮点类型产生Inf/NaN，整数类型行为未定义）。Ascend C实现中对于浮点类型保持与TBE一致的行为；对于整数类型，需确保除零不会导致核异常。

#### 3.2.2 kernel侧设计：

进行Init和Process两个阶段，其中Process包括数据搬入（CopyIn）、计算（Compute）、搬出（CopyOut）三个阶段。

1. **数据类型处理策略（参考TBE实现）：**

   TBE实现中，所有输入类型（包括整数类型）均先cast到float32进行计算，计算完成后再cast回原始类型。这种策略的优点是实现统一、代码简洁；缺点是整数类型大数场景下可能丢失精度（float32有效尾数24位，精确表示范围约±16,777,216）。

   Ascend C实现采用更精细的策略：
   - **整数类型（int8/uint8/int32）**：直接使用整数运算。C/C++11及以后标准中，整数除法向零截断，因此可直接使用 `%` 运算符实现TruncateMod：
     ```cpp
     y = x1 % x2;  // C++整数取余，向零截断
     ```
     该策略精度无损，性能优于浮点转换方案。
   - **浮点类型（bfloat16/float16/float32）**：参考TBE策略，在float32域计算：
     ```cpp
     // 伪代码
     float32 x1_f32 = Cast(x1, float32);
     float32 x2_f32 = Cast(x2, float32);
     float32 quot = x1_f32 / x2_f32;
     float32 quot_trunc = Fix(quot);  // Ascend C向零取整指令
     float32 result_f32 = x1_f32 - quot_trunc * x2_f32;
     y = Cast(result_f32, original_dtype);
     ```
     对于bfloat16和float16，上述cast和计算在Ascend C中可通过Vector指令高效完成。

2. **广播处理：**
   - 在kernel侧的copyin阶段完成广播数据的填充，首先获取host侧存储各个输入shape的一维数组以及按照广播规则获得的最终广播shape，通过特定的函数完成指定数组元素的地址映射。
   - 对于广播场景，每个核根据自身的block_idx和tiling信息，计算出当前核需要处理的输出元素在最终广播shape中的坐标，再反向映射回原始输入tensor的坐标，从GM中读取对应元素。

3. **分支执行：**
   - 根据不同的tilingkey执行不同的核函数，区分是否广播以及不同数据类型的计算路径。
   - 对于不广播场景，可直接按一维向量处理，简化索引计算。
   - 对于广播场景，需要额外的坐标映射计算。

4. **除零保护：**
   - TBE实现中未显式处理除零，依赖底层指令行为。Ascend C实现中：
     - 浮点类型：除零产生Inf/NaN，与TBE行为保持一致。
     - 整数类型：需避免除零导致核异常，可在计算前检测x2是否为0，若为0则输出0或保持x1（具体行为需与TBE实现对齐，建议通过测试确认）。

5. **Ascend C的TruncateMod算子流程：**
   - **CopyIn**：将x1、x2数据从GM搬运至UB
   - **Compute**：按元素执行TruncateMod计算
     - 整数分支：`y = x1 % x2`
     - 浮点分支：`y = x1 - Fix(x1 / x2) * x2`
   - **CopyOut**：将结果从UB搬运至GM

---

## 支持硬件

| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Atlas 800I/T A2 | √ |
| Atlas A3 系列产品 | √ |

> 以任务书要求为准，适配 Atlas A2 训练系列产品 / Atlas A3 系列产品。

## 算子约束限制

1. 除数 x2 不能为0，若输入包含0需按与TBE版本一致的方式处理。
2. 输入x1和x2的数据类型必须一致。
3. 输入数据总元素数不超过 1,000,000。
4. 输入数据维度不超过 8 维。
5. 由于不同架构差异，NPU与CPU计算结果可能存在不一致。
6. TBE实现仅支持6种数据类型（bfloat16、float16、float32、int8、uint8、int32），原型定义额外支持float64和int64，Ascend C实现优先保证6种类型的完整支持。
7. 其他约束以原TBE版本op_info配置为准。

---

# 可维可测分析

## 精度标准/性能标准

| 验收标准 | 描述(不涉及说明原因) | 标准来源 |
| --- | --- | --- |
| 精度标准 | 不低于TBE版本，满足CANN Judge平台对应题目精度默认阈值 | 任务书要求 |
| 性能标准 | 不低于TBE版本（所有核参与计算场景下性能不低于原算子95%；小shape 10us以下场景允许相差3us，但需提供性能仿真图和分析结论） | 任务书要求 |

## 兼容性分析

新算子，不涉及兼容性分析

---
