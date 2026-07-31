# SoftshrinkGrad 算子设计方案

**CANN SUMMER CAMPS 2026 · XJTU · 第16组**

## 1. 概述

Softshrink 是一种分段线性激活函数。SoftshrinkGrad 接收上游梯度和
Softshrink 的正向输入，根据输入是否位于阈值区间外决定梯度是否通过。
该算子无归约、无跨元素依赖，适合在 AI Vector 核上并行执行。

## 2. 数学定义

Softshrink 正向函数为：

```text
               x - lambd,  x > lambd
Softshrink(x)= x + lambd,  x < -lambd
               0,          其他
```

反向函数为：

```text
               gradOutput, x > lambd
gradInput(x) = gradOutput, x < -lambd
               0,          其他
```

实现使用等价表达式：

```text
gradInput = Select(abs(self) > lambd, gradOutput, 0)
```

因此 `self == lambd` 和 `self == -lambd` 时输出必须为 0。NaN 的比较结果为
false，输出同样为 0。

## 3. 算子规格

| 参数 | 输入/输出/属性 | 数据类型 | 格式 | 约束 |
|---|---|---|---|---|
| gradOutput | 输入 | FLOAT16/FLOAT/BFLOAT16 | ND | 0～8维 |
| self | 输入 | FLOAT16/FLOAT/BFLOAT16 | ND | 与gradOutput可broadcast |
| lambd | 属性 | FLOAT | - | 默认0.5，lambd≥0 |
| gradInput | 输出 | 同gradOutput | ND | broadcast结果shape |

### 3.1 分层约定

完整 ACLNN 语义允许 `gradOutput` 与 `self` broadcast。接口层负责：

1. 空 Tensor 和空指针检查。
2. 将非连续 Tensor 连续化。
3. 将输入统一到计算 dtype。
4. 对两个输入执行 broadcast。
5. 调用同 shape、同 dtype 的 SoftshrinkGrad 核心算子。
6. 必要时将结果转换回 `gradOutput` dtype。

本项目 `custom_op` 是核心算子工程，因此 Host Tiling 会拒绝元素数不同的输入。
`test/test_reference.py` 覆盖了完整 broadcast 语义。

## 4. 总体架构

```text
ACLNN接口层
  ├─ 参数校验
  ├─ Contiguous / Cast / Broadcast
  └─ SoftshrinkGrad核心算子
       ├─ Host: InferShape + InferDtype + Tiling
       └─ AI Core: CopyIn → Compute → CopyOut
```

## 5. Tiling 设计

### 5.1 多核切分

- 最大使用 40 个 AI Vector 核。
- 期望每核至少处理约 `2048 × 4` 个元素，避免小数据过度切核。
- 每核起始地址按 16 个元素对齐。16 个 FLOAT 为 64 Byte，16 个
  FLOAT16/BFLOAT16 为 32 Byte，满足 GM 搬运对齐要求。
- 最后一个有效核按照 `totalLength - blockStart` 计算真实长度。

### 5.2 UB 切分

Tile 固定为 2048 个元素。该值兼顾：

- FLOAT 三个双缓冲队列的 UB 占用。
- FLOAT16/BFLOAT16 转 FLOAT 所需的三个临时 Buffer。
- Abs 结果和 Compare Mask。
- 选择指令需要的临时空间。

### 5.3 TilingData

| 字段 | 类型 | 含义 |
|---|---|---|
| totalLength | uint64_t | 总元素数 |
| blockLength | uint32_t | 每核逻辑长度 |
| tileLength | uint32_t | 每个Tile元素数 |
| lambd | float | 比较阈值 |

## 6. Kernel 设计

### 6.1 流水线

每个 Tile 执行：

1. `DataCopyPad` 将 `gradOutput`、`self` 从 GM 搬入 UB。
2. FLOAT 直接计算；FLOAT16/BFLOAT16 先 Cast 为 FLOAT。
3. `Abs(self)` 计算绝对值。
4. `CompareScalar(abs(self), lambd, GT)` 生成掩码。
5. `Select(mask, gradOutput, 0)` 生成结果。
6. FLOAT16/BFLOAT16 将 FLOAT 结果转换回原 dtype。
7. 按有效字节数将结果写回 GM。

输入、输出队列深度为 2，用双缓冲重叠搬运和向量计算。

### 6.2 尾块

最后一个 Tile 可能不是 32 Byte 对齐：

- GM→UB 使用扩展搬运参数和 `DataCopyPad`。
- 向量计算长度向上对齐到 256 Byte 对应的 64 个 FLOAT 元素。
- UB Buffer 按完整 Tile 分配，不发生 UB 越界。
- UB→GM 仅复制 `validLength × sizeof(dtype)` 字节，不越界写入。

### 6.3 混合精度

FLOAT 直接比较；FLOAT16 和 BFLOAT16 转换为 FLOAT 后执行
`Abs/Compare/Select`。由于输出只能是原梯度或精确的零，转换不会引入算术累积误差。
BFLOAT16 输出采用 RINT 模式，FLOAT16 输出采用 CAST_NONE。

## 7. Host 设计

Host 侧完成：

- 检查上下文、Shape、TilingData 指针。
- 核心算子检查两个输入元素数相同。
- 检查 `lambd >= 0`。
- 推导输出 Shape 与 dtype。
- 动态选择 BlockDim。
- 注册 `ascend910b` 与 `ascend910_93`。

空 Tensor 的 `totalLength` 为 0，设置一个空操作核，Kernel 在初始化阶段直接返回。

## 8. 正确性分析

对于任意元素 `x=self[i]`：

- `x>lambd`：`abs(x)>lambd` 为真，Select 输出 `gradOutput[i]`。
- `x<-lambd`：`abs(x)>lambd` 为真，Select 输出 `gradOutput[i]`。
- `-lambd≤x≤lambd`：比较为假，Select 输出 0。
- `x=NaN`：比较为假，输出 0。

与数学定义逐项一致。

## 9. 测试方案

| 类别 | 用例 |
|---|---|
| 基础 | 正数、负数、零 |
| 边界 | `±lambd`、阈值内外相邻值 |
| Shape | 标量、1～8维、动态shape |
| 特殊 | 空Tensor、非32B对齐尾块 |
| 数据类型 | FLOAT、FLOAT16、BFLOAT16 |
| 接口 | 同shape、broadcast、非法shape |
| 参数 | 默认lambd、0、负数拒绝 |
| 特殊值 | NaN、+Inf、-Inf |

## 10. 性能分析

算子每个元素读取两个输入并写一个输出，计算仅包含绝对值、比较和选择，属于
访存受限算子。主要优化措施：

- 多核并行。
- GM/UB 双缓冲。
- 连续大块搬运。
- 合并两个分支为一次 `abs + compare + select`。
- FP32 临时 Buffer 复用，避免额外 GM 中间结果。

## 11. 风险与处理

| 风险 | 处理 |
|---|---|
| 边界误用 `>=` | 明确使用 `GT`，`±lambd` 输出0 |
| BF16比较精度 | 转FP32后比较 |
| 尾块越界 | DataCopyPad + 有效字节写回 |
| 负lambd | Host侧拒绝 |
| 广播造成Kernel复杂 | ACLNN层预广播，核心Kernel同shape |
| 空Tensor | Host/Kernel空操作路径 |

## 12. 参考接口

- CANN `aclnnSoftshrinkBackward` 接口语义。
- CANN ops-nn 中 SoftshrinkGrad 的公开算子规格。
- Ascend C `Abs`、`CompareScalar`、`Select`、`DataCopyPad` 接口。
