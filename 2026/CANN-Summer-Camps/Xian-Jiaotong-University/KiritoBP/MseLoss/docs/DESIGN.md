# aclnnMseLoss 设计文档

## 需求背景（required）

### 需求来源

社区任务《[CANN训练营2026暑期季 – 西安交通大学专场-MseLoss算子开发任务书](https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/docs/202607/MseLoss_task_doc.md)》。本次需求聚焦于 `aclnnMseLoss` 算子的 Ascend C 实现，替代原有的 TBE 实现。

### 背景介绍

#### MseLoss 算子实现优化

MseLoss（Mean Squared Error Loss，均方误差损失）是深度学习中最常用的回归损失函数之一。当前昇腾 CANN 内置的 `aclnnMseLoss` 基于 TBE（Tensor Boost Engine）实现，本次任务要求使用 Ascend C 编程语言重新实现该算子，在保证功能完全对齐的前提下，性能不低于原 TBE 版本的 95%。

#### MseLoss 算子（TBE）实现路径和相关 API 路径

- TBE kernel 实现路径：`/usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe/impl/dynamic/`
- 算子原型定义路径：`/usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_proto/inc/`
- 算子信息库路径：`/usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe/config/ascend910b`

#### MseLoss 算子现状分析

通过对 MseLoss 算子 TBE 版本的功能分析，当前支持的能力如下：

| 参数 | 参数含义 | 数据类型 | 支持数据类型 | 约束 | 形状 |
|------|---------|---------|-------------|------|------|
| predict | 输入 tensor（预测值） | tensor | float16, float32 | 与 label 同 shape、同 dtype | (N,…) |
| label | 输入 tensor（标签值） | tensor | float16, float32 | 与 predict 同 shape、同 dtype | (N,…) |
| y | 输出 tensor | tensor | float16, float32 | shape 与 predict 一致 | (N,…) |

Attr 属性：

| 属性名 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| reduction | string | "mean" | 归约方式: "none"（逐元素，shape 同输入）、"mean"（标量均值）、"sum"（标量和） |

计算公式：

- **reduction='none'**：`y[i] = (predict[i] - label[i])^2`
- **reduction='mean'**：`y = mean((predict - label)^2)`
- **reduction='sum'**：`y = sum((predict - label)^2)`

#### MseLoss 算子功能分析

- **算子功能**：计算预测值与标签之间的均方误差
- **输入**：predict（预测值张量）、label（标签张量）
- **输出**：y（损失值张量或标量）
- **支持数据类型**：float16、float32
- **支持广播**：不支持广播，predict 与 label 必须同 shape
- **Kernel 职责**：逐元素计算 `(predict - label)^2`，reduction 由 Host 侧 ACLNN 层完成

---

## 需求分析（required）

### 需求描述

使用 Ascend C 编程语言实现 `aclnnMseLoss` 算子，支持 float16 和 float32 数据类型，ND 数据格式。算子功能与原 TBE 实现完全对齐，比较方式从二进制比较改为逻辑值比较。必须实现算子泛化功能，满足各类合法输入场景。

### 需求拆解

1. **数据类型支持**：predict 和 label 均支持 `float16`（通过 `DT_FLOAT16` / `C_DT_FLOAT16`）和 `float32`（通过 `DT_FLOAT` / `C_DT_FLOAT`）
2. **多核并行**：利用平台全部 AI Core，通过 `SetBlockDim(num_cores_aiv)` 实现满核计算
3. **双缓冲流水线**：BUFFER_NUM=2，CopyIn 与 Compute/CopyOut 阶段流水并行
4. **泛化能力**：支持任意 N-D shape（通过将数据展平为 1D 处理，消除 shape 维度约束）
5. **性能目标**：所有核参与计算场景下，性能不低于原 TBE 算子的 95%

---

## 详细设计（required）

### 算子分析

#### 数学公式

```
输入:
  predict — shape [d0, d1, ..., dn], dtype=float16/float32
  label   — shape [d0, d1, ..., dn], dtype=float16/float32

输出:
  y       — shape [d0, d1, ..., dn], dtype=float16/float32

（Kernel 层输出逐元素平方差，reduction 由 ACLNN 层完成）

逐元素计算:
  diff[i]    = predict[i] - label[i]
  squared[i] = diff[i] * diff[i]
  y[i]       = squared[i]
```

#### 支持数据类型

| 参数 | 数据类型 | TilingKey 宏 | 说明 |
|------|---------|-------------|------|
| predict | float16 | C_DT_FLOAT16 | DT_PREDICT 模板参数 |
| predict | float32 | C_DT_FLOAT  | DT_PREDICT 模板参数 |
| label | (同 predict) | (自动匹配) | OpDef 约束同 dtype |
| y | (同 predict) | (自动匹配) | 输出自动匹配输入 dtype |

#### 支持形状

- 任意 N-D Tensor（`(N,…)`），predict 与 label 形状必须完全一致
- host 侧通过 `GetShapeSize()` 获取总元素数，展平为 1D 传递给 kernel
- kernel 侧按 1D 分块处理，不感知原始维度信息

### 算子实现

#### 实现方案

```
┌─────────────────────────────────────────────────────┐
│                    Host 侧 (Tiling)                   │
│  TilingFunc:                                         │
│    ├─ 获取平台核数 num_cores_aiv                       │
│    ├─ 获取 UB 大小                                     │
│    ├─ 获取输入 totalLength = GetShapeSize()            │
│    ├─ 设置 BlockDim = num_cores_aiv (全核并行)          │
│    ├─ 填充 tiling->length = totalLength                │
│    └─ 配置 DT_PREDICT TilingKey                        │
├─────────────────────────────────────────────────────┤
│                   Device 侧 (Kernel)                  │
│  mse_loss(predict, label, y, workspace, tiling):      │
│    ├─ GET_TILING_DATA → tiling_data.length            │
│    ├─ KernelMseLoss<DT_PREDICT>::Init(..., length)    │
│    │   ├─ blockLength = length / blockNum              │
│    │   ├─ tileLength = blockLength / TILE_NUM_PER_CORE │
│    │   └─ InitBuffer (BUFFER_NUM=2)                   │
│    └─ KernelMseLoss<DT_PREDICT>::Process()            │
│        ├─ CopyIn(0)  预取                              │
│        ├─ for i in 0..N-2:                            │
│        │   ├─ CopyIn(i+1)   GM→UB                     │
│        │   ├─ Compute(i)    Sub→Mul UB                │
│        │   └─ CopyOut(i)    UB→GM                     │
│        ├─ Compute(N-1)                                │
│        └─ CopyOut(N-1)                                │
└─────────────────────────────────────────────────────┘
```

#### Host 侧设计

**Tiling 策略**：MseLoss 不涉及广播，计算过程与维度无关。Host 侧将数据视为 1D 向量，仅传递总元素数 `totalLength` 给 Kernel 侧。

**任务均分**：`BlockDim = num_cores_aiv`（平台全部 AI Core），每核处理 `totalLength / num_cores_aiv` 个元素。若不能整除，余数由前几个核分摊。

**TilingKey 规划**：MseLoss 支持 float16 和 float32 两种 dtype，通过 TilingKey 实现 Kernel 侧模板实例化选择：
```cpp
ASCENDC_TPL_ARGS_DECL(MseLoss,
    ASCENDC_TPL_DATATYPE_DECL(DT_PREDICT, C_DT_FLOAT16, C_DT_FLOAT),
);
```

#### Kernel 侧设计

**Init 阶段**：
- 采用 `basePerCore + remainElem` 模式精确分配各核数据量，确保余数由前几个核分摊
- `tileBaseLen = coreDataLen / TILE_NUM`，`tileRemainLen = coreDataLen % TILE_NUM`
- 末 tile 自动处理余数，无尾部丢失
- `coreDataLen == 0` 的空核直接跳过 Init/Process

**Process 阶段（简单顺序循环）**：
参照 Lesson 1 的 `mul_custom` 模式，采用 for 循环内顺序执行 `CopyIn → Compute → CopyOut`：
```cpp
for (int32_t i = 0; i < TILE_NUM; ++i) {
    CopyIn(i);   // DataCopy predict/label GM→UB
    Compute(i);  // Sub → Mul (diff = pred-label, sqr = diff*diff)
    CopyOut(i);  // DataCopy result UB→GM
}
```
- `QuePosition::VECIN/VECOUT` 队列类型（CANN 8.5 兼容）
- `DT_PREDICT` 通过预处理器 `-D` 注入 + `#ifndef` fallback
- 使用独立临时 tensor 避免 `Mul` 的 src/dst 重叠

**数据流**：
```
Global Memory (predict, label) ──CopyIn(DataCopy)──► Unified Buffer (xLocal, lLocal)
                                                                  │
                                                     Sub(zLocal, xLocal, lLocal)
                                                     Mul(zLocal, zLocal, zLocal)
                                                                  │
Global Memory (y) ◄──CopyOut(DataCopy)── Unified Buffer (zLocal)
```

### 数据检测

| 检测项 | 检测位置 | 检测逻辑 |
|--------|---------|---------|
| predict dtype 合法性 | Host 侧 OpDef | 仅允许 DT_FLOAT16, DT_FLOAT |
| label dtype 合法性 | Host 侧 OpDef | 与 predict 一致 |
| shape 一致性 | Host 侧 | predict.GetShapeSize() == label.GetShapeSize() |
| y dtype | Host 侧 InferDataType | 自动设为与 predict 相同 |

### 支持硬件

| 支持的芯片版本 | 状态 |
|--------------|------|
| Atlas A2 训练系列产品 (ascend910b) | ✅ |
| Atlas A3 系列产品 | ✅ |

### 算子约束限制

- predict 与 label 必须同 shape、同 dtype，不支持广播
- 输出 y 的 dtype 与 shape 与 predict 一致（Kernel 输出逐元素平方差）
- reduction 归约操作由 ACLNN 层在 Kernel 调用后完成，不在 Kernel 内实现

---

## 可维可测分析

### 精度标准 / 性能标准

| 验收标准 | 描述 | 标准来源 |
|---------|------|---------|
| 精度标准 | 与原 TBE 算子逻辑值完全对齐，满足 CANNJudge 平台精度阈值 | 任务书要求 |
| 性能标准 | 所有核参与计算场景下，性能 ≥ TBE 的 95%；小 shape 场景若无法达标，提供性能仿真图和分析结论 | 任务书要求 |

### 兼容性分析

- **接口兼容**：算子接口与现有 `aclnnMseLoss` 完全一致，上层框架无需修改
- **数据兼容**：输入输出 dtype/formatshape 与原 TBE 实现对齐
- **向后兼容**：替换后的 Ascend C 实现在语义上与原 TBE 实现完全等价
