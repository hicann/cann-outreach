# SyncBatchNormBackwardReduce 算子设计方案

## 1. 需求背景（required）

### 1.1 需求来源

CANN 训练营 2026 暑期季 – 西安交通大学专场算子开发任务。参考昇腾版本内置 `aclnnBatchNormReduceBackward` 算子的 TBE 实现，在昇腾 NPU 上基于 Ascend C 编程语言实现功能一致的算子 `SyncBatchNormBackwardReduce`，完成算子设计、开发、测试全流程工作，验收通过后提交至昇腾算子开源仓。

### 1.2 背景介绍

#### 1.2.1 SyncBatchNormBackwardReduce 算子实现优化

基于 `aclnnBatchNormReduceBackward` 算子历史 TBE 版本使用 Ascend C 编程语言进行重构优化。

#### 1.2.2 算子实现路径和相关 API 路径

- TBE 算子实现路径：`/usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe/impl/ops_legacy/dynamic/sync_batch_norm_backward_reduce_apt.py`
- 算子原型路径：`/usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_proto/inc/sync_batch_norm_backward_reduce.h`
- 算子信息库路径：`/usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe/config/ascend910b/sync_batch_norm_backward_reduce.cc`

#### 1.2.3 算子现状分析

`aclnnBatchNormReduceBackward`（即 `SyncBatchNormBackwardReduce`）是 SyncBN 反向传播中的**逐元素组合算子**（非规约算子——规约在上游已完成，本算子接收已规约的 per-channel 统计量做组合）。BatchNorm 反向传播需要计算输入梯度 `grad_x`，其表达式为：

```
grad_x = (gamma · rsqrt(var + eps)) · (grad_y − sum_dy/M − x_norm · sum_dy_x_norm/M)
```

本算子负责其中的组合步骤：接收已规约量 `sum_dy`、`sum_dy_dx_pad`（pad 后的 sum(grad_y·dx)）与前向统计量 `mean`、`invert_std`（=1/sqrt(var+eps)），计算：

```
dy_mean = mean * sum_dy
sum_dy_xmu = sum_dy_dx_pad - dy_mean        // = sum(grad_y·(x−mean))
y = sum_dy_xmu * invert_std                  // = grad_weight 相关量
```

##### 1.2.3.1 TBE 算子支持的数据类型和数据格式

**数据类型**（已由 op_proto + TBE 源码 + op_info 确认）：sum_dy、sum_dy_dx_pad、mean、invert_std、sum_dy_xmu、y 均支持 float16、float32、bfloat16。
**数据格式**（已由 op_info 确认）：全部 ND。
**shape**：op_info 中均为 `[-2]`（动态 shape，任意维度），`opMode: dynamic`。
**bin 划分**：每种 dtype 一个 bin（fp16/fp32/bf16），simplifiedKey 中 dtype 编码 fp16=1, fp32=0, bf16=27。
**适配硬件**：binInfo 路径 `ascend910_93/`，对应 Atlas A2（910B 系列）。

##### 1.2.3.2 TBE 算子实现描述

TBE 源码（`sync_batch_norm_backward_reduce.py`）实现逻辑如下：

1. **dtype 处理**：若输入 dtype 为 bfloat16 或 float16，将四个输入（sum_dy、sum_dy_dx_pad、mean、invert_std）全部 `cast_to` 到 float32 后参与计算；float32 直接计算。
   - 注：TBE 源码此处有 Python 写法 `if input_dtype == "bfloat16" or "float16":`（因字符串非空恒为 True），实际效果是所有 dtype 都执行 cast 到 float32（对 float32 是 no-op），功能不受影响。
2. **逐元素计算**（均为 `tbe` 向量接口，广播后逐元素）：
   - `dy_mean = tbe.vmul(mean, sum_dy)` —— mean 与 sum_dy 逐元素相乘
   - `sum_dy_xmu = tbe.vsub(sum_dy_dx_pad, dy_mean)` —— 减法
   - `grad_weight_res = tbe.vmul(sum_dy_xmu, invert_std)` —— 乘法，即输出 y
3. **输出 cast 回原 dtype**：float16 用 `cast_to` 回 float16；bfloat16 用 `round` 回 bfloat16；float32 不转换。
4. 返回 `[sum_dy_xmu, grad_weight_res]`，分别对应 op_proto 的 `sum_dy_xmu` 和 `y`。

**关于"二进制比较→逻辑值比较"**：TBE 源码中未发现任何二进制比较/位运算/mask 逻辑（本算子为纯逐元素 vmul/vsub/vmul）。任务书该要求列于"功能实现要求"下，结合本算子无比较逻辑的事实，初步判断该要求体现在**测试验证阶段**——即精度比较从二进制比特逐位对比改为逻辑值容差对比（atol/rtol）。此判断待评审确认；若评审认为应作用于算子内部，则需进一步澄清作用点。

##### 1.2.3.3 TBE 算子实现流程图

```
输入: sum_dy, sum_dy_dx_pad, mean, invert_std (dtype ∈ {fp16, fp32, bf16})
        │
        ▼
[若 fp16/bf16] cast 四路输入 → fp32   (fp32 跳过)
        │
        ▼
广播对齐 (ELEWISE, variable_shape)
        │
        ▼
dy_mean = vmul(mean, sum_dy)
        │
        ▼
sum_dy_xmu = vsub(sum_dy_dx_pad, dy_mean)
        │
        ▼
y = vmul(sum_dy_xmu, invert_std)
        │
        ▼
[若 fp16] cast 回 fp16 / [若 bf16] round 回 bf16   (fp32 跳过)
        │
        ▼
输出: sum_dy_xmu, y
```

通过对 TBE 版本的功能分析，当前支持的能力如下（已由 op_proto + TBE 源码确认）：

1. 四个输入 `sum_dy`、`sum_dy_dx_pad`、`mean`、`invert_std` 支持 float16、float32、bfloat16 三种数据类型。
2. 输出 `sum_dy_xmu`、`y` 同样支持上述三种数据类型。
3. op_proto 中未声明属性，算子无 attribute。
4. 算子为**逐元素（ELEWISE）**运算，输入按广播规则对齐后逐元素计算，不涉及规约。
5. **算子内部无二进制比较逻辑**；任务书"二进制比较→逻辑值比较"的适用范围见 1.2.2.2 说明（待评审确认）。

TBE 版本整体流程：输入四路 → [fp16/bf16 cast fp32] → 广播 → vmul/vsub/vmul → [cast 回原 dtype] → 输出 sum_dy_xmu、y。

## 2. 需求分析

### 2.1 外部组件依赖

不涉及外部组件依赖。

### 2.2 内部适配模块

适配 Aclnn 接口和图模式调用。

## 3. 需求模块设计

### 3.1 算子原型

#### 3.1.1 原型设计

| 名称 | 类别 | dtype | format | shape | 介绍 |
|---|---|---|---|---|---|
| sum_dy | 输入 | fp16/fp32/bf16 | ND | [-2] 动态 | pad 后的 grad_bias（已规约） |
| sum_dy_dx_pad | 输入 | fp16/fp32/bf16 | ND | [-2] 动态 | pad 后的 sum(grad_y·dx) |
| mean | 输入 | fp16/fp32/bf16 | ND | [-2] 动态 | 前向输入的 per-channel 均值 |
| invert_std | 输入 | fp16/fp32/bf16 | ND | [-2] 动态 | 前向输入方差的倒数 1/sqrt(var+eps) |
| sum_dy_xmu | 输出 | fp16/fp32/bf16 | ND | [-2] 动态 | = sum_dy_dx_pad − mean·sum_dy |
| y | 输出 | fp16/fp32/bf16 | ND | [-2] 动态 | = sum_dy_xmu · invert_std（grad_weight 相关量） |

> - `sum_dy_xmu = sum_dy_dx_pad - mean * sum_dy`
> - `y = sum_dy_xmu * invert_std`

##### 3.1.2 Ascend C 算子相关约束

本算子为功能对齐重构，除任务书未要求适配的部分外，其余与 TBE 对齐：

- 支持的 dtype/format/shape 与 TBE + op_info 完全一致（fp16/fp32/bf16，ND，动态 shape）。
- 计算逻辑（cast→fp32 → vmul/vsub/vmul → cast 回原 dtype）与 TBE 完全一致。
- 无功能缺失。

#### 3.1.3 相关约束

- Atlas A2 训练系列产品 / Atlas A3 系列产品：float16、float32、bfloat16。
- 四路输入 dtype 必须一致，format 均为 ND。
- 四路输入 shape 需可按广播规则对齐（主场景为四路同 shape）。

### 3.2 需求详细设计

#### 3.2.1 使能方式

| 上层框架 | 涉及的框架勾选 |
|---|---|
| TF 训练/推理 | |
| Pytorch 训练/推理 | √ |
| ATC 推理 | √ |
| Aclnn 直调 | √ |
| OPAT 调优 | |
| SGAT 子图切分 | |

**与 TBE 相比缺失的功能：**

本算子为功能对齐重构，与 TBE 相比无功能缺失。支持的能力（dtype/format/shape/计算逻辑）与 TBE 完全一致。

#### 3.2.2 需求总体设计

##### 3.2.2.1 host 侧设计

**tiling 策略：**

本算子为逐元素（ELEWISE）运算，计算过程不涉及数据的维度信息，故在 host 侧将数据视为一维向量，仅考虑数据个数，不考虑数据维度信息。

host 侧需要：

1. 获取四路输入的 shape，按广播规则求得最终广播 shape，计算 `totalLength`（广播后的总元素数）。
2. 计算每个核处理的数据量，做核间负载均衡。
3. 计算单核内 UB 分块大小，做核内切分。
4. 将 `totalLength`、分核参数、tile 参数、dtype 信息写入 tilingData 传给 kernel。

**分核策略：**

优先使用满核原则。

- 如果核间能均分，可视作无大小核区分，大核小核数据块一致；
- 如果核间不能均分，需要将余出的数据块分配到前几个核上。

- 输入数据大小计算：通过 GetInputShape 和 GetDataTypeLength 函数获取输入数据的大小和类型长度，计算出输入数据的总字节数。
- UB 内存大小和核心数量获取：通过平台信息获取 UB 内存大小和核心数量，并根据这些信息调整核心数量。

**数据分块和内存优化策略：**

充分使用 UB 空间的原则。

单核内需同时容纳四路输入 tile（sum_dy、sum_dy_dx_pad、mean、invert_std）及中间结果（dy_mean、sum_dy_xmu、y），若 fp16/bf16 还需 fp32 中间 buffer。综合考虑 double buffer。

- UB 内存大小获取：通过 GetCoreMemSize 函数获取 UB 内存的大小，用于后续的数据切分计算。
- Tile 块计算：根据 UB 内存大小和预定义的 BLOCK_SIZE 及 BUFFER_NUM，计算出每个 Tile 块的数据数量。

  ```
  tileLength = min(M, floor(UB_SIZE / (N_BUF * dtype_size * BUFFER_NUM)))
  // N_BUF = 4 路输入 + 中间结果，fp16/bf16 需额外 fp32 buffer
  ```

- 数据切分：将输入数据按照计算出的 Tile 块大小进行切分，计算出每个 core 需要处理的数据块数量和最后一个 block 的剩余数据量。
- 尾块处理：totalLength 不能被 tileLength 整除时，最后一个 tile 单独处理剩余元素。
- 设置切分参数：将计算出的切分参数（如每个 core 的数据量、Tile 块大小等）设置到 tilingData 对象中。

这些策略确保了数据在多个核心之间的均匀分布，并且在单个核心内进行了合理的切分，以提高并行处理的效率。

**tilingKey 规划策略：**

**设置条件：**

- 条件1（dtype 分支）：根据输入数据类型设置 tilingKey，kernel 侧据此选择是否执行 cast 到 fp32 的分支（fp16/bf16 需 cast，fp32 直接计算）。dtype 编码与 op_info 的 simplifiedKey 对齐：fp32=0, fp16=1, bf16=27。

**数据检测：**

- 校验四路输入的 dtype 一致性。
- 校验输入 shape 可广播对齐。
- 对不支持的数据类型/格式在 tiling 阶段返回错误码。
- 对不支持 AscendC::Cast() bfloat16 向 float32 转换的硬件，在 tiling 策略时返回报错。

##### 3.2.2.2 kernel 侧实现描述

进行 Init 和 Process 两个阶段，其中 Process 包括数据搬入（CopyIn）、计算（Compute）、搬出（CopyOut）三个阶段。

**Init 阶段：**

- 从 tilingData 读取 `totalLength`、分核参数（perCoreLength、tileLength）、dtype 等参数。
- 初始化每个核负责的数据起止偏移。
- 分配 UB 上的 LocalTensor：四路输入 tile + 中间结果（fp16/bf16 需额外 fp32 buffer）。

**CopyIn 阶段：**

- 对当前核负责的数据范围，按 `tileLength` 分块，从 GM 搬入四路输入的对应 tile 到 UB。
- 若 fp16/bf16，搬入后 Cast 到 fp32（与 TBE 一致，fp32 精度计算）。
- 广播处理：若输入 shape 不同，在 CopyIn 阶段按广播规则填充数据（参考 Addcdiv 的广播实现）。

**Compute 阶段：**

- 对每个 tile（已在 fp32）：
  1. `dy_mean = Mul(mean, sum_dy)` —— 逐元素相乘
  2. `sum_dy_xmu = Sub(sum_dy_dx_pad, dy_mean)` —— 逐元素相减
  3. `y = Mul(sum_dy_xmu, invert_std)` —— 逐元素相乘
- 三个 Ascend C 向量 API：`Mul`、`Sub`、`Mul`，与 TBE 的 `vmul`/`vsub`/`vmul` 一一对应。

**CopyOut 阶段：**

- 若 fp16，Cast 回 fp16；若 bf16，Cast/round 回 bf16（与 TBE 一致）。
- 将 `sum_dy_xmu`、`y` 两个 tile 从 UB 写回 GM 对应位置。

**Ascend C 的 SyncBatchNormBackwardReduce 算子流程：**

```
Init: 读取tiling参数 → 确定本核数据范围 → 分配UB LocalTensor
  ↓
for tile in [offset, perCoreLength, tileLength):
  CopyIn:
    GM→UB: sum_dy[tile], sum_dy_dx_pad[tile], mean[tile], invert_std[tile]
    [若fp16/bf16] Cast 四路 → fp32
    [按广播规则填充]
  Compute (fp32):
    dy_mean = Mul(mean, sum_dy)
    sum_dy_xmu = Sub(sum_dy_dx_pad, dy_mean)
    y = Mul(sum_dy_xmu, invert_std)
    [若fp16] Cast 回 fp16 / [若bf16] Cast 回 bf16
  CopyOut:
    UB→GM: sum_dy_xmu[tile], y[tile]
```

##### 3.2.2.3 Ascend C 实现流程图

见上一节文字流程图或docx文档。

##### 3.2.2.4 Ascend C 实现流程图与 TBE 流程图存在的差异点和原因【重点】

| 差异点 | TBE 实现 | Ascend C 实现 | 原因 |
|---|---|---|---|
| 数据搬运 | TBE DSL 自动管理（`tbe.compute` + `auto_schedule`） | 显式 `DataCopy` + UB 管理 | Ascend C 显式控制内存层级，便于 UB 充分利用和性能调优 |
| dtype 提升 | `tbe.cast_to` 到 fp32 | `AscendC::Cast` 到 fp32 | 接口对应，Ascend C 暴露底层 Cast 原语 |
| 逐元素计算 | `tbe.vmul` / `tbe.vsub` | `AscendC::Mul` / `AscendC::Sub` | 接口一一对应，语义完全一致 |
| 广播处理 | TBE `variable_shape` + `classify` 自动广播 | CopyIn 阶段显式按广播规则填充（参考 Addcdiv） | Ascend C 需手动处理广播，但可控性更强 |
| bf16 回转 | `tbe.round` 到 bf16 | `AscendC::Cast` 到 bf16 | 接口对应；bf16 用 Cast 实现等价 round 语义 |
| 分核 | TBE 隐式（auto_schedule） | 显式按 totalLength 均分到各核 | Ascend C 显式控制分核，确保满核利用，满足性能要求 |
| **测试比较方式** | 二进制比特比较（传统验证） | **逻辑值容差比较**（任务书要求） | **任务书要求**：比较方式从二进制比较改为逻辑值比较。本算子 TBE 实现内部无比较逻辑，故该要求体现在测试验证阶段（atol/rtol 容差判定）；此理解待评审确认 |

> **结论**：算子内部计算逻辑（cast→fp32 → vmul/vsub/vmul → cast回原dtype）Ascend C 与 TBE **完全一致**，仅实现接口和内存管理方式不同。

#### 3.2.3 支持硬件

| 支持的芯片版本 | 涉及勾选 |
|---|---|
| 香橙派 OrangePi AIpro | |
| Atlas 200I/500 A2 推理产品 | |
| Atlas 800I/T A2 | √ |
| Atlas A2 训练系列 | √ |
| Atlas A3 系列 | √ |

#### 3.2.4 算子约束限制

- 四路输入 dtype 必须一致，支持 float16、float32、bfloat16。
- 四路输入 format 均为 ND。
- 四路输入 shape 需可按广播规则对齐（op_info 为动态 shape `[-2]`，运行期确定）。
- 适配硬件：Atlas A2 训练系列产品（binInfo: ascend910_93）。

#### 3.2.5 特性交叉分析

不涉及。

### 3.3 可维可测分析

- 支持通过 aclnn 接口直调测试。
- tiling 阶段对非法输入（shape 不一致、dtype 不支持）返回明确错误码。
- kernel 侧关键路径可加调测日志（DEBUG 模式）。

### 3.4 精度标准/性能标准

| 验收标准 | 描述 | 标准来源 |
|---|---|---|
| 精度标准 | 满足 cannjudge.cn 平台对应题目默认阈值，不低于 TBE 版本 | cannjudge.cn |
| 性能标准 | 所有核参与计算场景下，性能不低于原 TBE 算子的 95%；小 shape（<10us 场景相差 ≤3us）提供性能仿真图证明一致或更优 | 任务书 |

### 3.5 兼容性分析

新算子（开源仓 experimental 目录），不涉及向后兼容性分析。
