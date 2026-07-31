# TruncateMod 算子设计方案

## 1. 文档信息

| 项目 | 内容 |
| --- | --- |
| 算子名称 | TruncateMod |
| 技术方向 | Ascend C 自定义算子 |
| 适配硬件 | Atlas A2 训练系列产品；任务目标包含 Atlas A3 系列产品 |
| 实际验证环境 | Atlas A2，Ascend 910B4，CANN 8.5.0，EulerOS 2.0 SP10，aarch64 |
| 输入个数 | 2，`x1`、`x2` |
| 输出个数 | 1，`y` |
| 支持格式 | ND |
| 支持类型 | BF16、FLOAT16、FLOAT、INT32、INT8、UINT8 |
| 私仓路径 | `yaossss/TruncateMod` |
| 代码分支 | `main` |
| 算子代码目录 | `code/` |

本方案面向 CANN 训练营 2026 暑期季 TruncateMod 算子任务，设计依据为任务书、内置算子行为和当前 Ascend C 实现。

## 2. 需求分析

### 2.1 数学定义

对输入张量逐元素计算：

```text
y = x1 - trunc(x1 / x2) * x2
```

其中 `trunc` 表示向零截断。因此结果与 Python/C 语言中以零为方向的整数商保持一致，不采用向负无穷取整的 `floor` 语义。

### 2.2 输入输出约束

- `x1` 和 `x2` 为必选输入，数据类型必须一致。
- 输出 `y` 的数据类型与输入一致。
- 输入格式为 ND，输出格式为 ND。
- 支持同形状计算、标量广播、单侧完整输入广播和一般右对齐广播。
- 输入维度按末尾维度右对齐；每一维满足两者相等或其中一个为 1。
- 当前实现最多支持 8 维广播描述。
- 输出形状为两个输入按广播规则计算得到的形状。
- 除零、非有限浮点输入等边界行为以目标 CANN 版本内置算子语义为准，正常验收数据应遵循任务书定义的合法输入范围。

### 2.3 功能目标

1. 与内置 TruncateMod 的计算公式、数据类型和广播行为保持一致。
2. 覆盖小形状、非 32 字节对齐尾块、大形状和负数商等场景。
3. 通过 CANNJudge 的功能和精度校验。
4. 在任务书规定的性能场景中达到内置算子要求。

## 3. 算子接口设计

算子由 Host 侧注册为 `TruncateMod`，对外生成以下 ACLNN 接口：

```cpp
aclnnStatus aclnnTruncateModGetWorkspaceSize(
    const aclTensor *x1,
    const aclTensor *x2,
    const aclTensor *y,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

aclnnStatus aclnnTruncateMod(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);
```

调用流程如下：

1. 创建并准备 `x1`、`x2` 和输出张量 `y`。
2. 调用 `aclnnTruncateModGetWorkspaceSize` 获取 workspace 大小及 executor。
3. 按返回的大小申请 workspace。
4. 调用 `aclnnTruncateMod` 发起计算。
5. 在需要读取主机侧结果时同步 stream。
6. 释放 workspace、executor、tensor 和 stream 等资源。

## 4. 总体架构

工程按 Host、Kernel 和验证示例划分：

```text
code/
├── op_host/
│   ├── truncate_mod_def.cpp          算子输入输出及类型格式注册
│   ├── truncate_mod_infershape.cpp   广播输出 Shape 推导
│   └── truncate_mod_tiling.cpp       Tiling 参数计算和 BlockDim 设置
├── op_kernel/
│   ├── truncate_mod.cpp              Kernel 入口和模板分发
│   ├── truncate_mod.h                Ascend C Kernel 主实现
│   ├── truncate_mod_tiling_data.h    Host/Kernel 共享 Tiling 数据
│   └── truncate_mod_tiling_key.h     Tiling Key 定义
└── examples/
    └── test_aclnn_truncate_mod*.cpp  ACLNN 端到端验证示例
```

数据流为：

```text
GM x1/x2
   │
   ├─ Host 根据 Shape、dtype、广播关系生成 Tiling
   │
   ├─ CopyIn：连续搬运、标量填充或广播 Gather
   │
   ├─ Vector：转换为计算类型并执行除法、截断、乘法、减法
   │
   └─ CopyOut：处理尾块后写回 GM y
```

## 5. Host 侧设计

### 5.1 算子注册

`truncate_mod_def.cpp` 为三个 Tensor 注册相同的数据类型集合：

- `ge::DT_BF16`
- `ge::DT_FLOAT16`
- `ge::DT_FLOAT`
- `ge::DT_INT32`
- `ge::DT_INT8`
- `ge::DT_UINT8`

所有输入输出使用 `ge::FORMAT_ND`，并启用自动连续化以满足 Kernel 对连续 GM 访问的要求。

### 5.2 Shape 推导

`truncate_mod_infershape.cpp` 对输入 Shape 做右对齐：

1. 取两个输入的最大 Rank。
2. 在较低 Rank 的输入前补 1。
3. 逐维检查形状相等或存在 1。
4. 逐维取最大值生成输出 Shape。
5. 非法广播关系返回失败状态。

### 5.3 Tiling 数据

`truncate_mod_tiling.cpp` 将以下信息写入 `TruncateModTilingData`：

- 输出总元素数 `totalLength`。
- 两个输入的元素数 `x1Length`、`x2Length`。
- 广播 Rank 和输出各维大小 `outputDims`。
- 两个输入的广播步长 `x1Strides`、`x2Strides`。
- 同形状、标量、单侧完整输入等场景标志。
- 当前 dtype 对应的 tile 长度。

输入维度为 1 时，对应广播步长置为 0。Kernel 通过线性输出坐标和步长计算输入地址。

### 5.4 多核切分

以 32 个元素为一个基本 Block：

```text
totalBlocks = ceil(totalLength / 32)
blockDim = min(totalBlocks, deviceCoreNum)
```

当 Block 数不能被核数整除时，前若干个核多处理一个 Block，保证任务负载尽量均衡。每个核根据 `GetBlockIdx()` 计算自己的 `[start_, end_)` 范围，不发生跨核写冲突。

### 5.5 Tile 选择

为降低 UB 占用并减少循环次数，Tiling 按数据类型选择 tile：

| 数据类型 | tileLength |
| --- | ---: |
| FLOAT、FLOAT16、BF16 | 6144 |
| INT32、INT8、UINT8 | 4096 |

实际工作区由双缓冲输入输出、截断临时区和必要的浮点转换区组成。Host 侧将自定义 Kernel workspace 设置为 0，ACLNN 内部若需连续化则由 ACLNN 处理。

## 6. Kernel 设计

### 6.1 缓冲区规划

Kernel 使用 `TPipe` 和双缓冲队列：

- `inputQueueX1_`：输入 x1，2 个 Buffer。
- `inputQueueX2_`：输入 x2，2 个 Buffer。
- `outputQueueY_`：输出 y，2 个 Buffer。
- `truncBuf_`：保存向零截断后的 `int32` 商。
- `fpX1Buf_`、`fpX2Buf_`、`fpYBuf_`：非 FLOAT 输入的 FP32 计算缓冲区。
- `int16Buf_`：INT8/UINT8 转换使用的中间缓冲区。

单个 tile 的基本处理顺序为：

```text
CopyIn -> Compute -> CopyOut
```

队列保证搬运、计算和回收过程的资源生命周期清晰，避免局部 Tensor 被提前释放。

### 6.2 CopyIn 路径

根据 Tiling 标志选择路径：

1. **同形状**：x1、x2 直接连续搬运。
2. **x1 标量**：x1 使用 `FillScalar`，x2 连续搬运。
3. **x2 标量**：x2 使用 `FillScalar`，x1 连续搬运。
4. **一侧完整输入**：完整输入连续搬运，另一侧使用广播 Gather。
5. **一般广播**：根据输出线性坐标和输入步长逐元素计算源地址。

当搬运字节数不是 32 字节整数倍时，使用 `DataCopyPad` 处理尾块，避免越界访问和尾部数据污染。满足对齐条件时使用 `DataCopy`，减少小块搬运开销。

### 6.3 广播地址计算

对于输出线性位置 `linear`，从最后一维开始反解各维坐标：

```text
coord[axis] = linear % outputDims[axis]
linear      = linear / outputDims[axis]
sourceOffset += coord[axis] * inputStride[axis]
```

广播维的 stride 为 0，因此该维的所有输出元素会访问同一个输入元素。连续后缀和整齐重复段使用批量搬运或 `FillScalar`，非整齐段逐元素读取，优先保证任意合法广播形状的正确性。

### 6.4 向量计算

#### FLOAT

FLOAT 输入直接在 FP32 Tensor 上计算：

```text
q = Div(x1, x2)
q = Cast(q, INT32, CAST_TRUNC)
q = Cast(q, FLOAT)
y = x1 - q * x2
```

#### FLOAT16、BF16

先将输入转换到 FP32，完成除法、截断、乘法和减法，再转换回原类型。这样可以避免低精度除法导致商截断边界偏移。

#### INT32

转换到 FP32 完成商计算和向零截断，再按 INT32 输出语义转换回结果。

#### INT8、UINT8

由于目标 910B Cast 路径的限制，输入先经过 FP16 中间缓冲，再转换到 FP32 计算。结果按整数输出语义转换回 INT8 或 UINT8。

所有计算阶段在关键向量指令之间使用 `PipeBarrier<PIPE_V>()`，保证临时 Tensor 的生产和消费顺序明确。

## 7. 正确性与精度设计

### 7.1 重点场景

- 正数、负数和正负混合输入。
- 被除数绝对值小于除数。
- 商为负且接近整数边界。
- 同形状、标量广播、不同 Rank 广播。
- 输入元素数不是 32 字节对齐单位的尾块。
- 大形状多核切分。
- BF16、FLOAT16、FLOAT、INT32、INT8、UINT8 六类 dtype。

### 7.2 精度策略

- 浮点输入统一使用 FP32 完成核心除法和乘减计算。
- 商使用 `CAST_TRUNC` 实现向零截断。
- 输出转换遵循目标 dtype 的舍入和截断规则。
- CANNJudge 以任务平台默认精度阈值进行最终验收。

## 8. 性能设计

性能优化重点如下：

1. 通过多核 Block 切分覆盖大形状输入。
2. FLOAT/半精度使用 6144 tile，整数类型使用 4096 tile，平衡 UB 占用与循环次数。
3. 同形状和标量广播走连续搬运或 Duplicate 快路径。
4. 连续后缀使用批量 DataCopy，32 字节尾块使用 DataCopyPad。
5. 双缓冲队列减少 Tensor 分配和释放造成的停顿。
6. 仅在非 FLOAT 类型使用转换缓冲，FLOAT 路径避免额外转换。
7. 对重复广播段使用对齐安全的 FillScalar，非对齐场景回退逐元素路径。

目录中的 `核心算法性能测试-.xlsx` 记录了 50 个 CANNJudge 测试点。当前表格检查结果为：

| 指标 | 结果 |
| --- | ---: |
| 测试点总数 | 50 |
| 自定义算子通过数 | 50 |
| 精度错误数 | 0 |
| 自定义耗时高于或等于 Baseline 的测试点 | 0 |
| 最大自定义耗时 / Baseline 耗时 | 约 0.950 |

性能结论以 CANNJudge 的最终提交结果为准，表格和截图作为自验证证据保存。

## 9. 测试方案

### 9.1 本地构建

```bash
source /home/ma-user/Ascend/cann-8.5.0/set_env.sh
cd /home/ma-user/work/TruncateMod_problem_206_template/code
bash build.sh --make_clean
bash build.sh -j4
cd build
./install.sh
```

### 9.2 ACLNN 示例

```bash
cd /home/ma-user/work/TruncateMod_problem_206_template/code/examples
bash run.sh
```

覆盖同形状和广播场景，包含 FLOAT16、FLOAT32、INT8、UINT8、INT32。

### 9.3 AscendOpTest/CANNJudge

目录中的 `通过截图.png` 显示 CANNJudge 测试点 37 至 50 全部 Pass，整体结果为 `通过：50 / 50`。完整性能明细保存在 `核心算法性能测试-.xlsx` 中，覆盖所有 50 个测试点。

### 9.4 测试结果判定

- 功能：所有测试点 Pass。
- 精度：错误占比为 0 或满足平台默认阈值。
- 性能：每个测试点的自定义耗时低于对应 Baseline。
- 交付：报告、截图和性能表可复现并能定位到代码和测试命令。

## 10. 风险与后续工作

| 风险/事项 | 当前状态 | 后续动作 |
| --- | --- | --- |
| A3 硬件验证 | 当前环境无 A3 设备 | 获取 A3 环境后补充编译和运行记录 |
| 除零和非有限浮点输入 | 未作为稳定验收场景 | 对照目标 CANN 版本确认语义并补充用例 |
| 大形状广播性能 | 已实现通用路径 | 使用 CANNJudge 代表性大形状继续确认 |
| 开源仓合入 | 私仓已提交 | 按任务流程提交 Issue 和 PR 到 `ops-math/experimental/math` |

## 11. 交付清单

- [x] Ascend C Host 和 Kernel 实现。
- [x] 算子定义、Shape 推导和 Tiling 设计。
- [x] 六类 dtype 和 ND 广播路径设计。
- [x] ACLNN 端到端验证。
- [x] CANNJudge 50/50 通过截图。
- [x] 50 个测试点性能 Excel。
- [ ] A3 硬件验证证据。
- [ ] CANN 开源仓 `ops-math/experimental/math` PR 合入。

## 12. 版本记录

| 版本 | 日期 | 修改内容 |
| --- | --- | --- |
| 1.0 | 2026-07-27 | 根据当前 Ascend C 实现、CANNJudge 截图和性能表完成设计方案 |

