# SoftshrinkGrad 项目设计题目

## 项目名称

基于 Ascend C 的 SoftshrinkGrad 算子开发与性能优化

## 项目背景

Softshrink 是神经网络中常用的激活函数，常用于稀疏化、去噪和阈值化场景。SoftshrinkGrad 是 Softshrink 的反向梯度算子，负责在反向传播阶段根据前向输入和阈值属性过滤上游梯度。

CANN 训练营 2026 暑期季西安交通大学专场社区任务要求参考 CANN 内置 `aclnnSoftshrinkBackward` 的 TBE 实现，使用 Ascend C 在昇腾 NPU 上实现功能一致的 SoftshrinkGrad 算子。项目完成后，算子代码目标提交至 `ops-nn` 仓库 `experimental/activation` 目录，并通过 CANN Judge 平台验收。

## 任务来源

- 任务书：[SoftshrinkGrad 算子开发任务书](https://gitcode.com/cann/cann-ops-competitions/blob/master/04_tasks/01_community-task-2026/docs/202607/SoftshrinkGrad_task_doc.md)
- 目标开源仓：[cann/ops-nn](https://gitcode.com/cann/ops-nn)
- 目标目录：`experimental/activation/softshrink_grad`
- 适配硬件：Atlas A2 训练系列产品、Atlas A3 系列产品
- 开发语言：Ascend C
- CANN 版本：以 `ops-nn` 仓指定版本为准

## 项目目标

1. 实现与 CANN 内置 TBE `aclnnSoftshrinkBackward` 功能一致的 Ascend C 算子。
2. 支持 `float16`、`float32`、`bfloat16` 数据类型。
3. 支持 ND 格式、动态 shape、动态 rank。
4. 支持泛化输入场景，包括空 tensor、小 shape、非对齐尾块、大 shape 和多维 shape。
5. 在所有核参与计算场景下，性能不低于 TBE 版本的 95%。
6. 通过 CANN Judge 平台默认精度阈值验收。
7. 输出完整交付件，包括算子设计文档、代码、README、自验证报告和验收截图。

## 算子功能定义

### 输入输出

| 名称 | 输入/输出 | 含义 | 数据类型 | 数据格式 | shape 约束 |
| --- | --- | --- | --- | --- | --- |
| grad_output | 输入 | 上游梯度 | float16, float32, bfloat16 | ND | 与 self 相同 |
| self | 输入 | Softshrink 前向输入 | float16, float32, bfloat16 | ND | 与 grad_output 相同 |
| lambd | 属性 | Softshrink 阈值 | float | 标量属性 | 默认 0.5 |
| output | 输出 | 前向输入梯度 | float16, float32, bfloat16 | ND | 与输入相同 |

### 计算公式

```text
output_i = grad_output_i, self_i > lambd or self_i < -lambd
output_i = 0,             -lambd <= self_i <= lambd
```

边界规则：

- `self_i == lambd` 时输出 0。
- `self_i == -lambd` 时输出 0。
- `lambd == 0` 时，只有 `self_i != 0` 的位置透传梯度。

## 技术路线

### 阶段一：TBE 行为对齐

1. 阅读任务书和 CANN 内置 TBE 参考路径。
2. 对齐 `aclnnSoftshrinkBackward` 的输入输出定义、属性默认值、shape 推导和边界比较规则。
3. 使用 NumPy/PyTorch 构造参考实现，用于后续自验证：

```python
expected = np.where((self > lambd) | (self < -lambd), grad_output, 0)
```

### 阶段二：Ascend C 工程实现

计划在 `ops-nn/experimental/activation/softshrink_grad` 下组织代码：

```text
softshrink_grad/
├── op_host/
│   ├── softshrink_grad_def.cpp
│   ├── softshrink_grad_infershape.cpp
│   └── softshrink_grad_tiling.cpp
├── op_kernel/
│   ├── softshrink_grad.cpp
│   ├── softshrink_grad.h
│   ├── softshrink_grad_tiling_data.h
│   └── softshrink_grad_tiling_key.h
├── op_api/
│   ├── aclnn_softshrink_backward.cpp
│   ├── aclnn_softshrink_backward.h
│   ├── softshrink_grad.cpp
│   └── softshrink_grad.h
├── docs/
│   └── aclnnSoftshrinkBackward.md
├── examples/
│   └── test_aclnn_softshrink_grad.cpp
└── tests/
    └── ut/
```

实现重点：

1. host 侧完成 dtype、shape、空指针、属性合法性校验。
2. host 侧根据输入总元素数、UB 大小、数据类型和 AICore 数生成 tiling 参数。
3. kernel 侧按 tile 搬入 `grad_output` 和 `self`，执行比较、Select 和搬出。
4. 对 `float32` 走直接计算路径。
5. 对 `float16`、`bfloat16` 预留 cast 到 `float32` 的计算路径，优先保证比较逻辑和精度稳定。
6. 根据测试结果选择 single buffer 或 double buffer 作为默认性能路径。

### 阶段三：泛化与性能优化

1. 小 shape：减少使用核数，降低调度和搬运开销。
2. 中大 shape：尽量使用全部 AICore，提高并行度。
3. 非对齐尾块：确保 DataCopy 和 CopyOut 不越界，输出结果完整。
4. fp16/bf16：对比 direct 与 cast_fp32 的精度和性能，保留通过 CANN Judge 的实现。
5. 性能不足场景：结合 profiling 或性能仿真图定位瓶颈，重点分析 UB 利用率、搬运次数、临时空间占用和分核均衡性。

## 模块设计

### host 侧模块

| 模块 | 职责 |
| --- | --- |
| 算子定义 | 声明输入、输出、属性、dtype、format、动态 shape 支持和芯片配置 |
| shape 推导 | 将输出 shape 设置为输入 shape，校验输入 shape 一致性 |
| tiling 计算 | 计算总元素数、使用核数、单核处理长度、tile 长度、尾块长度 |
| tiling key | 根据 dtype、buffer 模式、计算模式选择 kernel 分支 |

### kernel 侧模块

| 模块 | 职责 |
| --- | --- |
| Init | 初始化 GM 地址、tiling 参数、本核 offset、队列和常量 |
| CopyIn | 搬入当前 tile 的 `grad_output` 和 `self` |
| Compute | 执行 `self > lambd || self < -lambd` 比较，并 Select 输出 |
| CopyOut | 将结果写回 output GM |

### aclnn 接口模块

使用两段式 aclnn 接口：

```cpp
aclnnStatus aclnnSoftshrinkBackwardGetWorkspaceSize(
    const aclTensor *gradOutput,
    const aclTensor *self,
    double lambd,
    aclTensor *output,
    uint64_t *workspaceSize,
    aclOpExecutor **executor);

aclnnStatus aclnnSoftshrinkBackward(
    void *workspace,
    uint64_t workspaceSize,
    aclOpExecutor *executor,
    aclrtStream stream);
```

接口层负责：

1. 参数合法性检查。
2. 创建执行器。
3. 获取 workspace size。
4. 下发算子执行。

## 测试方案

### 功能测试

| 类型 | 用例说明 |
| --- | --- |
| 全透传 | `self > lambd` 或 `self < -lambd`，输出等于 `grad_output` |
| 全置零 | `self` 全部位于 `[-lambd, lambd]`，输出全 0 |
| 混合分支 | 随机输入覆盖大于、小于、区间内三类位置 |
| 边界点 | 覆盖 `self == lambd`、`self == -lambd` |
| 阈值为 0 | 覆盖 `lambd == 0` 的退化场景 |
| 空 tensor | 输入元素数为 0 |
| 尾块 | 元素数不是 32B 对齐倍数 |
| 多维 | 覆盖 1D、2D、4D、动态 rank |

### 数据类型测试

| dtype | 期望 |
| --- | --- |
| float32 | 直接与参考结果比较 |
| float16 | 按 CANN Judge 默认阈值比较 |
| bfloat16 | 按 CANN Judge 默认阈值比较 |

### 性能测试

| 场景 | 目标 |
| --- | --- |
| 小 shape | 分析 launch、分核和搬运开销 |
| 单核 shape | 验证单核 tiling 正确性 |
| 满核 shape | 性能不低于 TBE 版本 95% |
| 大 shape | 验证吞吐、分核均衡和 double buffer 收益 |

## 验收标准

1. CANN Judge 功能和精度用例全部通过。
2. CANN Judge 提交排名页面中各测试点性能满足任务书要求。
3. 自验证报告包含用例通过截图、性能数据截图和关键日志。
4. README 能说明算子功能、接口、编译、运行和测试方式。
5. 代码 PR 通过 `ops-nn` 仓检视并合入。
6. 算子设计文档通过评审并按要求提交。

## 交付件规划

| 交付件 | 说明 |
| --- | --- |
| 算子设计文档 | 描述需求、公式、输入输出、host/kernel 设计、tiling、测试与性能方案 |
| 项目设计题目 | 描述项目目标、技术路线、模块划分、计划和验收标准 |
| 算子源码 | `softshrink_grad` Ascend C 工程 |
| README | 算子使用、编译、部署、运行说明 |
| 自验证报告 | 覆盖功能、精度、性能和 CANN Judge 截图 |
| 验收压缩包 | 按任务书要求上传 CANN Judge |

## 进度计划

| 阶段 | 工作内容 | 输出 |
| --- | --- | --- |
| D1 | 任务书阅读、TBE 行为分析、设计文档编写 | 设计文档、项目设计题目 |
| D2 | 搭建 `softshrink_grad` 工程，完成 host 与 kernel 基础实现 | 可编译工程 |
| D3 | 完成 aclnn 接口、README、基础 example | 可运行单算子样例 |
| D4 | 补齐 dtype、动态 shape、尾块、空 tensor 测试 | 功能自验证记录 |
| D5 | 性能调优、CANN Judge 提交、自验证报告整理 | 验收材料 |
| D6 | 根据评审意见修改，提交 `ops-nn` 代码 PR | 合入申请 PR |

## 风险与应对

| 风险 | 影响 | 应对 |
| --- | --- | --- |
| TBE 对边界值或 NaN 行为存在特殊处理 | 精度对齐失败 | 使用内置 TBE 结果作为 golden，对边界用例单独校验 |
| fp16/bf16 原生比较精度不稳定 | 精度验收失败 | 使用 cast_fp32 路径保证比较逻辑一致 |
| 小 shape 性能低于 TBE | 小用例耗时偏高 | 减少 usedCoreNum，降低 UB 临时空间和搬运次数 |
| 非对齐尾块处理错误 | 泛化用例失败 | 单独设计尾块测试，检查 CopyIn/CopyOut 长度 |
| CANN Judge 环境和本地环境差异 | 验收复现困难 | 固定 CANN 版本、记录分支、commit、运行命令和截图 |

## 预期成果

完成 SoftshrinkGrad Ascend C 算子设计与实现方案，后续按该方案开发并提交到 `ops-nn`。项目最终应达到：

1. 功能与 `aclnnSoftshrinkBackward` TBE 实现一致。
2. 支持任务书要求的数据类型和泛化 shape。
3. 精度通过 CANN Judge 默认阈值。
4. 满核场景性能达到 TBE 版本 95% 以上。
5. 文档、代码、测试报告和验收材料完整可复现。
