# Softmax 算子参考答案

## 概述

Softmax 是深度学习中极其常用的算子，尤其在 Attention 机制中。本参考答案使用 PyPTO 框架从底层逻辑开始实现一个数值稳定的 Softmax 算子，支持**动态 Batch Size**，并展示了分块（Tiling）配置与内核循环处理的完整流程。

计算公式：

$$\text{Softmax}(x_i) = \frac{e^{x_i - \max(x)}}{\sum_j e^{x_j - \max(x)}}$$

其中减去最大值（Max Subtraction）是为了保证数值稳定性，防止指数运算溢出。

## 算子规格

| 项目 | 说明 |
|------|------|
| 算子名称 | softmax |
| 计算公式 | `softmax(x) = exp(x - amax(x, dim=-1)) / sum(exp(x - amax(x, dim=-1)), dim=-1)` |
| 数据类型 | float32 |
| 精度标准 | rtol=3e-3, atol=3e-3 |
| 动态轴 | 所有维度均为动态（`pypto.DYNAMIC`） |
| 计算维度 | dim=-1（最后一维） |

### 输入输出

| 参数 | 方向 | shape | dtype | 说明 |
|------|------|-------|-------|------|
| input_tensor | 输入 | [batch, seqlen, head, dim] | float32 | 输入张量，全部维度动态 |
| output_tensor | 输出 | [batch, seqlen, head, dim] | float32 | Softmax 计算结果 |

## 文件结构

```
kadc_test/kadc_answers/
├── softmax.py    # Softmax 算子完整实现（kernel + 测试验证）
└── README.md     # 本文档
```

## 实现要点

### 1. 核心计算逻辑 (`softmax_core`)

将 Softmax 分解为四个基本操作，保证数值稳定性：

```python
def softmax_core(x: pypto.Tensor) -> pypto.Tensor:
    # exp(x - amax(x, dim=-1)) / sum(exp(x - amax(x, dim=-1))
    # 沿最后一维求最大值 amax(x)
    # 减去最大值，防止溢出
    # 计算指数
    # 求指数之和
    # 归一化
    # return 
```

**涉及的 PyPTO API**:

| API | 用途 |
|-----|------|
| `pypto.frontend.jit` | Kernel JIT 编译装饰器 |
| `pypto.Tensor([pypto.DYNAMIC, ...], ...)` | 声明全动态维度张量 |
| `pypto.loop` | 生成硬件级循环 |
| `pypto.set_vec_tile_shapes` | 设置向量操作的 tile 尺寸 |
| `pypto.amax` | 沿指定维度求最大值 |
| `pypto.exp` | 逐元素指数运算 |
| `pypto.sum` | 沿指定维度求和 |

**参考文档**：
- https://gitcode.com/cann/pypto/blob/master/docs/api/config/pypto-set_vec_tile_shapes.md
- https://gitcode.com/cann/pypto/blob/master/docs/api/operation/pypto-amax.md
- https://gitcode.com/cann/pypto/blob/master/docs/api/operation/pypto-exp.md
- https://gitcode.com/cann/pypto/blob/master/docs/api/operation/pypto-sum.md


### 2. JIT 内核封装 (`softmax_kernel`)

内核函数负责管理 Tiling 配置和 Batch 维度的循环处理：

```python
@pypto.frontend.jit(runtime_options={"run_mode": global_run_mode})
def softmax_kernel(
    input_tensor: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_FP32),
    output_tensor: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_FP32)):
    bs, seqlen, head, dim = input_tensor.shape
    tile_b = 1
    b_loop = bs // tile_b

    pypto.set_vec_tile_shapes(1, 4, 1, 64)

    for idx in pypto.loop(0, b_loop, 1, name="LOOP_L0_bIdx", idx_name="idx"):
        b_offset = idx * tile_b
        b_offset_end = (idx + 1) * tile_b
        input_view = input_tensor[b_offset:b_offset_end, :seqlen, :head, :dim]
        softmax_out = softmax_core(input_view)
        output_tensor[b_offset:, ...] = softmax_out
```

### 3. 关键设计决策

#### 全动态形状声明

使用 `[pypto.DYNAMIC, ...]` 声明张量形状，所有维度在运行时确定，支持任意 batch size 输入：

```python
input_tensor: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_FP32)
```

#### 向量分块配置

通过 `pypto.set_vec_tile_shapes(1, 4, 1, 64)` 设置向量处理单元的 tile 尺寸，优化 NPU 硬件执行效率。其中 `(1, 4, 1, 64)` 对应 `(batch, seqlen, head, dim)` 各维度的分块大小。

#### Batch 维度循环切分

使用 `pypto.loop` 对 batch 维度进行逐条处理，每次处理 `tile_b=1` 条数据，在循环内部通过切片获取当前 batch 的视图后调用 `softmax_core`。

## 环境要求

- CANN 工具链（Ascend 950PR / Atlas A2 / Atlas A3）
- PyPTO 框架
- PyTorch + torch_npu
- 环境变量 `TILE_FWK_DEVICE_ID` 已设置（如 `export TILE_FWK_DEVICE_ID=0`）

## 运行方式

### 环境准备

```bash
# 配置 CANN 环境变量
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 设置设备 ID
export TILE_FWK_DEVICE_ID=0
```

### 执行脚本

```bash
# NPU 模式运行（默认）
python softmax.py

# 列出所有测试用例
python softmax.py --list

# 运行指定测试用例
python softmax.py softmax::test_softmax

# 仿真模式运行
python softmax.py --run_mode sim
```

### 测试用例

| 测试用例 | 说明 | 输入 shape |
|----------|------|------------|
| `softmax::test_softmax` | 动态 Batch Size 的 Softmax 验证，与 `torch.softmax` 对比 | (32, 32, 1, 256) |

## 注意事项

- **Tiling 调优**：算子性能高度依赖于 `set_vec_tile_shapes` 的设置，建议根据实际的隐层维度（Hidden Size）进行调整。
- **计算维度**：本实现固定在 `dim=-1`（最后一维）上计算 Softmax，如需在其他维度计算，需相应调整 `amax` 和 `sum` 的 `dim` 参数。
- **DYNAMIC Loop**：首次调用时确定循环次数并编译，后续调用可使用更少的迭代次数，但不可超出首次编译时的循环次数，因此首次调用应使用最大的 batch size。
