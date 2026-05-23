# softmax 算子设计文档

## 1. 概述

### 1.1 算子名称

softmax

### 1.2 功能描述

对输入矩阵的每一行独立计算 softmax，将每个元素转换为概率值（所有元素之和为 1）。

### 1.3 数学公式

$$
\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}
$$

其中 $x_i$ 为同一行中的元素，归约沿最后一维（N 维）进行。

### 1.4 算法描述

采用数值稳定的 safe softmax 实现，分 5 步计算：

1. **求行最大值**：$m_i = \max_j(x_{i,j})$
2. **减最大值**：$x'_{i,j} = x_{i,j} - m_i$（防止 exp 溢出）
3. **求指数**：$e_{i,j} = \exp(x'_{i,j})$
4. **求行指数和**：$s_i = \sum_j e_{i,j}$
5. **归一化**：$o_{i,j} = e_{i,j} / s_i$

### 1.5 数据流图

```
输入 A(M, N) → [reduce_max → m(M)] → [sub max → x'(M,N)] → [exp → e(M,N)] → [reduce_sum → s(M)] → [div → 输出 B(M,N)]
```

---

## 2. 编程模式选型

### 2.1 模式结论

**选定模式**: Developer + T.tile 扩展原语（混合风格）

### 2.2 选型理由

softmax 为纯 Vector 算子（element-wise + reduction），不含 matmul/Cube 操作：
- 计算类型：纯 Vector（无 GEMM）
- 无需 Cube/Vector 核间流水线
- 无需手动管理 L0A/L0B/L0C
- 内存路径仅涉及 GM → UB → GM
- 使用 T.alloc_ub 显式指定 UB（参考 normalization 示例模式）
- 使用 T.tile.xxx 扩展原语直接触发 Vector 硬件指令，性能优于 T.Parallel + 符号运算
- 使用 T.tile.broadcast 处理归约结果到 2D 的广播
- 开启 AUTO_CV_COMBINE + AUTO_CV_SYNC + AUTO_SYNC 确保编译器正确处理

### 2.3 模式影响

| 维度 | 本算子的选择 |
|------|-------------|
| 内存分配 | T.alloc_ub（显式指定 UB） |
| 计算方式 | T.tile.sub/exp/div/broadcast（Vector 硬件指令） |
| 作用域 | 编译器自动分离（AUTO_CV_COMBINE） |
| 同步方式 | 自动同步（AUTO_SYNC + AUTO_CV_SYNC） |

---

## 3. API 映射设计

### 3.1 公式拆解

| 步骤 | 数学表达 | 说明 |
|------|----------|------|
| 1 | $m_i = \max_j(x_{i,j})$ | 求每行最大值 |
| 2 | $x'_{i,j} = x_{i,j} - m_i$ | 减最大值，防溢出 |
| 3 | $e_{i,j} = \exp(x'_{i,j})$ | 求指数 |
| 4 | $s_i = \sum_j e_{i,j}$ | 求每行指数和 |
| 5 | $o_{i,j} = e_{i,j} / s_i$ | 归一化 |

### 3.2 TileLang API 映射

| 步骤 | 数学表达 | TileLang API | 参数 | 模式 |
|------|----------|-------------|------|------|
| 1 | $m_i = \max_j(x_{i,j})$ | T.reduce_max(a_ub, max_ub, dim=-1) | buffer=(ROWS, N), out=(ROWS, 1), dim=-1 | Developer+tile |
| 2 | $x'_{i,j} = x_{i,j} - m_i$ | T.tile.broadcast + T.tile.sub | max_ub(ROWS,1) → broadcast → max_tile(ROWS,N); T.tile.sub(exp_ub, a_ub, max_tile) | Developer+tile |
| 3 | $e_{i,j} = \exp(x'_{i,j})$ | T.tile.exp | T.tile.exp(exp_ub, exp_ub) | Developer+tile |
| 4 | $s_i = \sum_j e_{i,j}$ | T.reduce_sum(exp_ub, sum_ub, dim=-1) | buffer=(ROWS, N), out=(ROWS, 1), dim=-1 | Developer+tile |
| 5 | $o_{i,j} = e_{i,j} / s_i$ | T.tile.broadcast + T.tile.div | sum_ub(ROWS,1) → broadcast → sum_tile(ROWS,N); T.tile.div(b_ub, exp_ub, sum_tile) | Developer+tile |

### 3.3 计算伪代码

```python
with T.Kernel(m_num, is_npu=True) as (cid, vid):
    row_start = cid * block_M + vid * ROWS
    ROWS = block_M // VEC_NUM

    a_ub = T.alloc_ub((ROWS, N), dtype)
    max_ub = T.alloc_ub((ROWS, 1), dtype)
    max_tile = T.alloc_ub((ROWS, N), dtype)
    exp_ub = T.alloc_ub((ROWS, N), dtype)
    sum_ub = T.alloc_ub((ROWS, 1), dtype)
    sum_tile = T.alloc_ub((ROWS, N), dtype)
    b_ub = T.alloc_ub((ROWS, N), dtype)

    T.copy(A[row_start : row_start + ROWS, :], a_ub)

    T.reduce_max(a_ub, max_ub, dim=-1)
    T.tile.broadcast(max_tile, max_ub)
    T.tile.sub(exp_ub, a_ub, max_tile)
    T.tile.exp(exp_ub, exp_ub)

    T.reduce_sum(exp_ub, sum_ub, dim=-1)
    T.tile.broadcast(sum_tile, sum_ub)
    T.tile.div(b_ub, exp_ub, sum_tile)

    T.copy(b_ub, B[row_start : row_start + ROWS, :])
```

### 3.4 API 可行性确认

| API | 来源 | 验证状态 |
|-----|------|---------|
| T.alloc_ub | api-kernel-memory.md §2 Expert 模式 | ✅ 已确认 |
| T.copy（显式切片语法） | rms_norm 示例 | ✅ 已确认 |
| T.reduce_max(buffer, out, dim=-1) | api-compute.md §2 + rms_norm 示例 | ✅ 已确认 |
| T.reduce_sum(buffer, out, dim=-1) | api-compute.md §2 + rms_norm 示例 | ✅ 已确认 |
| T.tile.broadcast | rms_norm 示例（inv_rms 广播） | ✅ 已确认 |
| T.tile.sub/exp/div | api-compute.md §4 | ✅ 已确认 |

---

## 4. 数据规格与内存规划

### 4.1 输入张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| A | (M, N) | float32 | 输入矩阵 |

### 4.2 输出张量

| 参数名 | Shape | dtype | 说明 |
|--------|-------|-------|------|
| B | (M, N) | float32 | 输出矩阵（每行归一化为概率分布） |

### 4.3 中间缓冲区

| Buffer 名 | Shape | dtype | 存储层级 | 用途 |
|-----------|-------|-------|----------|------|
| a_ub | (ROWS, N) | float32 | UB（T.alloc_ub） | 输入行 tile 缓冲 |
| max_ub | (ROWS, 1) | float32 | UB（T.alloc_ub） | 行最大值（reduce_max 输出） |
| max_tile | (ROWS, N) | float32 | UB（T.alloc_ub） | max_ub 广播后的 2D tile |
| exp_ub | (ROWS, N) | float32 | UB（T.alloc_ub） | 指数结果（sub+exp 输出） |
| sum_ub | (ROWS, 1) | float32 | UB（T.alloc_ub） | 行指数和（reduce_sum 输出） |
| sum_tile | (ROWS, N) | float32 | UB（T.alloc_ub） | sum_ub 广播后的 2D tile |
| b_ub | (ROWS, N) | float32 | UB（T.alloc_ub） | 输出行 tile 缓冲（div 输出） |

其中 ROWS = block_M // VEC_NUM = block_M // 2

### 4.4 内存搬运路径

```
GM[A] --T.copy--> UB[a_ub]
UB[a_ub] --reduce_max--> UB[max_ub]
UB[max_ub] --T.tile.broadcast--> UB[max_tile]
UB[a_ub] + UB[max_tile] --T.tile.sub--> UB[exp_ub]
UB[exp_ub] --T.tile.exp--> UB[exp_ub]
UB[exp_ub] --reduce_sum--> UB[sum_ub]
UB[sum_ub] --T.tile.broadcast--> UB[sum_tile]
UB[exp_ub] + UB[sum_tile] --T.tile.div--> UB[b_ub]
UB[b_ub] --T.copy--> GM[B]
```

### 4.5 UB 内存预算

以 block_M=2, ROWS=1, N=1024 为例：

| Buffer | Shape | dtype | 大小 (Bytes) |
|--------|-------|-------|-------------|
| a_ub | (1, 1024) | float32 | 4096 |
| max_ub | (1, 1) | float32 | 4 |
| max_tile | (1, 1024) | float32 | 4096 |
| exp_ub | (1, 1024) | float32 | 4096 |
| sum_ub | (1, 1) | float32 | 4 |
| sum_tile | (1, 1024) | float32 | 4096 |
| b_ub | (1, 1024) | float32 | 4096 |
| **总计（无复用）** | | | 16412 |
| **总计（MEMORY_PLANNING 复用）** | | | ≤ 12288（a_ub/max_tile/exp_ub 可复用） |

UB 容量上限（A2/A3）：192KB = 196608 Bytes，远小于上限 ✓

**N 上限估算**（无复用，ROWS=1）：6 * N * 4 ≤ 196608 → N ≤ 8192

### 4.6 动态轴定义

无。M 和 N 为编译时常量。若需动态 shape，可使用 T.dyn['M'] / T.dyn['N']。

### 4.7 JIT 配置

```python
@tilelang.jit(
    out_idx=[-1],
    pass_configs={
        tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
        tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
        tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
        tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
    },
)
```

---

## 5. Tiling 策略

### 5.1 计算类型

**类型**: 纯 Vector

**判定依据**: 算子仅包含 element-wise 运算（sub、exp、div）和 reduction（reduce_max、reduce_sum），无 matmul，无 Cube 操作。

### 5.2 Block 划分

```python
block_M = 2     # 每个 block 处理 2 行
VEC_NUM = 2     # 每个 AI Core 有 2 个 Vector 计算单元
ROWS = block_M // VEC_NUM  # 每个 vid 处理 1 行
m_num = M // block_M       # block 数量 = M 行数 / 2
```

- **block_M=2 选择理由**：softmax 每行独立计算，block_M 决定每个 block 处理的行数。选择 2 配合 VEC_NUM=2，每个 vid 处理 1 行，UB 占用最小。
- **N 维不切分**：softmax 的 reduce 操作需要完整行数据，N 维必须整行加载到 UB。

### 5.3 约束分析

- **对齐约束**: N 不需要对齐约束（reduce_max/sum 支持任意长度）
- **UB 容量**: N=1024 时总 buffer ≈ 16KB < 192KB ✓
- **L0 容量**: 不适用（纯 Vector 算子，无 Cube 计算）

### 5.4 注意事项

- **N 过大时**：若 N > 8192（float32，含 broadcast buffer），单行无法完整放入 UB，需采用 online softmax（分块迭代累加 max 和 sum），类似 FlashAttention 中的 softmax 处理方式。
- **M 非整除**：使用 T.ceildiv(M, block_M) 计算 block 数量，边界 block 需处理不足 block_M 行的情况。当前实现假设 M 可被 block_M 整除；非整除场景需添加尾块保护逻辑。

---

## 6. 循环与调度结构

### 6.1 循环结构总结

| 维度 | 循环类型 | API | 理由 |
|------|----------|-----|------|
| M 方向（行） | block 级并行 | T.Kernel(m_num) | 每个 block 处理 block_M 行 |
| vid 行切分 | 硬件并行 | VEC_NUM=2 | 每个 vid 处理 ROWS 行 |
| reduce_max | 行级归约 | T.reduce_max(dim=-1) | 每行求最大值 |
| sub/exp/div | 元素级向量化 | T.tile.sub/exp/div/broadcast | 直接触发 Vector 硬件指令 |
| reduce_sum | 行级归约 | T.reduce_sum(dim=-1) | 每行求指数和 |

### 6.2 循环伪代码

```python
# Block 级并行（隐式，由 T.Kernel 管理）
with T.Kernel(m_num, is_npu=True) as (cid, vid):
    row_start = cid * block_M + vid * ROWS

    # 每个 vid 独立处理 ROWS 行
    T.copy(A[row_start : row_start + ROWS, :], a_ub)
    T.reduce_max(a_ub, max_ub, dim=-1)
    T.tile.broadcast(max_tile, max_ub)
    T.tile.sub(exp_ub, a_ub, max_tile)
    T.tile.exp(exp_ub, exp_ub)
    T.reduce_sum(exp_ub, sum_ub, dim=-1)
    T.tile.broadcast(sum_tile, sum_ub)
    T.tile.div(b_ub, exp_ub, sum_tile)
    T.copy(b_ub, B[row_start : row_start + ROWS, :])
```

### 6.3 流水线优化

当前为单次计算（无迭代），不使用 T.Pipelined。若未来采用 online softmax（N 维分块迭代），可引入 T.Pipelined 优化 DMA 搬运与计算重叠。

### 6.4 尾块处理

当前实现假设 M 可被 block_M 整除。若 M 不能整除：
- 方案一：使用 T.ceildiv 计算 block 数量，最后一个 block 读取超出范围时依赖硬件零填充
- 方案二：添加条件判断，仅处理有效行（需引入 T.alloc_var 作为标志位）

---

## 7. 同步策略

### 7.1 同步模式

**模式**: 自动同步（AUTO_SYNC + AUTO_CV_SYNC）

### 7.2 同步点说明

开启 AUTO_SYNC + AUTO_CV_SYNC，编译器自动在 T.copy、T.tile 操作和 reduce 操作之间插入同步指令，无需手动同步。

关键同步需求：
- T.copy(GM → UB) 后：需等待 DMA 搬运完成 → AUTO_SYNC 自动处理
- T.reduce_max/sum 后：需等待归约完成 → AUTO_SYNC 自动处理
- T.tile.broadcast/sub/exp/div 后：需等待 Vector 计算完成 → AUTO_SYNC 自动处理
- T.copy(UB → GM) 前：需等待计算完成 → AUTO_SYNC 自动处理

### 7.3 pass_configs 配置

```python
pass_configs = {
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_COMBINE: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_CV_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_AUTO_SYNC: True,
    tilelang.PassConfigKey.TL_ASCEND_MEMORY_PLANNING: True,
}
```

AUTO_CV_COMBINE 和 AUTO_CV_SYNC 确保编译器正确处理纯 Vector 算子的核间调度和同步。

---

## 8. 验证方案

### 8.1 Golden 函数

```python
def golden_softmax(A):
    """基于 PyTorch 的参考实现"""
    import torch
    return torch.nn.functional.softmax(A, dim=-1)
```

### 8.2 测试用例

| 用例名 | 级别 | Shape | dtype | 说明 |
|--------|------|-------|-------|------|
| basic_small | Level 0 | (2, 32) | float32 | 最小功能验证（block_M=2, N=32） |
| typical_1 | Level 1 | (1024, 1024) | float32 | 典型配置 |
| typical_2 | Level 1 | (4096, 1024) | float32 | 多行典型配置 |
| boundary | Level 2 | (2, 2) | float32 | 极小 shape |
| large_scale | Level 3 | (8192, 4096) | float32 | 性能测试 |

### 8.3 精度标准

| dtype | atol | rtol |
|-------|------|------|
| float32 | 1e-4 | 1e-4 |

---

## 9. 风险点与注意事项

### 9.1 已知约束

- N 维必须整行加载到 UB，N > 8192（float32，含 broadcast buffer）时需改为 online softmax
- 当前实现假设 M 可被 block_M 整除，非整除场景需额外处理

### 9.2 常见错误

| 错误 | 触发场景 | 影响 | 解决方案 |
|------|----------|------|----------|
| UB 溢出 | N 过大（>8K float32 含 broadcast buffer） | 编译失败 | 采用 online softmax 分块迭代 |
| 精度偏差 | 未减 max 直接 exp | float32 在极端值下溢出 | 使用 safe softmax（先减 max） |
| shape 不匹配 | M 不可被 block_M 整除 | 边界行数据错误 | 添加尾块保护或使用 T.ceildiv |

### 9.3 特殊场景处理

- **极小 shape（M=1, N=1）**：仍可正常计算，block_M=2 时仅有 1 个有效行
- **大 N 场景**：需切换至 online softmax 模式，分块迭代计算 max 和 sum
- **混合精度**：若输入为 float16，建议在 UB 中用 float32 计算（T.tile.cast），输出再转回 float16

---

## 10. 交付清单

### 10.1 目录结构

```
examples/softmax/
├── example_softmax.py     # 算子实现 + 简单测试
├── design.md              # 本设计文档
```

### 10.2 文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `design.md` | 已完成 | 设计文档 |
| `example_softmax.py` | 已完成 | 算子实现 + 测试 |

### 10.3 命名规范

- 目录名: `softmax`
- 实现文件: `example_softmax.py`

### 10.4 实现顺序

1. ✅ 设计文档（design.md）
2. ✅ 算子实现（example_softmax.py）— Level 0 验证通过
3. ✅ 典型配置测试（Level 1）— (32,32), (1024,1024), (4096,4096), (8192,4096) 全部通过
4. ⬜ 边界测试（Level 2）
5. ⬜ 性能测试（Level 3，可选）