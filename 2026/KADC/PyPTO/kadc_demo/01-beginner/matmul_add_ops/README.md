# matmul_add 算子

## 概述

matmul_add 算子实现矩阵乘法与偏置加法的融合操作，计算公式为：

$$y(a, b, c) = a \times b^T + c$$

该算子基于 PyPTO 框架开发，运行于华为昇腾 AI 处理器，支持 **m 轴动态**（`pypto.DYNAMIC`），即运行时可在不重新编译的前提下更换 m 维度大小。

## 算子规格

| 项目 | 说明 |
|------|------|
| 算子名称 | matmul_add |
| 计算公式 | `y = a @ b^T + c` |
| 数据类型 | bfloat16 |
| 精度标准 | atol=0.0001, rtol=0.0078125 |
| 动态轴 | m（第一维度） |

### 输入输出

| 参数 | 方向 | shape | dtype | 说明 |
|------|------|-------|-------|------|
| a | 输入 | [m, k] | bfloat16 | 左矩阵，m 为动态轴 |
| b | 输入 | [n, k] | bfloat16 | 右矩阵，计算时转置 |
| c | 输入 | [m, n] | bfloat16 | 偏置矩阵，m 为动态轴 |
| y | 输出 | [m, n] | bfloat16 | 计算结果 |

## 文件结构

```
kadc_test/01-beginner/matmul_add_ops/
├── matmul_add.py             # 算子 kernel 实现 + 测试用例
├── matmul_add_in_agent.md    # 原始需求描述
└── README.md                 # 本文档
```

## 实现要点

### 核心代码结构

```python
@pypto.frontend.jit(runtime_options={"run_mode": global_run_mode})
def matmul_add_kernel(
    a: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16),  # m 动态
    b: pypto.Tensor([], pypto.DT_BF16),                              # shape 不限定
    c: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16),  # m 动态
    out: pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], pypto.DT_BF16),# m 动态
    n_val: int,                                                       # n 维度值
    k_val: int):                                                      # k 维度值
```

### 关键设计决策

#### 1. 动态轴声明

使用 `pypto.DYNAMIC` 标记 m 轴，使其可在不同调用间变化而无需重新编译。由于 DYNAMIC 张量不能直接作为算子操作数，必须在 `pypto.loop` 内通过 `pypto.view` 转为静态 shape 的 tile 后再计算。

#### 2. 仅对动态轴手动切分

只对动态的 m 轴进行手动 tiling（`pypto.loop` + `pypto.view` + `pypto.assemble`），静态的 n、k 轴由框架内部处理：

```python
for m_idx in pypto.loop(0, m_loop, 1, name="LOOP_m", idx_name="m_idx"):
    m_offset = m_idx * tile_m
    valid_m = (m_val - m_offset).min(tile_m)   # 符号表达式边界管理

    # view a, 为处理非对齐场景，需要手动设置validshape

    # view c, 为处理非对齐场景，需要手动设置validshape

    # mm_res = a @ b.T

    # add_result = mm_res + c_view

    pypto.assemble(add_result, [m_offset, 0], out)
```

**涉及的 PyPTO API**

| API | 用途 |
|-----|------|
| `pypto.frontend.jit` | Kernel JIT 编译装饰器 |
| `pypto.Tensor([pypto.DYNAMIC, pypto.STATIC], ...)` | 声明动态/静态维度 |
| `pypto.loop` | 生成硬件级循环 |
| `pypto.view` | 从动态张量中切分静态 shape 的 tile |
| `pypto.assemble` | 将计算结果写回输出张量 |
| `pypto.set_cube_tile_shapes` | 设置矩阵乘法的 L0/L1 cache tile 尺寸 |
| `pypto.set_vec_tile_shapes` | 设置向量操作的 tile 尺寸 |
| `pypto.matmul(..., b_trans=True)` | 矩阵乘法，支持随路转置 |
| `pypto.add` | 元素级加法 |

参考文档：
https://gitcode.com/cann/pypto/blob/master/docs/api/operation/pypto-view.md
https://gitcode.com/cann/pypto/blob/master/docs/api/config/pypto-set_cube_tile_shapes.md
https://gitcode.com/cann/pypto/blob/master/docs/api/operation/pypto-matmul.md
https://gitcode.com/cann/pypto/blob/master/docs/api/config/pypto-set_vec_tile_shapes.md
https://gitcode.com/cann/pypto/blob/master/docs/api/operation/pypto-add.md


#### 3. b^T 通过 matmul 随路转置实现

使用 `pypto.matmul()` 中的 `b_trans=True` 参数，在矩阵乘法时随路完成 b 的转置，避免额外的数据搬移开销。

#### 4. Tile Shape 对齐

BF16 数据类型要求 `set_cube_tile_shapes` 的所有维度满足 **16 元素对齐**（即 32 字节 / 2 字节每元素）。当实际 k 或 n 不满足对齐时，向上取整：

```python
tile_k = ((k_val + 15) // 16) * 16
tile_n = ((n_val + 15) // 16) * 16
```

#### 5. 边界管理使用 `.min()` 方法

对于动态维度产生的符号表达式，**必须使用 `.min()` 方法**而非 Python 内置 `min()` 函数：

```python
# 正确：符号表达式的 .min() 方法
valid_m = (m_val - m_offset).min(tile_m)

# 错误：Python min() 无法处理符号表达式
# valid_m = min(m_offset + tile_m, m_val)
```

### DYNAMIC Loop 注意事项

PyPTO 的 DYNAMIC loop 在首次调用时确定循环次数并编译，后续调用：

- **可以**使用更少的迭代次数（通过 `valid_shape` 边界管理跳过多余迭代）
- **不可以**使用更多的迭代次数（超出编译时的循环次数）

因此，**首次调用应使用最大的 m 值**，以确保编译的循环次数覆盖后续所有场景。

## 环境要求

- CANN 工具链（Ascend 950PR / Atlas A2 / Atlas A3）
- PyPTO 框架
- PyTorch + torch_npu
- 环境变量 `TILE_FWK_DEVICE_ID` 已设置（如 `export TILE_FWK_DEVICE_ID=0`）

## 运行方式

```bash
# 设置设备 ID
export TILE_FWK_DEVICE_ID=0

# 运行全部测试
python matmul_add.py

# 列出所有测试用例
python matmul_add.py --list

# 运行指定测试用例
python matmul_add.py matmul_add::test_matmul_add_basic

# 仿真模式运行
python matmul_add.py --run_mode sim
```

## 测试用例

| 测试用例 | 说明 | 覆盖场景 |
|----------|------|----------|
| `test_matmul_add_basic` | 4x4 矩阵基础验证（b 为单位矩阵） | 基本功能正确性 |
| `test_matmul_add_dynamic_m` | m=128/64/32/16 动态轴验证（从大到小） | 动态 m 轴、多轮调用 |
| `test_matmul_add_non_square` | m=7, n=5, k=11 非对齐非方阵 | 非 16 对齐维度 |
| `test_matmul_add_edge_cases` | 全零输入 / 单行(m=1) / 大数值 | 边界值场景 |

### 上板验证结果

在 Ascend 910 NPU 上全部通过，所有测试用例 max_diff=0.00000000。
