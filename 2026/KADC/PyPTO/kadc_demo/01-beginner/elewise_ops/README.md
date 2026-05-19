# add_abs 算子

## 概述

`add_abs` 是一个 PyPTO 逐元素（element-wise）动态算子，计算 $y = a + |b|$，即输入 `a` 与输入 `b` 的绝对值逐元素相加。

| 属性 | 值 |
|------|-----|
| 算子名称 | `add_abs` |
| 内核函数 | `add_abs_kernel` |
| 算子类型 | 逐元素（Element-wise） |
| 动态轴 | n 轴（支持运行时可变 n） |
| 源文件 | `add_abs.py`（算子 + 测试一体化） |

---

## 数学公式

$$ y = a + |b| $$

对于任意位置 $(i, j)$：

$$ y[i][j] = a[i][j] + |\ b[i][j]\ | $$

---

## 张量规格

| 参数 | 形状 | 数据类型 | 说明 |
|------|------|----------|------|
| 输入 `a` | `[n, d]` | `float32` | 第一输入张量 |
| 输入 `b` | `[n, d]` | `float32` | 第二输入张量（取绝对值后与 a 相加） |
| 输出 `y` | `[n, d]` | `float32` | 计算结果张量 |

- **n 轴**: 动态轴，支持运行时指定任意正整数值。
- **d 轴**: 固定维度（编译时由输入 shape 决定）。

---

## 精度标准

| 指标 | 值 |
|------|-----|
| 绝对容差 (atol) | `0.000025` |
| 相对容差 (rtol) | `0.005` |

---

## 实现细节

### 内核签名

```python
@pypto.frontend.jit(runtime_options={"run_mode": global_run_mode})
def add_abs_kernel(
    a: pypto.Tensor([], pypto.DT_FP32),
    b: pypto.Tensor([], pypto.DT_FP32),
    out: pypto.Tensor([], pypto.DT_FP32)
):
```

- **JIT 编译**: 使用 `pypto.frontend.jit` 装饰器自动编译。
- **动态形状**: 张量声明使用 `Tensor([], ...)` 表示动态 shape 支持。
- **Tile 配置**: `set_vec_tile_shapes(2, 8)` 设置向量 tile 形状。

### 核心计算

```python
# torch
  # y = a + torch.abs(b)

# pypto: 
  # tiling
  # operator
```

**涉及的 PyPTO API**：
| API | 用途 | 约束 |
|-----|------|------|
| `pypto.set_vec_tile_shapes` | 设置基本块tiling | 尾轴必须32B对齐 |
| `pypto.abs` | 计算张量 `b` 的逐元素绝对值 | |
| `pypto.add` | 将 `a` 与 `abs(b)` 逐元素相加 | |

**API 使用指导**：
  https://gitcode.com/cann/pypto/blob/master/docs/api/config/pypto-set_vec_tile_shapes.md
  https://gitcode.com/cann/pypto/blob/master/docs/api/operation/pypto-abs.md
  https://gitcode.com/cann/pypto/blob/master/docs/api/operation/pypto-add.md

### 运行模式

- **NPU 模式** (`--run_mode npu`): 在 Ascend NPU 上真实执行，输出实际计算结果并进行精度校验。
- **SIM 模式** (`--run_mode sim`): 编译 + 代价模型仿真，不执行实际计算（输出为未初始化内存），常用于编译验证和性能预估算。

---

## 快速开始

### 环境准备

```bash
# 设置 NPU 设备 ID
export TILE_FWK_DEVICE_ID=0

# 安装依赖（如未安装）
pip install pypto torch numpy
```

### 运行全部测试（NPU 模式）

```bash
python add_abs.py
```

### 运行仿真模式

```bash
python add_abs.py --run_mode sim
```

### 列出所有可用测试

```bash
python add_abs.py --list
```

### 运行指定测试

```bash
python add_abs.py add_abs::test_add_abs_basic
python add_abs.py add_abs::test_add_abs_dynamic_n
python add_abs.py add_abs::test_add_abs_edge_cases
```

---

## 测试用例

| 测试 ID | 名称 | 说明 |
|---------|------|------|
| `add_abs::test_add_abs_basic` | 基础功能验证 | 固定小张量 `[2, 2]` 验证 $y = a + \|b\|$ 的正确性 |
| `add_abs::test_add_abs_dynamic_n` | 动态 n 轴验证 | 测试 `n ∈ {3, 7, 15}` 运行时动态 shape，验证算子对不同 n 值的正确性 |
| `add_abs::test_add_abs_edge_cases` | 边界场景验证 | 覆盖全零输入、b 全部为负数、大数值等边界情况 |

### 预期输出示例（NPU 模式）

```
============================================================
PyPTO add_abs Operator Tests
============================================================

============================================================
Test: Basic Usage of add_abs Operator
============================================================
Input a:    tensor([[ 1., -2.], [ 3., -4.]])
Input b:    tensor([[-2.,  3.], [-4.,  5.]])
Output:     tensor([[3., 1.], [7., 1.]])
Expected:   tensor([[3., 1.], [7., 1.]])
✓ Basic usage of add_abs operator completed successfully

============================================================
Test: add_abs Operator - Dynamic n-axis
============================================================
  n=  3, d=4: max_diff=0.00000000  ✓
  n=  7, d=4: max_diff=0.00000000  ✓
  n= 15, d=4: max_diff=0.00000000  ✓
✓ Dynamic n-axis test completed successfully

============================================================
Test: add_abs Operator - Edge Cases
============================================================
  [zeros]   a=zeros, b=zeros     ✓
  [neg_b]   a=ones,  b=-3        ✓
  [large]   a=[100,-200], b=[-50,150]  ✓
✓ Edge cases test completed successfully

============================================================
All add_abs tests passed!
============================================================
```

---

## 文件结构

```
kadc_test/01-beginner/elewise_ops/
├── add_abs.md          # 需求规格文档
├── add_abs.py          # 算子实现 + 测试用例（唯一文件）
├── ReadMe.md           # 本文档
└── __pycache__/        # Python 字节码缓存
```

---

## 编译产物

运行后在项目根 `output/` 目录下生成性能分析数据：
- `bubble_analysis.log` — 气泡分析报告
- `merged_swimlane.json` — 泳道图数据（可在 [ui.perfetto.dev](https://ui.perfetto.dev) 或 vscode插件 PyPTO ToolKits 打开）

---

## 已知限制

1. **SIM 模式**: 不会产生实际计算结果（仅编译验证和代价模型仿真），精度校验自动跳过。
2. **NPU 环境依赖**: 需要正确安装 CANN 及相关库（`libhccl.so`、`libc_sec.so`、`torch_npu`）。
3. **Tile 配置**: 当前使用固定 `set_vec_tile_shapes(2, 8)`，尾轴需要32B对齐。
