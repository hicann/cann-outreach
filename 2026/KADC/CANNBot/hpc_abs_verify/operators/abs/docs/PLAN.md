# abs 算子开发计划

---

## 1. 需求概述

| 项目 | 内容 |
|-----|------|
| 算子名称 | abs |
| 数学公式 | y = |x| |
| 输入 | x: shape=[任意], dtype=half/float |
| 输出 | y: shape=[与x相同], dtype=half/float |
| 算子类别 | Elementwise |
| 需求类型 | 通用 |

---

## 2. 文件清单

| 文件 | 状态 |
|------|------|
| `kernel/abs_tiling.h` — Tiling 结构体（kernel/host 共用） | ⬜ |
| `kernel/abs_kernel.asc` — Kernel 计算逻辑 | ⬜ |
| `host/abs.asc` — Host + main 入口 | ⬜ |
| `torch_library/abs_torch.cpp` — PyTorch host（Tiling + launch） | ⬜ |
| `torch_library/register.cpp` + `torch_library/ops.h` — TORCH_LIBRARY 注册 | ⬜ |
| `CMakeLists.txt` — 双 target（可执行文件 + .so） | ⬜ |
| `run.sh` + `scripts/gen_data.py` + `scripts/verify_result.py` | ⬜ |
| `scripts/test_torch.py` — PyTorch 通路测试 | ⬜ |

---

## 3. 测试计划

### 3.1 精度标准

**社区标准（浮点计算类）**：

| 数据类型 | Threshold | 数值 |
|---------|-----------|------|
| FLOAT16 | 2^-10 | 约0.000977 |
| FLOAT32 | 2^-13 | 约0.000122 |

**判定条件**：
- MERE < Threshold
- MARE < 10 * Threshold

**误差指标定义**：
```
MERE = avg(|actual - golden| / (|golden| + 1e-7))
MARE = max(|actual - golden| / (|golden| + 1e-7))
```

### 3.2 Golden 计算

定义在 `scripts/golden.py` 中：

```python
import torch

def abs_golden(x):
    """计算绝对值的 golden 结果"""
    return torch.abs(x)
```

### 3.3 测试用例

**用例设计（T=可执行文件, P=PyTorch, 1:1 对应）**：

| 编号 | 用例 | 输入 shape | dtype | 预期输出 | 说明 |
|-----|------|-----------|-------|---------|------|
| T1/P1 | 随机数据 FP16 | [1, 1024] | half | y=|x| | 基础验证 |
| T2/P2 | 随机数据 FP32 | [1, 1024] | float | y=|x| | 基础验证 |
| T3/P3 | 正负混合 FP16 | [1, 1024] | half | y=|x| | 正负数混合 |
| T4/P4 | 正负混合 FP32 | [1, 1024] | float | y=|x| | 正负数混合 |
| T5/P5 | 零值 FP16 | [1, 1024] | half | y=0 | 零输入边界 |
| T6/P6 | 零值 FP32 | [1, 1024] | float | y=0 | 零输入边界 |
| T7/P7 | 大 shape FP16 | [1024, 1024] | half | y=|x| | 多核切分验证 |
| T8/P8 | 大 shape FP32 | [1024, 1024] | float | y=|x| | 多核切分验证 |
| T9/P9 | 小 shape FP16 | [1, 64] | half | y=|x| | 单核/小数据量 |
| T10/P10 | 小 shape FP32 | [1, 64] | float | y=|x| | 单核/小数据量 |
| T11/P11 | 非对齐 shape FP16 | [1, 513] | half | y=|x| | 非对齐边界 |
| T12/P12 | 非对齐 shape FP32 | [1, 513] | float | y=|x| | 非对齐边界 |

---

## 4. 开发进度

| 阶段 | 检查项 | 状态 |
|------|--------|------|
| 框架搭建 | 工程创建 + CMake 双 target + 空 Kernel 编译通过 | ⬜ |
| Kernel 实现 | TilingData + Host Tiling + Kernel Compute + 编译通过 | ⬜ |
| 可执行文件验证 | T1-T12 全部通过 | ⬜ |
| PyTorch 验证 | TORCH_LIBRARY 注册 + `torch.ops.npu.abs()` 可调用 + P1-P12 全部通过 | ⬜ |
| 性能验收 | msprof 采集 + 数据归档 + 达标判定 | ⬜ |

---

## 5. 已知问题和决策记录

| 日期 | 问题/决策 | 说明 |
|------|----------|------|
| 2026-05-23 | 技术路线决策 | 采用通用 SIMD/MemBase 路线，使用 Memory 矢量计算 Abs API |
| 2026-05-23 | 精度标准决策 | 使用社区标准（浮点计算类），FP16: 2^-10, FP32: 2^-13 |

---

## 6. 测试结果

### 6.1 可执行文件通路

**状态**: ⬜ | **脚本**: run.sh + scripts/verify_result.py

| 编号 | 结果 | MERE | MARE | Max Diff |
|-----|------|------|------|----------|
| T1 | ⬜ | | | |
| T2 | ⬜ | | | |
| T3 | ⬜ | | | |
| T4 | ⬜ | | | |
| T5 | ⬜ | | | |
| T6 | ⬜ | | | |
| T7 | ⬜ | | | |
| T8 | ⬜ | | | |
| T9 | ⬜ | | | |
| T10 | ⬜ | | | |
| T11 | ⬜ | | | |
| T12 | ⬜ | | | |

### 6.2 PyTorch 通路

**状态**: ⬜ | **脚本**: scripts/test_torch.py | **约束**: 与 §6.1 逐行对应，相同输入和 golden

| 编号 | 结果 | MERE | MARE | Max Diff |
|-----|------|------|------|----------|
| P1 | ⬜ | | | |
| P2 | ⬜ | | | |
| P3 | ⬜ | | | |
| P4 | ⬜ | | | |
| P5 | ⬜ | | | |
| P6 | ⬜ | | | |
| P7 | ⬜ | | | |
| P8 | ⬜ | | | |
| P9 | ⬜ | | | |
| P10 | ⬜ | | | |
| P11 | ⬜ | | | |
| P12 | ⬜ | | | |

### 6.3 产物 & 执行状态

- [ ] `build/abs` 可执行文件存在
- [ ] `build/libabs_ops.so` 存在
- [ ] `torch.ops.load_library` + `torch.ops.npu.abs` 可调用

| 通路 | 状态 | 运行时间 | 跳过原因 |
|------|------|---------|---------|
| 可执行文件 | ⬜ | | |
| PyTorch | ⬜ | | |

---

## 7. 性能验收

**状态**: ⬜ | **数据**: docs/perf/round_NNN/

| 指标 | 值 | 判定 |
|------|------|------|
| Task Duration | | |
| Block Dim | | |
| 主导流水 | | |

**达标判定**: ⬜ | **理由**:

---

## 8. 汇总

| 通路 | 用例数 | 通过 | 失败 | 状态 |
|------|--------|------|------|------|
| 可执行文件 | 12 | | | ⬜ |
| PyTorch | 12 | | | ⬜ |
| 性能 | - | | | ⬜ |