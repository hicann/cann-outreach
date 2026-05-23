# online_safe_softmax 算子开发计划

> 本文档在开发流程中持续更新。

---

## 1. 需求概述

| 项目 | 内容 |
|-----|------|
| 算子名称 | online_safe_softmax |
| 数学公式 | y[i,j] = exp(x[i,j] - max(x[i,:])) / sum(exp(x[i,:] - max(x[i,:]))) |
| 输入 | x: shape=[A1, R], dtype=float32 |
| 输出 | y: shape=[A1, R], dtype=float32 |
| 算子类别 | Reduction (AR 模式) |
| 需求类型 | 通用 |

---

## 2. 文件清单

| 文件 | 状态 |
|------|------|
| `kernel/online_safe_softmax_tiling.h` — Tiling 结构体（kernel/host 共用） | ⬜ |
| `kernel/online_safe_softmax_kernel.asc` — Kernel 计算逻辑 | ⬜ |
| `host/online_safe_softmax.asc` — Host + main 入口 | ⬜ |
| `torch_library/online_safe_softmax_torch.cpp` — PyTorch host（Tiling + launch） | ⬜ |
| `torch_library/register.cpp` + `torch_library/ops.h` — TORCH_LIBRARY 注册 | ⬜ |
| `CMakeLists.txt` — 双 target（可执行文件 + .so） | ⬜ |
| `run.sh` + `scripts/gen_data.py` + `scripts/verify_result.py` | ⬜ |
| `scripts/test_torch.py` — PyTorch 通路测试 | ⬜ |

---

## 3. 测试计划

精度标准：FP32 atol=1e-5, rtol=1e-5 (社区标准 threshold=2^-13 ≈ 0.000122)

**Golden 计算**：定义在 `scripts/golden.py` 中，gen_data.py 和 test_torch.py 共用。

**Golden 实现**：
```python
def online_safe_softmax_golden(x):
    # x: numpy array, shape [A1, R]
    # online safe softmax: subtract max per row for numerical stability
    max_vals = np.max(x, axis=-1, keepdims=True)
    shifted = x - max_vals
    exp_vals = np.exp(shifted)
    sum_vals = np.sum(exp_vals, axis=-1, keepdims=True)
    return exp_vals / sum_vals
```

**用例（T=可执行文件, P=PyTorch, 1:1 对应）**：

| 编号 | 用例 | 输入 | 预期输出 | 覆盖场景 |
|-----|------|------|---------|---------|
| T1/P1 | 随机数据 小shape | randn(8, 128), FP32 | softmax golden | FullLoad, 32B 对齐 |
| T2/P2 | 随机数据 大shape | randn(4, 4096), FP32 | softmax golden | FullLoad, 大行 |
| T3/P3 | 随机数据 超大shape | randn(2, 16384), FP32 | softmax golden | ColSplit Online |
| T4/P4 | 零值 | zeros(4, 128), FP32 | 均值 1/128 | 边界值 |
| T5/P5 | 极大极小混合 | 含 1e5 和 -1e5 的数据 | softmax golden | 数值稳定性 |
| T6/P6 | 非对齐行长度 | randn(8, 97), FP32 | softmax golden | 非 32B 对齐 |
| T7/P7 | 最小行长度 | randn(4, 8), FP32 | softmax golden | 最小 R |

---

## 4. 开发进度

| 阶段 | 检查项 | 状态 |
|------|--------|------|
| 框架搭建 | 工程创建 + CMake 双 target + 空 Kernel 编译通过 | ⬜ |
| Kernel 实现 | TilingData + Host Tiling + Kernel Compute (FullLoad + ColSplit) + 编译通过 | ⬜ |
| 可执行文件验证 | T1-T7 全部通过 | ⬜ |
| PyTorch 验证 | TORCH_LIBRARY 注册 + `torch.ops.npu.online_safe_softmax()` 可调用 + P1-P7 全部通过 | ⬜ |
| 性能验收 | msprof 采集 + 数据归档 + 达标判定 | ⬜ |

---

## 5. 已知问题和决策记录

| 日期 | 问题/决策 | 说明 |
|------|----------|------|
| 2026-05-23 | Subs/Divs 在 DAV_2201 不支持 | 决策：使用 Adds(-val) 替代 Subs，Muls(1/val) 替代 Divs |
| 2026-05-23 | ReduceSum Level 2 仅支持 float | 决策：统一使用 FP32 计算，不支持 FP16 ReduceSum Level 2 |
| 2026-05-23 | ColSplit Online 需两次遍历 | 决策：Phase1 累积 max/sum，Phase2 计算输出；两次遍历不可避免但内存开销极小（仅维护两个标量） |

---

## 6. 测试结果

### 6.1 可执行文件通路

**状态**: ⬜ | **脚本**: run.sh + scripts/verify_result.py

| 编号 | 结果 | Max Diff |
|-----|------|----------|
| T1 | ⬜ | |
| T2 | ⬜ | |
| T3 | ⬜ | |
| T4 | ⬜ | |
| T5 | ⬜ | |
| T6 | ⬜ | |
| T7 | ⬜ | |

### 6.2 PyTorch 通路

**状态**: ⬜ | **脚本**: scripts/test_torch.py | **约束**: 与 §6.1 逐行对应，相同输入和 golden

| 编号 | 结果 | Max Diff |
|-----|------|----------|
| P1 | ⬜ | |
| P2 | ⬜ | |
| P3 | ⬜ | |
| P4 | ⬜ | |
| P5 | ⬜ | |
| P6 | ⬜ | |
| P7 | ⬜ | |

### 6.3 产物 & 执行状态

- [ ] `build/online_safe_softmax` 可执行文件存在
- [ ] `build/libonline_safe_softmax_ops.so` 存在
- [ ] `torch.ops.load_library` + `torch.ops.npu.online_safe_softmax` 可调用

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
| 可执行文件 | 7 | | | ⬜ |
| PyTorch | 7 | | | ⬜ |
| 性能 | | | | ⬜ |