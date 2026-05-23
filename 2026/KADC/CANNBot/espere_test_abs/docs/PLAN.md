# abs 算子开发计划

> 本文档在开发流程中持续更新。

---

## 1. 需求概述

| 项目 | 内容 |
|-----|------|
| 算子名称 | `abs` |
| 数学公式 | y[i] = \|x[i]\|, 逐元素取绝对值 |
| 输入 | x: shape=[1,128]/[4,2048]/[32,4096], dtype=float16 |
| 输出 | y: shape=[1,128]/[4,2048]/[32,4096], dtype=float16 |
| 算子类别 | Elementwise（逐元素一元运算） |
| 需求类型 | 特定用例（shape 和 dtype 已明确） |

### 关键技术选择

| 决策项 | 选择 | 理由 |
|-------|------|------|
| 技术路线 | 通用 SIMD/MemBase（TPipe + TQue） | DAV_2201 (Ascend910B3) 非 DAV_3510 |
| 计算 API | `Abs< half >(dst, src, count)` Level 2 | 连续 1D 数据，最简单的批量接口 |
| 数据搬运 | `DataCopyPad` | 自动处理非对齐边界 |
| 精度策略 | 原 dtype 直算 | Abs 无需升精度 |

---

## 2. 文件清单

| 文件 | 状态 |
|------|------|
| `op_kernel/abs_tiling.h` — Tiling 结构体（kernel/host 共用） | ⬜ |
| `op_kernel/abs_kernel.asc` — Kernel 计算逻辑（KernelAbs 类 + 核函数入口） | ⬜ |
| `op_host/abs.asc` — Host + main 入口（包含 kernel_operator.h 和 abs_kernel.asc） | ⬜ |
| `op_host/data_utils.h` — 数据读写工具 | ⬜ |
| `op_extension/abs_torch.cpp` — PyTorch host（Tiling 计算 + kernel launch） | ⬜ |
| `op_extension/register.cpp` + `op_extension/ops.h` — TORCH_LIBRARY 注册 | ⬜ |
| `CMakeLists.txt` — 双 target（可执行文件 + libabs_ops.so） | ⬜ |
| `run.sh` — 一键运行脚本 | ⬜ |
| `scripts/gen_data.py` — 测试数据生成 | ⬜ |
| `scripts/golden.py` — Golden 计算函数 | ⬜ |
| `scripts/verify_result.py` — 直调通路精度验证 | ⬜ |
| `scripts/test_torch.py` — PyTorch 通路测试 | ⬜ |

---

## 3. 测试计划

精度标准：float16 社区标准 MERE < 2^-10 (≈0.000977), MARE < 10 × 2^-10 (≈0.00977)

**Golden 计算**：定义在 `scripts/golden.py` 中，使用 NumPy `np.abs()` 作为 golden 参考。gen_data.py 和 test_torch.py 共用。

**用例（T=可执行文件, P=PyTorch, 1:1 对应）**：

| 编号 | 用例 | 输入 shape | 输入数据 | 预期输出 |
|-----|------|-----------|---------|---------|
| T1/P1 | 正负混合随机数 | [1,128] | `np.random.randn(1,128).astype(np.float16)` | `np.abs(x)` |
| T2/P2 | 正负混合随机数 + 零值 | [4,2048] | 含正负随机数 + 部分零值 | `np.abs(x)` |
| T3/P3 | 正负混合随机数 + 边界值 | [32,4096] | 含正负随机数 + FP16 特殊值 (±65504, ±0) | `np.abs(x)` |
| T4/P4 | 全零值 | [1,128] | `np.zeros((1,128), dtype=np.float16)` | `np.abs(x)` |
| T5/P5 | 全正值 | [4,2048] | `np.abs(np.random.randn(4,2048)).astype(np.float16)` | x（不变） |
| T6/P6 | 全负值 | [32,4096] | `-np.abs(np.random.randn(32,4096)).astype(np.float16)` | `-x` |

---

## 4. 开发进度

| 阶段 | 检查项 | 状态 |
|------|--------|------|
| 框架搭建 | 工程创建 + CMake 双 target + 空 Kernel 编译通过 | ⬜ |
| Kernel 实现 | TilingData + Host Tiling + Kernel Compute + 编译通过 | ⬜ |
| 可执行文件验证 | T1-T6 全部通过 | ⬜ |
| PyTorch 验证 | TORCH_LIBRARY 注册 + `torch.ops.npu.abs()` 可调用 + P1-P6 全部通过 | ⬜ |
| 性能验收 | msprof 采集 + 数据归档 + 达标判定 | ⬜ |

---

## 5. 已知问题和决策记录

| 日期 | 问题/决策 | 说明 |
|------|----------|------|
| 2026-05-23 | 技术路线决策 | DAV_2201 通用 SIMD/MemBase + TPipe/TQue 流水线 |
| 2026-05-23 | API 选择 | `Abs<half>` Level 2 批量接口 + `DataCopyPad` |
| 2026-05-23 | 精度策略 | 原 dtype 直算，无需升精度 |

---

## 6. 测试结果

### 6.1 可执行文件通路

**状态**: ⬜ | **脚本**: `run.sh` + `scripts/verify_result.py`

| 编号 | 结果 | Max Diff |
|-----|------|----------|
| T1 | ⬜ | |
| T2 | ⬜ | |
| T3 | ⬜ | |
| T4 | ⬜ | |
| T5 | ⬜ | |
| T6 | ⬜ | |

### 6.2 PyTorch 通路

**状态**: ⬜ | **脚本**: `scripts/test_torch.py` | **约束**: 与 §6.1 逐行对应，相同输入和 golden

| 编号 | 结果 | Max Diff |
|-----|------|----------|
| P1 | ⬜ | |
| P2 | ⬜ | |
| P3 | ⬜ | |
| P4 | ⬜ | |
| P5 | ⬜ | |
| P6 | ⬜ | |

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

**状态**: ⬜ | **数据**: `docs/perf/round_NNN/`

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
| 可执行文件 | 6 | | | ⬜ |
| PyTorch | 6 | | | ⬜ |
| 性能 | 3 shape | | | ⬜ |
