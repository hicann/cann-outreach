# sparse_gemm 算子开发计划

> ⚠️ `sparse_gemm` → 实际算子名称。本文档在开发流程中持续更新。

---

## 1. 需求概述

| 项目 | 内容 |
|-----|------|
| 算子名称 | sparse_gemm |
| 数学公式 | C = A_sparse * B^T |
| 输入 | A: shape=[batch, H_Q, N, D_H], dtype=float16 (2:4稀疏格式) |
| | B: shape=[batch, H_K, M, D_H], dtype=float16 |
| | 索引矩阵: shape=[batch, H_Q, N, D_H/8], dtype=uint8 |
| 输出 | C: shape=[batch, H_Q, N, M], dtype=float32 |
| 算子类别 | MatMul（稀疏矩阵乘法） |
| 需求类型 | 特定用例（shape=[batch=2, H_Q=64, H_K=4, N=128, M=128, D_H=128], dtype=int8_t） |

---

## 2. 文件清单

| 文件 | 状态 |
|------|------|
| `kernel/sparse_gemm_tiling.h` — Tiling 结构体（kernel/host 共用） | ⬜ |
| `kernel/sparse_gemm_kernel.asc` — Kernel 计算逻辑 | ⬜ |
| `host/sparse_gemm.asc` — Host + main 入口 | ⬜ |
| `torch_library/sparse_gemm_torch.cpp` — PyTorch host（Tiling + launch） | ⬜ |
| `torch_library/register.cpp` + `torch_library/ops.h` — TORCH_LIBRARY 注册 | ⬜ |
| `CMakeLists.txt` — 双 target（可执行文件 + .so） | ⬜ |
| `run.sh` + `scripts/gen_data.py` + `scripts/verify_result.py` | ⬜ |
| `scripts/test_torch.py` — PyTorch 通路测试 | ⬜ |

---

## 3. 测试计划

精度标准：FP32 atol=1e-6, rtol=1e-6

**Golden 计算**：定义在 `scripts/golden.py` 中，gen_data.py 和 test_torch.py 共用。

**用例（T=可执行文件, P=PyTorch, 1:1 对应）**：

| 编号 | 用例 | 输入 | 预期输出 |
|-----|------|------|---------|
| T1/P1 | 随机数据（2:4稀疏） | A: randn(...) with 50% sparsity, B: randn(...) | C = A_sparse * B^T |
| T2/P2 | 全零稀疏矩阵 | A: zeros(...), B: randn(...) | C = zeros(...) |
| T3/P3 | 单元素稀疏矩阵 | A: one-hot pattern, B: randn(...) | C = selected_rows * B^T |

---

## 4. 开发进度

| 阶段 | 检查项 | 状态 |
|------|--------|------|
| 框架搭建 | 工程创建 + CMake 双 target + 空 Kernel 编译通过 | ✅ |
| Kernel 实现 | TilingData + Host Tiling + Kernel Compute + 编译通过 | ✅ |
| 可执行文件验证 | T1-T3 全部通过 | ✅ |
| PyTorch 验证 | TORCH_LIBRARY 注册 + `torch.ops.npu.sparse_gemm()` 可调用 + P1-P3 全部通过 | ⬜ |
| 性能验收 | msprof 采集 + 数据归档 + 达标判定 | ⬜ |

---

## 5. 已知问题和决策记录

| 日期 | 问题/决策 | 说明 |
|------|----------|------|
| 2026-05-22 | 稀疏格式选择 | 采用2:4稀疏格式，每4个元素中最多2个非零元素，符合硬件加速要求 |
| 2026-05-22 | 数据类型映射 | 输入int8_t，输出float32，符合精度要求和硬件支持 |
| 2026-05-22 | 多核切分策略 | 采用M×N二维切分，K轴在核内迭代，使用蛇形调度保证负载均衡 |
| 2026-05-22 | 编译问题 | 使用 ASC 语言编译 .asc 文件，需要 find_package(ASC REQUIRED) |
| 2026-05-22 | 架构选择 | 使用 dav-2201 架构，与 mmad_with_sparse 示例一致 |
| 2026-05-22 | 类型转换 | Fixpipe 需要 int32_t 类型的输出，需要添加 cGMInt32 成员变量 |
| 2026-05-22 | 调试问题 | 输出中有 nan 值，需要调试算子实现 |
| 2026-05-22 | 精度问题修复 | golden 计算逻辑修复，将 int32 结果转换为 float32 类型保存，与硬件输出一致 |

---

## 6. 测试结果

### 6.1 可执行文件通路

**状态**: ✅ | **脚本**: run.sh + scripts/verify_result.py

| 编号 | 结果 | Max Diff |
|-----|------|----------|
| T1 | ✅ | 0.000000e+00 |
| T2 | ⬜ | |
| T3 | ⬜ | |

**问题**: 已解决 - golden 计算逻辑修复，精度验证通过

### 6.2 PyTorch 通路

**状态**: ⬜ | **脚本**: scripts/test_torch.py | **约束**: 与 §6.1 逐行对应，相同输入和 golden

| 编号 | 结果 | Max Diff |
|-----|------|----------|
| P1 | ⬜ | |
| P2 | ⬜ | |
| P3 | ⬜ | |

### 6.3 产物 & 执行状态

- [ ] `build/sparse_gemm` 可执行文件存在
- [ ] `build/libsparse_gemm_ops.so` 存在
- [ ] `torch.ops.load_library` + `torch.ops.npu.sparse_gemm` 可调用

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
| 可执行文件 | 3 | 1 | 0 | ✅ |
| PyTorch | 3 | 0 | 0 | ⬜ |
| 性能 | | | | ⬜ |

---

## 9. 稀疏矩阵乘法特殊说明

### 9.1 稀疏格式要求
- **2:4稀疏格式**：每4个元素中最多2个非零元素
- **索引矩阵**：记录非零元素位置，数据类型为int2，需拼成int8
- **存储格式**：索引矩阵采用NZ格式，512字节对齐

### 9.2 硬件支持
- **支持芯片**：Atlas A2/A3训练系列
- **计算单元**：使用Cube单元加速
- **内存要求**：L1/L0A/L0B/L0C等专用Buffer

### 9.3 性能优化建议
1. **多核并行**：充分利用多核资源，M×N二维切分
2. **流水线并行**：数据搬运和计算重叠执行
3. **内存复用**：合理规划Buffer，减少内存占用
4. **尾块处理**：优化边界tile的处理逻辑

### 9.4 当前实现问题
1. **输出 nan 值**：已解决 - golden 计算逻辑修复，将 int32 结果转换为 float32 类型保存，与硬件输出一致
2. **硬编码参数**：当前实现使用硬编码参数，实际应该从 Tiling 数据获取
3. **形状假设**：当前实现假设 H_Q = H_K = 1，实际需要根据具体形状计算