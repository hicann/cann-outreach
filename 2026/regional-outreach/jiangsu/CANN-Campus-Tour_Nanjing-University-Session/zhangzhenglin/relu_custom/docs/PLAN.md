# relu_custom 算子开发计划

---

## 1. 需求概述

| 项目 | 内容 |
|-----|------|
| 算子名称 | relu_custom |
| 数学公式 | y = max(0, x) |
| 输入 | x: shape=[N4,N3,N2,N1], dtype=float16, Format=ND |
| 输出 | y: shape=[N4,N3,N2,N1], dtype=float16 |
| 算子类别 | Elementwise（一元激活函数） |
| 需求类型 | 特定用例（4D shape, float16, ND） |

---

## 2. 文件清单

| 文件 | 状态 |
|------|------|
| `kernel/relu_custom_tiling.h` — Tiling 结构体 | ⬜ |
| `kernel/relu_custom_kernel.asc` — Kernel 计算逻辑 | ⬜ |
| `host/relu_custom.asc` — Host + main 入口 | ⬜ |
| `torch_library/relu_custom_torch.cpp` — PyTorch host | ⬜ |
| `torch_library/register.cpp` + `torch_library/ops.h` — TORCH_LIBRARY 注册 | ⬜ |
| `CMakeLists.txt` — 双 target | ⬜ |
| `run.sh` + `scripts/gen_data.py` + `scripts/verify_result.py` | ⬜ |
| `scripts/golden.py` — Golden 计算 | ⬜ |
| `scripts/test_torch.py` — PyTorch 通路测试 | ⬜ |

---

## 3. 测试计划

精度标准：float16 atol=1e-3, rtol=1e-3（与 PyTorch `torch.relu` 对比）

**Golden 计算**：定义在 `scripts/golden.py` 中。

**用例（4D shape, ND 格式, float16）**：

| 编号 | 用例 | shape | 数据分布 | 期望 |
|-----|------|-------|---------|------|
| T1/P1 | 小 shape | [1,1,1,128] | 正负随机 | torch.relu(x) |
| T2/P2 | 中 shape | [1,4,32,64] | 正负随机 | torch.relu(x) |
| T3/P3 | 大 shape | [2,8,64,128] | 正负随机 | torch.relu(x) |
| T4/P4 | 全正数 | [1,4,32,64] | 全正 | y = x |
| T5/P5 | 全负数 | [1,4,32,64] | 全负 | y = 0 |
| T6/P6 | 含零值 | [1,4,32,64] | 混合含零 | torch.relu(x) |

---

## 4. 开发进度

| 阶段 | 检查项 | 状态 |
|------|--------|------|
| 框架搭建 | 工程创建 + CMake + 空 Kernel 编译通过 | ⬜ |
| Kernel 实现 | TilingData + Host Tiling + Kernel Compute + 编译通过 | ⬜ |
| 可执行文件验证 | T1-T6 全部通过 | ⬜ |
| PyTorch 验证 | TORCH_LIBRARY 注册 + P1-P6 全部通过 | ⬜ |

---

## 5. 已知问题和决策记录

| 日期 | 问题/决策 | 说明 |
|------|----------|------|

---

## 6. 测试结果

### 6.1 可执行文件通路

**状态**: ⬜ | **脚本**: run.sh + scripts/verify_result.py

| 编号 | 用例 | 结果 | Max Diff |
|-----|------|------|----------|
| T1 | [1,1,1,128] 随机 | ⬜ | |
| T2 | [1,4,32,64] 随机 | ⬜ | |
| T3 | [2,8,64,128] 随机 | ⬜ | |
| T4 | 全正数 | ⬜ | |
| T5 | 全负数 | ⬜ | |
| T6 | 含零值 | ⬜ | |

### 6.2 PyTorch 通路

**状态**: ⬜ | **脚本**: scripts/test_torch.py

| 编号 | 用例 | 结果 | Max Diff |
|-----|------|------|----------|
| P1 | [1,1,1,128] 随机 | ⬜ | |
| P2 | [1,4,32,64] 随机 | ⬜ | |
| P3 | [2,8,64,128] 随机 | ⬜ | |
| P4 | 全正数 | ⬜ | |
| P5 | 全负数 | ⬜ | |
| P6 | 含零值 | ⬜ | |
