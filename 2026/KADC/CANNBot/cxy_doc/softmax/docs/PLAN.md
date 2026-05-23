# softmax 算子开发计划

> 本文档在开发流程中持续更新。

---

## 1. 需求概述

| 项目 | 内容 |
|-----|------|
| 算子名称 | softmax |
| 数学公式 | softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x))) |
| 输入 | x: shape=任意, dtype=float16/float32 |
| 输出 | y: shape=同输入, dtype=同输入 |
| 算子类别 | Reduction/Norm（归约+广播变换） |
| 需求类型 | 通用 |
| 归约轴 | 默认 last axis（-1），当前实现仅支持尾轴 |

---

## 2. 文件清单

| 文件 | 状态 |
|------|------|
| `kernel/softmax_tiling.h` — Tiling 结构体 + Tiling 计算函数 | ⬜ |
| `kernel/softmax_kernel.asc` — Kernel 计算逻辑（FP32/FP16 双模板） | ⬜ |
| `host/softmax.asc` — Host + main 入口 | ⬜ |
| `torch_library/softmax_torch.cpp` — PyTorch host（Tiling + launch） | ⬜ |
| `torch_library/register.cpp` + `torch_library/ops.h` — TORCH_LIBRARY 注册 | ⬜ |
| `CMakeLists.txt` — 双 target（可执行文件 + .so） | ⬜ |
| `run.sh` + `scripts/gen_data.py` + `scripts/verify_result.py` | ⬜ |
| `scripts/test_torch.py` — PyTorch 通路测试 | ⬜ |

---

## 3. 测试计划

精度标准：FP32 atol=1e-5, rtol=1e-5；FP16 atol=1e-3, rtol=1e-3

**Golden 计算**：定义在 `scripts/golden.py` 中，gen_data.py 和 test_torch.py 共用。

```python
# golden.py
import torch

def softmax_golden(x, dim=-1):
    return torch.nn.functional.softmax(x.float(), dim=dim).to(x.dtype)
```

**用例（T=可执行文件, P=PyTorch, 1:1 对应）**：

| 编号 | 用例 | 输入 shape | dtype | 预期输出 |
|-----|------|-----------|-------|---------|
| T1/P1 | 随机数据（对齐） | [32, 128] | float32 | softmax_golden(x) |
| T2/P2 | 随机数据（非对齐） | [16, 37] | float32 | softmax_golden(x) |
| T3/P3 | 随机数据（大 shape） | [1024, 2048] | float32 | softmax_golden(x) |
| T4/P4 | 随机数据（FP16 对齐） | [32, 128] | float16 | softmax_golden(x) |
| T5/P5 | 随机数据（FP16 非对齐） | [16, 37] | float16 | softmax_golden(x) |
| T6/P6 | 极端值测试（含 -FLT_MAX 行） | [8, 128] | float32 | softmax_golden(x) |
| T7/P7 | 小 shape（1 行） | [1, 64] | float32 | softmax_golden(x) |

**特殊用例说明**：
- T2/P2/T5/P5：非对齐列数（37 不是 8 的倍数），验证 DataCopyPad + oriSrcK 处理
- T6/P6：含极端负值行（全部为 -FLT_MAX），验证 AdjustSoftMaxRes 后处理
- T7/P7：单行输入，验证最小 shape 场景

---

## 4. 开发进度

| 阶段 | 检查项 | 状态 |
|------|--------|------|
| 框架搭建 | 工程创建 + CMake 双 target + 空 Kernel 编译通过 | ⬜ |
| Kernel 实现 | TilingData + Host Tiling + Kernel Compute + FP32/FP16 双模板 + 编译通过 | ⬜ |
| 可执行文件验证 | T1-T7 全部通过 | ⬜ |
| PyTorch 验证 | TORCH_LIBRARY 注册 + `torch.ops.npu.softmax()` 可调用 + P1-P7 全部通过 | ⬜ |
| 性能验收 | msprof 采集 + 数据归档 + 达标判定 | ⬜ |

---

## 5. 已知问题和决策记录

| 日期 | 问题/决策 | 说明 |
|------|----------|------|
| 2026-05-23 | 当前仅支持尾轴（axis=-1） | SoftMax 高阶 API 仅支持尾轴归约；非尾轴支持为 Phase 2 扩展 |
| 2026-05-23 | 使用 SoftMax 高阶 API | 内部完成 ReduceMax+Sub+Exp+ReduceSum+Div 全流程，数值稳定且性能优化 |
| 2026-05-23 | AdjustSoftMaxRes 后处理 | 处理 max==-FLT_MAX 的极端值行，将结果置为 0 |
| 2026-05-23 | FP16 混合精度 | FP16 输入使用 SoftMax<half> 异质模板，sum/max 内部 float 计算 |

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

- [ ] `build/softmax` 可执行文件存在
- [ ] `build/libsoftmax_ops.so` 存在
- [ ] `torch.ops.load_library` + `torch.ops.npu.softmax` 可调用

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

---

## 9. Phase 2 扩展计划（非尾轴 Softmax）

> 当前实现仅支持尾轴（axis=-1）。以下为非尾轴支持的扩展规划：

| 扩展项 | 说明 | 优先级 |
|--------|------|--------|
| 非尾轴 Softmax | 合轴后 ARA 模式，手动实现 ReduceMax(RA)+Broadcast+Exp+ReduceSum(RA)+Broadcast+Div | P2 |
| 动态轴参数 | PyTorch 接口增加 `dim` 参数，Tiling 根据轴位置选择 AR/ARA 分支 | P2 |
| 大 R ColSplit | R 超过全载阈值时的分载策略 | P2 |

**ARA 模式设计概要**：
- 合轴后 shape 为 (A1, R, A0)，A0 > 1
- 使用 Pattern::Reduce::RA 进行 ReduceMax/ReduceSum
- 广播使用 src1RepStride=0 技巧（BinaryRepeatParams）
- 需要额外的 transpose 或 reshape 逻辑