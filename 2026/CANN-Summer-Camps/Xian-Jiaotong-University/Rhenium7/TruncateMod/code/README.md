# TruncateMod 算子

> **CANN Summer Camp 2026 — 西安交通大学 / Rhenium7 小组**  
> Ascend C 自定义算子 · Elementwise + Broadcast · Ascend 910B3 / CANN 8.5.2

---

## 目録

- [1. 算子概述](#1-算子概述)
- [2. 数学公式](#2-数学公式)
- [3. 支持的数据类型与广播](#3-支持的数据类型与广播)
- [4. 架构设计](#4-架构设计)
- [5. 项目结构](#5-项目结构)
- [6. 环境准备](#6-环境准备)
- [7. 编译](#7-编译)
- [8. 测试](#8-测试)
- [9. 测试结果摘要](#9-测试结果摘要)
- [10. 已知限制](#10-已知限制)
- [11. 性能](#11-性能)
- [12. 文档参考](#12-文档参考)

---

## 1. 算子概述

`TruncateMod`（算子名 `truncate_mod`）是一个二元逐元素截断取模算子，数学行为为：**先做除法，向 0 截断，乘回被除数，作减法**：

```
y = x1 - trunc(x1 / x2) * x2
```

其中 `trunc` 为向 0 取整（直接丢弃小数部分），`x1` 与 `x2` 按 numpy 右对齐广播规则生成输出 `y`。

| 项目 | 内容 |
|------|------|
| 算子名称 | TruncateMod (`truncate_mod`) |
| 算子类别 | Elementwise（二元逐元素） + Broadcast |
| 输入 | `x1`、`x2` — dtype 需一致，均为 BF16 / FP16 / FP32 |
| 输出 | `y` — shape 为 `broadcast(x1, x2)`，dtype 与输入一致 |
| 支持芯片 | Ascend 910B（Ascend 910B3，NPU Arch **DAV_2201**）；op_def 同时注册 Ascend 950（方案兼容） |
| CANN 版本 | 8.5.2 |
| 编译架构 | `--npu-arch=dav-2201` |

>**设计原则**：在 Ascend 910B 的硬件约束下（BF16 不支持直算 Div/Mul/Sub/Trunc、DataCopyPad GM→UB 仅 4 参原型、UB Copy 仅 mask 形式且 repeatTime ≤ 255），本算子采用 **FP32 全链计算** 策略，保证三种 dtype 均达标 `ops-precision-standard` 浮点精度要求。详见 [§4 架构设计](#4-架构设计)。

---

## 2. 数学公式

```
输入: x1 — shape[S1], dtype ∈ {BF16, FP16, FP32}
      x2 — shape[S2], dtype ∈ {BF16, FP16, FP32}（与 x1 同 dtype）
输出: y  — shape[S] = broadcast(S1, S2), dtype 与输入一致

计算步骤（逐元素）:
  q   = x1 / x2            // 商
  qt  = trunc(q)           // 向 0 截断
  y   = x1 - qt * x2       // 截断取模
```

**数值语义**：

- `trunc(v)` 向 0 截断。例如 `trunc(3.9) = 3`，`trunc(-3.9) = -3`。
- `x2 == 0` 时：商为 ±Inf/NaN，经 `trunc`→`Mul`→`Sub` 传播，输出 Inf/NaN，与 CPU float64 golden（IEEE 语义）一致，**不特殊处理**。
- **禁止**使用 `Cast<float→int32_t>` 实现截断 — 当 `|q| ≥ 2^31` 时 int32 会溢出。本算子采用 `Cast<float, float, RoundMode::CAST_TRUNC>`（等价官方高阶 `Trunc` 在 220 架构的实现），规避溢出风险。

---

## 3. 支持的数据类型与广播

### 3.1 数据类型

| dtype | tilingKey / 模式 | 计算策略 | bufferDivisor (B/elem) |
|-------|------------------|----------|----------------------|
| FP16  | `MODE_0` = 0 | 升 FP32 全链计算 | 24 |
| BF16  | `MODE_1` = 1 | 升 FP32 全链计算 | 24 |
| FP32  | `MODE_2` = 2 | FP32 直算 | 28 |

>**为什么 BF16 必须升 FP32？** Ascend 910B 的 `Div`/`Mul`/`Sub`/`Trunc` API **不支持** `bfloat16_t`，这是平台硬约束，不是精度考虑。FP16 升精度则是为达到 `max_abs_error_limit` 精度标准所必需。

### 3.2 广播语义

采用 **numpy 右对齐广播**：低 rank 的输入在左侧补 1，逐维对齐；输出 rank = `max(rank(x1), rank(x2))`。

**支持的广播形态**（由 row-run 模型 + 4 种加载策略实现）：

| 形态 | 示例 | 说明 |
|------|------|------|
| ELEWISE | `[4,8] × [4,8]` | 双输入 shape 完全一致 |
| 标量广播 | `[1024] × [1]` | 标量输入 1 元素搬运 + `Duplicate` 展开 |
| 行/前缀广播 | `[1,64] × [64,64]`，`[1,1,32] × [4,5,32]` | 输出的最后一个/最后几个维度与输入一致，row-run 模型满足 rowCount=1 |

>**⚠️ 已知不支持**：**列广播**（如 `[64,1] × [1,64]`）和**中间轴广播**（如 `[2,1,3] → [2,4,3]`）这些 `rowCount > 1` 的形态，row-run 模型在当前设计下**无法正确处理**，Tiling 层会以 `GRAPH_FAILED` 主动拒绝（防御性校验），详见 [§10 已知限制](#10-已知限制)。

---

## 4. 架构设计

完整技术设计详见 [`docs/DESIGN.md`](docs/DESIGN.md)。

### 4.1 计算链

```
GM x1 (T) ──DataCopyPad(4 参)──▶ VECIN queue x1Local (T, 双缓冲)
GM x2 (T) ──DataCopyPad(4 参)──▶ VECIN queue x2Local (T, 双缓冲)
                          │ EnQue/DeQue
                          ▼
          [mode0/1] Cast<T→float> → x1F / x2F (VECCALC)
          [mode2]   x1Local / x2Local 直接作为 FP32 输入
                          ▼
          q = Div(x1F/x1Local, x2F/x2Local)          // 商
          q = Cast<float,float,CAST_TRUNC>(q)          // 截断
          q = Mul(q, x2F/x2Local)                      // q * x2
          y = Sub(x1F/x1Local, q)                      // x1 - q*x2
                          ▼
          [mode0/1] Cast<float→T, RINT> → VECOUT queue yLocal (T, 双缓冲)
          [mode2]   Sub(yLocal, x1Local, qF) 直接写 VECOUT queue yLocal (T, 双缓冲)
                          ▼
VECOUT queue yLocal ──DataCopyPad(3 参)──▶ GM y (T)
```

### 4.2 三种 Tiling 模式

| 模式 | 输入 dtype | 升精度 | 输出路径 | 说明 |
|------|-----------|--------|----------|------|
| mode0 | FP16 | ✅ Cast half→float | outQueueY | VECCALC 全流程 FP32 |
| mode1 | BF16 | ✅ Cast bf16→float | outQueueY | VECCALC 全流程 FP32 |
| mode2 | FP32 | ❌ 直算 | outQueueY | 直接使用 VECIN 队列缓冲 |

**全模式统一走 `outQueueY`（VECOUT）标准输出路径**，而非 VECIN 就地输出，确保正确的流水同步与缓冲生命周期管理。

### 4.3 广播 — row-run 模型与 4 种加载策略

Host 侧 Tiling 阶段对每个输入计算 row-run 描述 `(innerLen, rowCount)`：

- `innerLen`：输出最内维向左扫描，与输出相同的连续段累乘。
- `rowCount = inTotal / innerLen`：输入自身的行重复次数。
- 重复周期 `period = innerLen × rowCount = inTotal`。

Kernel 对每个输入、每个 tile `[s, s+T)` 采用以下 4 种加载策略：

| 策略 | 触发条件 | 说明 |
|------|----------|------|
| **0** (标量) | `inTotal == 1` | 1 元素 `DataCopyPad` → `GetValue` → `Duplicate` 展开 |
| **1** (单行) | `s % innerLen + T ≤ innerLen` | tile 位于单个源行内，一次连续搬运 |
| **2** (整周期) | `s % period == 0 && T % period == 0 && (period×sizeof(T)) % 32 == 0 && period ≤ maskCap` | 搬入 1 个周期到 `[0,P)`，mask 形式 `Copy` 扩展到 `[P,T)`，`repeatTime > 255` 时分片 |
| **3** (逐段兜底) | 其它情况 | tile 按 period 边界切成连续段，逐段 `DataCopyPad` |

> `maskCap = 256 / sizeof(T)`：Copy 掩码连续模式单次迭代元素上限（16 位=128，32 位=64），详见 [WALKTHROUGH.md 问题 3](docs/WALKTHROUGH.md)。

### 4.4 多核与 UB 切分

| 策略 | 公式 |
|------|------|
| 每核任务量 | `blockFactor = CeilAlign(ceil(totalNum / coreNum), 512)` |
| 使用核数 | `coreNum = min(ceil(totalNum × sizeof(T) / 4KB), GetCoreNumAiv())` |
| UB 每次循环元素数 | `ubFactor = FloorAlign((ubSize - extraSize) / bufferDivisor, 256 / sizeof(T))` |

- `bufferDivisor = 24`（mode0/1，T=2B）或 `28`（mode2，T=4B）
- `extraSize ≈ 1KB` 预留
- 运行时通过 `GetCoreMemSize(UB)` 获取 UB 容量，**零硬编码**

---

## 5. 项目结构

```
TruncateMod/
├── CMakeLists.txt                # 顶层 CMake：find_package(ASC) + op_kernel/op_host 子目录
├── build.sh                      # 一键式构建脚本：build / clean / ut / example
│
├── op_host/                      # Host 侧（Tiling + InferShape + OpDef）
│   ├── CMakeLists.txt
│   ├── truncate_mod_def.cpp      # 算子定义：BF16/FP16/FP32 dtype，ND 格式，注册 ascend910b/950
│   ├── truncate_mod_infershape.cpp  # numpy 右对齐广播推导 + dtype 一致性校验
│   └── truncate_mod_tiling.cpp   # 多核切分、UB 切分、row-run 描述、tilingKey 三路选择
│
├── op_kernel/                    # Kernel 侧（Ascend C kernel）
│   ├── CMakeLists.txt
│   ├── truncate_mod.cpp          # kernel 入口：三路 if constexpr 分派
│   ├── truncate_mod.h            # TruncateMod<T> 模板类：Init/Process/CopyIn/Compute/CopyOut
│   ├── truncate_mod_tiling_data.h # TilingData 结构体（7 个 int64 字段）
│   └── truncate_mod_tiling_key.h  # tilingKey 模板参数（MODE_0/1/2）
│
├── examples/                     # aclnn 调用示例
│   ├── CMakeLists.txt
│   ├── run.sh                    # 编译 + 运行 aclnn 示例
│   └── test_aclnn_truncate_mod.cpp   # A1-A3 用例：FP16/BF16/FP32 广播 + FP64 golden 比对
│
├── tests/ut/                     # 单元测试
│   ├── CMakeLists.txt
│   ├── run.sh                    # UT 入口：op_host UT → gen_data → op_kernel UT → compare
│   ├── cmake/BuildGoogleTest.cmake  # 从源构建 GoogleTest（OLD ABI）
│   ├── common/                   # UT 公共框架（tiling_context_faker 等）
│   ├── op_host/
│   │   ├── CMakeLists.txt
│   │   ├── test_truncate_mod_tiling.cpp  # Tiling UT（T1-T12，12 用例）
│   │   └── test_op_host_main.cpp
│   └── op_kernel/
│       ├── CMakeLists.txt
│       ├── test_truncate_mod.cpp       # Kernel UT（K1-K10，12 用例）
│       ├── truncate_mod_tiling.h       # Kernel UT 侧 TilingData 定义
│       └── truncate_mod_data/
│           ├── gen_data.py             # 测试数据生成 + FP64 golden
│           ├── compare_data.py         # 精度比对（MERE/MARE + NaN/Inf 按位）
│           └── run_precision_check.py  # ops-precision-standard 精度验收
│
├── docs/                         # 设计与验收文档
│   ├── DESIGN.md                 # 技术设计（v1.1）
│   ├── PLAN.md                   # 开发计划
│   ├── REVIEW.md                 # 代码审查报告（Round 0）
│   ├── WALKTHROUGH.md            # 设计串讲质疑清单
│   ├── environment.md            # 环境检查报告
│   ├── perf/round_001/summary.txt    # 性能验收摘要
│   └── precision/summary.txt         # 精度验收摘要
│
└── build/                        # 构建输出（构建后生成）
    ├── custom_opp_*.run           # 算子安装包
    ├── op_host/truncate_mod_op_host_ut
    └── op_kernel/truncate_mod_op_kernel_ut
```

---

## 6. 环境准备

| 组件 | 要求 | 备注 |
|------|------|------|
| 操作系统 | Linux (aarch64) | |
| 芯片 | Ascend 910B3 | SocVersion = `ASCEND910B`，NpuArch = `DAV_2201` |
| CANN | 8.5.2 | `ASCEND_HOME_PATH=/home/developer/Ascend/cann-8.5.2` |
| 编译器 | bisheng (`ccec`) | `$ASCEND_HOME_PATH/bin/ccec` |
| asc-devkit | ≥ 2800 API 文档 | `/mnt/workspace/code/asc-devkit` |
| Python | 3.x + `numpy` + `ml_dtypes` | 用于测试数据生成与精度比对 |

>**环境自检**：参见 [`docs/environment.md`](docs/environment.md)（执行 `/ops-env-check` skill）。

---

## 7. 编译

### 7.1 构建算子包

```bash
# 默认编译 Ascend 910B，8 线程
bash build.sh -j8

# 清空构建产物
bash build.sh --make_clean
```

构建产物：

- `build/custom_opp_truncate_mod_custom.run` — 算子安装包
- `build/op_kernel/ascendc_kernels/binary/ascend910b/` — 包含 3 个 dtype 实例的 `.o`

### 7.2 安装算子包

```bash
cd build
./custom_opp_truncate_mod_custom.run --quiet
# 安装到 $ASCEND_HOME_PATH/opp/vendors/truncate_mod_custom/
```

安装后生成 `aclnn_truncate_mod.h` 头文件与 `libcust_opapi.so` 动态库，用于 aclnn 调用。

---

## 8. 测试

### 8.1 单元测试（UT）

```bash
# 运行所有 UT：Tiling UT + Kernel UT + 精度比对
bash build.sh -u
```

UT 脚本 (`tests/ut/run.sh`) 执行流程：

1. **Tiling UT** (`op_host/truncate_mod_op_host_ut`) — 验证 Tiling 逻辑（T1-T12）
2. **数据生成** (`gen_data.py`) — 生成 K1-K10 的输入/输出/golden bin 文件
3. **Kernel UT** (`op_kernel/truncate_mod_op_kernel_ut`) — 运行 kernel，输出 bin 文件
4. **精度比对** (`compare_data.py`) — MERE < threshold && MARE < 10×threshold

#### Tiling UT 用例（T1-T12）

| 编号 | 用例 | dtype | 输入 | 说明 |
|------|------|-------|------|------|
| T1 | FP16 ELEWISE 回归 | FP16 | `[5]×[5]` | tilingKey=0，row-run=(5,1) |
| T2 | FP16 ELEWISE 多维 | FP16 | `[4,8]×[4,8]` | |
| T3 | BF16 ELEWISE | BF16 | `[1024]×[1024]` | tilingKey=1 |
| T4 | FP32 ELEWISE | FP32 | `[1024]×[1024]` | tilingKey=2 |
| T5 | 标量广播 | FP32 | `[1024]×[1]` | x2 标量 |
| T6 | 行广播 | FP16 | `[1,64]×[64,64]` | rowCount=1 |
| T7 | 左侧前缀广播 | FP32 | `[1,1,32]×[4,5,32]` | |
| T8 | 全标量 | BF16 | scalar×scalar | EnsureNotScalar 语义 |
| T9 | 非法广播 | FP16 | `[2,3]×[4]` | 期望 GRAPH_FAILED |
| T10 | 大 shape 多核 | FP32 | `[1000000]×[1000000]` | 多核切分 |
| T11 | period 对齐优化 | FP32 | `[1000,8]×[1000,8]` | ubFactor=8000 |
| T12 | xfail 列广播 | FP32 | `[64,1]×[1,64]` | 期望 GRAPH_FAILED |

#### Kernel UT 用例（K1-K10）

| 编号 | 用例 | dtype | 输入 | 说明 |
|------|------|-------|------|------|
| K1 | FP16 ELEWISE 非对齐 | FP16 | `[5]` | 非 32B 对齐回归 |
| K2 | FP16 ELEWISE 多核 | FP16 | `[8192]` | 8 核 |
| K3 | BF16 ELEWISE 多核 | BF16 | `[8192]` | |
| K4 | FP32 ELEWISE 多核 | FP32 | `[8192]` | |
| K5 | FP32 行广播 | FP32 | `[1,64]×[64,64]` | 策略 1/2 覆盖 |
| K6 | FP16 标量广播 | FP16 | `[1024]×[1]` | 策略 0 |
| K7 | 正负混合 + x2=0 | FP32 | `[2048]` | IEEE NaN/Inf |
| K8 | 大商值 | FP32 | `[1024]` | `\|q\|≥2^31`，FP32 语义 golden |
| K9 | 非对齐 tile + 尾块 | 3 dtype | `[1003]` | |
| K10 | 左侧前缀广播 | BF16 | `[1,1,32]×[4,5,32]` | 策略 3 跨行兜底 |

### 8.2 aclnn 端到端测试

```bash
# 需先安装算子包（见 §7.2），然后运行
bash build.sh -e
```

| 编号 | 用例 | dtype | 输入 | 说明 |
|------|------|-------|------|------|
| A1 | FP16 基本 | FP16 | `[16]×[16]` | |
| A2 | BF16 基本 | BF16 | `[16]×[16]` | |
| A3 | FP32 广播 | FP32 | `[4,8]×[8]` | |

>**⚠️**：aclnn 示例需要物理 NPU（Ascend 910B3），设置 `ASCEND_VISIBLE_DEVICES=3` 使用逻辑设备 0。CANN 8.5.2 注意事项：`ACL_FLOAT32` 不存在（用 `ACL_FLOAT`）、`ACLNN_SUCCESS` 需 `#include "aclnn/opdev/op_errno.h"`、执行器销毁接口为 `aclDestroyAclOpExecutor`。

---

## 9. 测试结果摘要

### 9.1 Tiling UT

```
脚本: tests/ut/run.sh → build/op_host/truncate_mod_op_host_ut
状态: ✅ 12/12 通过  (2026-07-31)
```

### 9.2 Kernel UT + 精度比对

```
脚本: tests/ut/run.sh → build/op_kernel/truncate_mod_op_kernel_ut + compare_data.py
状态: ✅ 12/12 + 精度比对全 PASS  (2026-07-31)
```

### 9.3 精度验收 (ops-precision-standard)

```
状态: ✅ 通过  (12 用例 / 3 dtype，详见 docs/precision/summary.txt)
```

| dtype | rtol | atol | max_abs_error_limit | max_abs (实测) | match_ratio |
|-------|------|------|--------------------|----------------|-------------|
| FP16 | 2⁻⁹ (1.95e-3) | 2⁻⁹ | 0.1 | 0 | 1.0 |
| BF16 | 2⁻⁶ (1.56e-2) | 2⁻⁶ | 1.0 | 0 | 1.0 |
| FP32 | 2⁻¹⁰ (9.77e-4) | 2⁻¹⁶ | 1e-2 | ≤4.77e-7 | 1.0 |

### 9.4 aclnn 通路

```
脚本: bash build.sh -e
状态: ✅ A1-A3 全 PASS  (2026-07-31 上板，NPU 3 / Ascend 910B3 / CANN 8.5.2)
```

| 用例 | matched_ratio | max_abs_error | MERE | MARE |
|------|--------------|---------------|------|------|
| A1 (FP16) | 1.0 | 0 | 0 | 0 |
| A2 (BF16) | 1.0 | 0 | 0 | 0 |
| A3 (FP32) | 1.0 | 2.38e-7 | 4.5e-8 | 6.8e-7 |

### 9.5 代码审查

```
审查: REVIEW.md — Round 0（Step 4 初审）
判定: ✅ PASS  总分: 92/100
```

---

## 10. 已知限制

### 10.1 不支持的广播形态

| 形态 | 示例 | 原因 | 处理 |
|------|------|------|------|
| 列广播 | `[64,1] × [1,64]` | row-run 模型 `rowCount > 1`，核加载映射错误 | Tiling 层返回 `GRAPH_FAILED`（防御性拒绝） |
| 中间轴广播 | `[2,1,3] → [2,4,3]` | 同上 | Tiling 层返回 `GRAPH_FAILED` |

>**后续优化方向**：引入 3D Stride 模型或通用化逐段 `DataCopyPad`，可支持 `rowCount > 1` 的广播。详见 [`docs/PLAN.md` §5](docs/PLAN.md)。

### 10.2 trunc 翻转概率

`trunc` 在整数边界不连续，任何有限精度实现相对于 float64 golden 均可能产生**翻转**：单元素翻转概率 `P ≈ 2⁻²³ × |q|`，与商幅值 `|q|` 线性相关。

**影响**：当 `|q|` 较大或数据量 `N` 巨大时，期望翻转数 `λ = N × 2⁻²³ × E|q|` 可能 ≥ 1，导致个别元素误差 `≈ |x2|`，可能击穿 `max_abs_error_limit`。

**规范约束**：

1. **验收数据域需保证 `λ ≪ 1`**：限制 `|q| = |x1/x2|` 的上界、`x2` 下界远离 0，或使用整数/有限小数域数据。
2. **K8 类大商值用例仅做功能性校验**（验证无 int32 溢出、Inf/NaN 传播正确），**不纳入精度指标**。
3. 验证脚本 `gen_data.py` / `compare_data.py` 已按上述约束处理。

> 详见 [`docs/DESIGN.md` §2.3](docs/DESIGN.md)、[`docs/WALKTHROUGH.md` 问题 4](docs/WALKTHROUGH.md)。

### 10.3 rank-0 标量输出语义

numpy 中两个 0-d 标量广播结果为 0-d；本算子统一输出 shape `[1]`（沿用骨架 `EnsureNotScalar` 语义）。对纯 aclnn 调用方，输出 shape 与 numpy 语义有轻微偏差。

---

## 11. 性能

性能验收数据位于 [`docs/perf/round_001/`](docs/perf/round_001/summary.txt)。

| 指标 | 值 |
|------|------|
| Task Duration | 4.36 us (FP16) / 4.52 us (BF16/FP32) |
| Block Dim | 1 (单核运行) |
| 主导流水 | Vector (VECIN/VECOUT) |
| 计算利用率 | aic_cube_ratio ≈ 3% |
| 内存带宽 | aic_mte2_active_bw ≈ 0.67 GB/s |
| 达标状态 | ✅ |

**性能特征**：TruncateMod 为 elementwise + broadcast 算子，计算量相对较小，受限于内存搬运。3 个 tilingKey 模式性能相当，`Block Dim=1` 单核运行。

**优化建议** (见 [`docs/REVIEW.md` §4.4](docs/REVIEW.md))：

1. **策略 2 的 UB Copy 扩展缺显式同步**：当前依赖 `CopyIn` 阶段的隐式顺序，存在潜在 MTE2→V 竞态风险。建议在 `Copy` 前插入 `PipeBarrier()` 或将扩展逻辑移入 `Compute()`（V 管道内天然有序）。
2. **多核切分**：当前验收为单核；可通过调整 `blockFactor` 提升多核并行度。

---

## 12. 文档参考

| 文档 | 说明 |
|------|------|
| [`docs/DESIGN.md`](docs/DESIGN.md) | 技术设计（v1.1）：API 映射、数据流、内存规划、多核/UB 切分、分支场景、Host 侧设计 |
| [`docs/PLAN.md`](docs/PLAN.md) | 开发计划：需求概述、文件清单、测试计划、开发进度、已知问题决策记录、测试结果 |
| [`docs/REVIEW.md`](docs/REVIEW.md) | 代码审查报告（Round 0）：8 维度评审，总分 92/100，PASS |
| [`docs/WALKTHROUGH.md`](docs/WALKTHROUGH.md) | 设计串讲质疑清单：10 个问题的审查与 Architect 回应 |
| [`docs/environment.md`](docs/environment.md) | 环境检查报告：硬件/CANN/编译器/asc-devkit |
| [`docs/perf/round_001/summary.txt`](docs/perf/round_001/summary.txt) | 性能验收摘要 |
| [`docs/precision/summary.txt`](docs/precision/summary.txt) | 精度验收摘要 |

---

> **版本信息**：v1.1（2026-07-31）  
> **适配环境**：Ascend 910B3 / CANN 8.5.2 / NpuArch DAV_2201  
> **代码检出**：`proj/source` 分支
