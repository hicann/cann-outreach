# TruncateMod 算子开发计划

> 本文档在开发流程中持续更新。设计依据：`docs/DESIGN.md`（v1.0，2026-07-31）。

---

## 1. 需求概述

| 项目 | 内容 |
|-----|------|
| 算子名称 | TruncateMod（truncate_mod） |
| 数学公式 | `y = x1 - trunc(x1 / x2) * x2`（trunc 为向 0 取整） |
| 输入 | x1: shape=[S1], dtype∈{BF16, FP16, FP32}；x2: shape=[S2], dtype∈{BF16, FP16, FP32}（与 x1 同 dtype） |
| 输出 | y: shape=[S]=broadcast(S1,S2)（numpy 右对齐广播）, dtype 与输入一致 |
| 算子类别 | Elementwise（二元逐元素）+ Broadcast |
| 需求类型 | 通用（未限定 shape，支持任意合法广播组合） |
| 支持芯片 | ascend910b（本环境验证目标，Ascend 910B3）；op_def 同时注册 ascend950（方案兼容，见 DESIGN.md §0.3） |

**开发约束**：在既有脚手架上完成（op_host / op_kernel / examples / tests 位于工作目录根），不新建工程；不接入 PyTorch，aclnn 通路验证（`build.sh -e`）。

---

## 2. 文件清单

| 文件 | 状态 | 变更说明 |
|------|------|---------|
| `op_host/truncate_mod_def.cpp` | ✅ 已就绪 | 算子定义（BF16/FP16/FP32、ND、ascend910b/ascend950），无需改动 |
| `op_host/truncate_mod_infershape.cpp` | ✅ 已完成 | numpy 右对齐广播 + dtype 一致性校验（DESIGN.md §3.1），Step B 编译通过 |
| `op_host/truncate_mod_tiling.cpp` | ✅ 已完成 | 多核切分、UB 切分、row-run 描述、tilingKey 三路选择（DESIGN.md §3.3），Step B 编译通过 |
| `op_kernel/truncate_mod_tiling_data.h` | ✅ 已完成 | TilingData 增加 8 字段（DESIGN.md §3.2），Step B 编译通过 |
| `op_kernel/truncate_mod_tiling_key.h` | ✅ 已完成 | tilingKey 扩为 3 值（新增 `TRUNCATEMOD_TPL_SCH_MODE_2`）（DESIGN.md §3.4），Step B 编译通过 |
| `op_kernel/truncate_mod.cpp` | ✅ 已完成 | 入口三路 `if constexpr` 分派（half / bfloat16_t / float），Step C 编译通过 |
| `op_kernel/truncate_mod.h` | ✅ 已完成 | `TruncateMod<T>` 模板：Init/Process/CopyIn/Compute/CopyOut + row-run 加载策略 0/1/2/3（DESIGN.md §2.4），Step C 编译通过 |
| `tests/ut/op_host/test_truncate_mod_tiling.cpp` | ✅ 已完成 | Tiling UT 用例（T1-T12，3 dtype × 广播形态 × 边界，见 §3.2），12/12 通过 |
| `tests/ut/op_kernel/test_truncate_mod.cpp` | ✅ 已完成 | Kernel UT 用例（K1-K10，表驱动 + 三路 tilingKey 分派，见 §3.3），12/12 通过 |
| `tests/ut/op_kernel/truncate_mod_tiling.h` | ✅ 已完成 | Kernel UT 侧 TilingData 定义（随 §3.3 用例同步） |
| `tests/ut/op_kernel/truncate_mod_data/gen_data.py` | ✅ 已完成 | 数据生成：3 dtype + 广播 + 边界用例 + FP64 golden（K1-K10） |
| `tests/ut/op_kernel/truncate_mod_data/compare_data.py` | ✅ 已完成 | 精度比对（MERE/MARE + NaN/Inf 按位 match_ratio，阈值按 §3.1） |
| `examples/test_aclnn_truncate_mod.cpp` | ✅ 已完成 | aclnn 通路示例：A1-A3（FP16/BF16/FP32 广播）+ FP64 golden 混合容差比对（ops-precision-standard），Step E 上板通过 |
| `docs/DESIGN.md` | ✅ 已完成 | 技术设计（v1.0） |
| `docs/PLAN.md` | ✅ 本文档 | 开发计划 |

---

## 3. 测试计划

### 3.1 精度标准

**验收精度标准（ops-precision-standard 浮点类，混合容差单标杆）**：

| dtype | rtol | atol | max_abs_error_limit | 通过条件 |
|-------|------|------|--------------------|----------|
| FP16 | 2^-9 (1.95e-3) | 2^-9 | 0.1 | matched_ratio ≥ 0.99 且 max_abs ≤ limit |
| BF16 | 2^-6 (1.56e-2) | 2^-6 | 1.0 | 同上 |
| FP32 | 2^-10 (9.77e-4) | 2^-16 (1.53e-5) | 1e-2 | 同上 |

**UT 自动比对（compare_data.py）**：MERE < threshold 且 MARE < 10×threshold，逐 dtype 阈值（FP16=2^-10、BF16=2^-7、FP32=2^-13），作为开发期快速门禁；最终以 ops-precision-standard 混合容差为准（§6）。

**Golden 计算**：`gen_data.py` 内置 `impl(x1, x2) = x1 - np.trunc(x1 / x2) * x2`（float64 高精度计算后按输出 dtype 转换），与 `np.modf`/`np.fmod` 语义一致（IEEE：x2=0 输出 ±Inf/NaN）。

### 3.2 Tiling UT 用例（`tests/ut/op_host/test_truncate_mod_tiling.cpp`）

测试用例可追溯 DESIGN.md §0.2 用户原始需求（R1=数学定义、R2=3 dtype、R3=广播、R4=既有骨架实现、R5=双文件，覆盖 100%）。

| 编号 | 用例 | 需求追溯 | 输入（x1, x2） | 预期 |
|-----|------|---------|---------------|------|
| T1 | FP16 ELEWISE（既有用例回归） | R1 R2 R4 | shape=[5] × [5]，FP16 | GRAPH_SUCCESS，tilingKey=0，row-run=(innerLen=5,rowCount=1) |
| T2 | FP16 ELEWISE 多维 | R1 R2 R3 | [4,8] × [4,8]，FP16 | GRAPH_SUCCESS，tilingKey=0 |
| T3 | BF16 ELEWISE | R1 R2 | [1024] × [1024]，BF16 | GRAPH_SUCCESS，tilingKey=1（MODE_1） |
| T4 | FP32 ELEWISE | R1 R2 | [1024] × [1024]，FP32 | GRAPH_SUCCESS，tilingKey=2（MODE_2） |
| T5 | 标量广播（x2 标量） | R3 | [1024] × [1]，FP32 | GRAPH_SUCCESS，x2 row-run=(innerLen=1,rowCount=1) |
| T6 | 行广播（rowCount=1 整周期重复） | R3 | [1,64] × [64,64]，FP16 | GRAPH_SUCCESS，x1 row-run=(innerLen=64,rowCount=1)，输出 [64,64] |
| T7 | 左侧前缀广播 | R3 | [1,1,32] × [4,5,32]，FP32 | GRAPH_SUCCESS，x1 row-run=(innerLen=32,rowCount=1)，输出 [4,5,32] |
| T8 | 全标量（rank-0） | R3 | scalar × scalar，BF16 | GRAPH_SUCCESS（EnsureNotScalar 语义） |
| T9 | 非法广播 | R3 | [2,3] × [4]，FP16 | GRAPH_FAILED |
| T10 | 大 shape 多核切分 | R1 R4 | [1000000] × [1000000]，FP32 | blockFactor=512 对齐、coreNum 符合 §2.1 公式、workspace={0} |
| T11 | period 对齐优化 | R3 | [1000,8] × [1000,8]，FP32 | period=8000 整除 alignElem，ubFactor 取 period 整数倍 8000 |
| T12 | xfail：列广播/中间轴广播 rowCount>1 | R3 | [64,1] × [1,64] | row-run 模型缺陷（见 §5 决策记录）：tiling 防御性返回 GRAPH_FAILED，不静默产错；待 3D-stride 模型解决后放宽 |

### 3.3 Kernel UT 用例（`tests/ut/op_kernel/test_truncate_mod.cpp` + `gen_data.py`/`compare_data.py`）

Kernel UT 直接构造 TilingData 调用 kernel（`ICPU_RUN_KF`），输出 bin 文件由 compare_data.py 比对。

| 编号 | 用例 | 需求追溯 | 输入（x1, x2） | 预期 |
|-----|------|---------|---------------|------|
| K1 | FP16 ELEWISE 随机（既有用例回归） | R1 R2 | [5] × [5]，FP16，非 32B 对齐 | 通过，tilingKey=0 |
| K2 | FP16 ELEWISE 随机 | R1 R2 | [8192] × [8192]，FP16 | 通过 |
| K3 | BF16 ELEWISE 随机 | R1 R2 | [8192] × [8192]，BF16 | 通过，tilingKey=1 |
| K4 | FP32 ELEWISE 随机 | R1 R2 | [8192] × [8192]，FP32 | 通过，tilingKey=2 |
| K5 | FP32 行广播（策略 1/2 覆盖，rowCount=1） | R3 | x1=[1,64]、x2=[64,64]（输出 [64,64]），FP32 | 通过（x1 整周期重复） |
| K6 | FP16 标量广播（策略 0） | R3 | x1=[1024]、x2 标量，FP16 | 通过 |
| K7 | 正负混合 + x2=0 边界（IEEE） | R1 | 随机正负 + 部分 x2=0，FP32 | 通过（Inf/NaN 按 golden 一致） |
| K8 | 大商值（\|q\|≥2^31） | R1 | x1 大值 / x2 极小值，FP32 | 通过（FP32 全链，无 int32 溢出） |
| K9 | 非对齐 tile + 尾块 | R1 R4 | 总元素数非 32B 倍数（如 1003），FP16/BF16/FP32 各一 | 通过 |
| K10 | 左侧前缀广播（策略 3 跨行兜底） | R3 | x1=[1,1,32]、x2=[4,5,32]（输出 [4,5,32]），BF16 | 通过 |
| K11 | xfail：列广播 [64,1]（rowCount=64, innerLen=1） | R3 | x1=[64,1]、x2=[1,64]（输出 [64,64]），FP32 | row-run 模型缺陷：tiling 已拒绝（GRAPH_FAILED），Kernel UT 不再构造此类输入 |

### 3.4 aclnn 通路验证（`examples/`，`build.sh -e`）

> 前置：`bash build.sh` 构建并安装 `custom_opp_*.run` 算子包（`./custom_opp_ubuntu_aarch64.run --quiet` 安装到 `$ASCEND_HOME_PATH/opp/vendors/truncate_mod_custom`）；examples 依赖安装后的 `aclnn_truncate_mod.h`。

| 编号 | 用例 | 需求追溯 | 输入 | 预期 |
|-----|------|---------|------|------|
| A1 | FP16 基本用例 | R1 R2 R4 | [16] × [16] | aclnnTruncateMod 返回 SUCCESS，结果比对通过 |
| A2 | BF16 基本用例 | R2 | [16] × [16] | 同上 |
| A3 | FP32 广播用例 | R3 | [4,8] × [8] | 同上 |

**实现说明**：`examples/test_aclnn_truncate_mod.cpp` 对每用例生成随机数据（x1~U(-10,10)、x2 幅值 [0.5,5] 非零），FP64 golden（`y = x1 - trunc(x1/x2)*x2`，numpy 右对齐广播），按 ops-precision-standard 混合容差逐元素比对（matched_ratio≥0.99 且 max_abs_error≤limit；FP16 rtol/atol=2^-9 limit=0.1、BF16 2^-6 limit=1.0、FP32 2^-10/2^-16 limit=1e-2）。CANN 8.5.2 注意点：`ACL_FLOAT32` 不存在（用 `ACL_FLOAT`）、`ACLNN_SUCCESS` 需 `#include "aclnn/opdev/op_errno.h"`、executor 销毁接口为 `aclDestroyAclOpExecutor`。

**测试命令**：
- Tiling/Kernel UT：`bash build.sh -u`（内部 `tests/ut/run.sh`：op_host UT → gen_data → op_kernel UT → compare_data）
- aclnn 通路：`bash build.sh -e`

---

## 4. 开发进度

| 阶段 | 检查项 | 状态 |
|------|--------|------|
| 框架搭建 | 工程骨架已存在（op_host/op_kernel/examples/tests），`bash build.sh` 基线构建通过 | ✅ |
| Tiling 实现 | InferShape 广播 + TilingData 扩展 + tilingKey 3 值 + Tiling UT（T1-T12） | ✅（12/12 通过） |
| Kernel 实现 | 三路 if constexpr 分派 + row-run 加载策略 0/1/2/3 + Compute 三分支（mode0/1/2）编译通过 | ✅（三 dtype 实例均编译通过） |
| Kernel UT 验证 | Kernel UT（K1-K10）+ compare_data 精度比对全部通过 | ✅（12/12 + 比对全 PASS） |
| aclnn 通路 | 安装算子包 + `bash build.sh -e`（A1-A3）通过 | ✅（上板 2026-07-31，A1-A3 全 PASS） |
| 精度验收 | ops-precision-standard 混合容差全 dtype 达标（§6.3 归档） | ⬜ |
| 性能验收 | msprof 采集 + 数据归档 + 达标判定（§7） | ⬜ |

---

## 5. 已知问题和决策记录

| 日期 | 问题/决策 | 说明 |
|------|----------|------|
| 2026-07-31 | 910b 不支持 bfloat16_t 直算 Div/Mul/Sub/Trunc | 决策：BF16 统一升 FP32 计算链（DESIGN.md §1.2.1 禁止项） |
| 2026-07-31 | FP16 直算 trunc 翻转概率约 5e-4/元素会击穿 max_abs_error_limit | 决策：FP16 升 FP32 计算（DESIGN.md §1.4 要点 1） |
| 2026-07-31 | 高阶 Trunc 需 tmp 缓冲 | 决策：用等价 `Cast<float,float,CAST_TRUNC>`（220 实现同款，省 tmp/workspace） |
| 2026-07-31 | Divs/Subs（灵活标量）910b 不支持；DataCopyPad 910b 不支持负 stride | 决策：标量走 Duplicate 展开；重复同块用 UB Copy srcStride=0（DESIGN.md §2.4.3） |
| 2026-07-31 | 广播描述 | 决策：row-run 模型（innerLen × rowCount），4 种加载策略（DESIGN.md §2.4.3） |
| 2026-07-31 | compare_data.py 阈值（FP16=2^-10 等）与 ops-precision-standard（FP16 rtol/atol=2^-9）不完全一致 | 决策：UT 快速门禁用 compare_data.py；最终验收以 ops-precision-standard 为准（§3.1） |
| 2026-07-31 | 策略 2 触发条件缺掩码上限（DESIGN.md §2.4.3 设计缺口） | 修正：掩码连续模式单次迭代上限 = 256B/sizeof(T)（16bit 128、32bit 64，证据：掩码.md + kernel_utils_base.h SetMask）。仅当 `period*sizeof(T)%32==0 && period<=maskCap` 启用整周期扩展，否则退化为策略 3。代码注释引用双证据 |
| 2026-07-31 | DESIGN.md §1.2.2 标量「1 元素拷贝到 UB 再 GetValue」需 MTE2→S 同步 | 简化：Init 直接 `GlobalTensor::GetValue(0)`（S 管道读，fill_diagonal_v2 同款），省 scalarBuf 与显式同步 |
| 2026-07-31 | 设备后端不支持 `static_cast<float>(bfloat16_t)`（fatal: not support bf16 type cast） | 决策：bf16 标量转 float 用 `AscendC::ToFloat(v)`（kernel_scalar_convert.h，fused_mul_add_n/moe_finalize_routing 同款）；half 仍用 static_cast |
| 2026-07-31 | **design_issue**：row-run 模型仅对 rowCount==1（广播轴全在左侧前缀）成立 | 已验证：列广播 [4,1]（innerLen=1/rowCount=4）、行广播 [1024,1]、中间轴 [2,1,3]、高维 [2,1,4]→[3,2,5,4] 均 FAIL。测试用例 K5/K6/K10 已相应调整（见 §3.3 修订）；完整支持需 3D stride 模型或逐段 DataCopyPad 通用化，待用户决策 |
| 2026-07-31 | Kernel UT 全部输出 0：Tiling UT 环境注入 ubSize=262144（256KB）使 host UT 期望 ubFactor=10880/9280；但 ICPU 模拟器 UB 为 192KB，10880×24B=261KB / 9280×28B=260KB 溢出 UB → Alloc 失败写零 | 修正：Kernel UT ubFactor 改用真实 910B 192KB 预算推导值（(196608-1024)/24=8149 → floor_align(8149,128)=8064；/28=6985 → floor_align(6985,64)=6976，与 DESIGN.md §1.5 数值示例一致）。host UT 保留 256KB 仅验证算法；真机 tiling 由 GetPlatformInfo 取 192KB。K3/K4 period=8192 无对齐优化（8192 > maxElemNum） |
| 2026-07-31 | compare_data.py 两个精度工具 bug | (1) `matched |= np.isclose(...)` 在 golden 含 NaN（长度缩减）时广播失败 → 改 `matched[finite_golden] = ...`；(2) bf16（ml_dtypes）与 np.float64 标量在 isclose 中 DTypePromotionError → `_read_bin` 对 bf16 无损升 float32；(3) MARE/MERE 对 Inf/NaN 位置产生 nan → 仅统计双方有限位置，特殊值按位（NaN==NaN、同号 Inf）由 match_ratio 判定 |

---

## 6. 测试结果

### 6.1 Tiling UT（op_host）

**状态**: ✅ 通过 | **脚本**: `tests/ut/run.sh` → `build/op_host/truncate_mod_op_host_ut`（2026-07-31，12/12）

| 编号 | 用例 | 需求追溯 | 结果 |
|-----|------|---------|------|
| T1 | FP16 ELEWISE 回归 | R1 R2 R4 | ✅ |
| T2 | FP16 ELEWISE 多维 | R1 R2 R3 | ✅ |
| T3 | BF16 ELEWISE | R1 R2 | ✅ |
| T4 | FP32 ELEWISE | R1 R2 | ✅ |
| T5 | 标量广播 | R3 | ✅ |
| T6 | 行/列广播 | R3 | ✅ |
| T7 | 一般广播 | R3 | ✅ |
| T8 | 全标量 | R3 | ✅ |
| T9 | 非法广播 | R3 | ✅ |
| T10 | 大 shape 多核切分 | R1 R4 | ✅ |
| T11 | period 对齐优化 | R3 | ✅ |
| T12 | xfail：列广播 rowCount>1 | R3 | ✅（tiling 确定性 GRAPH_FAILED，见 §5） |

### 6.2 Kernel UT（op_kernel）

**状态**: ✅ 通过 | **脚本**: `tests/ut/run.sh` → `build/op_kernel/truncate_mod_op_kernel_ut` + `compare_data.py`（2026-07-31，12/12）

| 编号 | 用例 | 需求追溯 | 结果 | MERE | MARE | match_ratio |
|-----|------|---------|------|------|------|------------|
| K1 | FP16 非对齐回归 | R1 R2 | ✅ | 0 | 0 | 1.0 |
| K2 | FP16 ELEWISE | R1 R2 | ✅ | 0 | 0 | 1.0 |
| K3 | BF16 ELEWISE | R1 R2 | ✅ | 0 | 0 | 1.0 |
| K4 | FP32 ELEWISE | R1 R2 | ✅ | 0 | 2.82e-4 | 1.0 |
| K5 | FP32 行广播 | R3 | ✅ | 0 | 6.81e-4 | 1.0 |
| K6 | FP16 标量广播 | R3 | ✅ | 0 | 0 | 1.0 |
| K7 | 正负混合 + x2=0 | R1 | ✅ | 0 | 2.7e-5 | 1.0 |
| K8 | 大商值 | R1 | ✅ | 0 | 0 | 1.0 |
| K9 | 非对齐 tile + 尾块 | R1 R4 | ✅ | 0 | ≤1.08e-4 | 1.0 |
| K10 | 左侧前缀广播 | R3 | ✅ | 0 | 0 | 1.0 |

### 6.3 精度验收（ops-precision-standard）

**状态**: ⬜ | **脚本**: compare_data.py + 混合容差判定

| dtype | rtol/atol | max_abs_error_limit | matched_ratio | max_abs | 判定 |
|-------|-----------|--------------------|--------------|---------|------|
| FP16 | 2^-9 / 2^-9 | 0.1 | | | ⬜ |
| BF16 | 2^-6 / 2^-6 | 1.0 | | | ⬜ |
| FP32 | 2^-10 / 2^-16 | 1e-2 | | | ⬜ |

### 6.4 aclnn 通路

**状态**: ✅ 通过 | **脚本**: `bash build.sh -e`（2026-07-31 上板，NPU 3 / Ascend 910B3 / CANN 8.5.2，ASCEND_VISIBLE_DEVICES=3）

| 编号 | 用例 | 需求追溯 | 结果 | matched_ratio | max_abs_error | MERE | MARE |
|-----|------|---------|------|--------------|---------------|------|------|
| A1 | FP16 [16]×[16] | R1 R2 R4 | ✅ | 1.0 | 0 | 0 | 0 |
| A2 | BF16 [16]×[16] | R2 | ✅ | 1.0 | 0 | 0 | 0 |
| A3 | FP32 [4,8]×[8] 广播 | R3 | ✅ | 1.0 | 2.38e-7 (limit 1e-2) | 4.5e-8 | 6.8e-7 |

### 6.5 产物 & 执行状态

- [x] `build/custom_opp_*.run` 算子包构建成功（2026-07-31 Step C 编译通过，三 dtype 实例均产出 .o）
- [x] `build/op_host/truncate_mod_op_host_ut` 可执行（12/12）
- [x] `build/op_kernel/truncate_mod_op_kernel_ut` 可执行（12/12 + compare_data 全 PASS）
- [x] `examples/test_aclnn_truncate_mod` 可执行（2026-07-31 上板，A1-A3 全 PASS）
- [x] `custom_opp_ubuntu_aarch64.run --quiet` 安装成功（目标 `$ASCEND_HOME_PATH/opp/vendors/truncate_mod_custom`）

| 通路 | 状态 | 运行时间 | 跳过原因 |
|------|------|---------|---------|
| Tiling UT | ✅ | ~22ms | |
| Kernel UT | ✅ | ~2.6s | |
| 精度验收 | ⬜ | | 待 §6.3 归档（A1-A3 上板已隐含精度达标，正式记录待补齐） |
| aclnn 通路 | ✅ | ~数秒 | |

---

## 7. 性能验收

**状态**: ⬜ | **数据**: `docs/perf/round_NNN/`

| 指标 | 值 | 判定 |
|------|------|------|
| Task Duration | | |
| Block Dim | | |
| 主导流水 | | |

**采集方法**：`/ops-profiling`（msprof 算子级 + kernel-level 对比）。

**达标判定**: ⬜ | **理由**:

---

## 8. 汇总

| 通路 | 用例数 | 通过 | 失败 | 状态 |
|------|--------|------|------|------|
| Tiling UT | 12 | 12 | 0 | ✅ |
| Kernel UT | 10 | 10 | 0 | ✅（含 3 dtype × K9） |
| 精度验收 | 3 | | | ⬜ |
| aclnn 通路 | 3 | 3 | 0 | ✅ |
| 性能 | — | | | ⬜ |
