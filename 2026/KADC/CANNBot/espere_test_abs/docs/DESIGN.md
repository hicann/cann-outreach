# Abs 算子设计与实施文档

> **算子名称**: `abs`
> **支持芯片**: Ascend910B3 (DAV_2201 / arch22)

---

## 0. 概述

### 0.0 需求类型判断

**特定用例**：用户明确指定了具体的 shape 集合和 dtype（`float16`，shape 为 `[1,128]`、`[4,2048]`、`[32,4096]`）。

### 0.1 基本信息

| 项目 | 内容 |
|-----|------|
| 算子名称 | `abs` |
| 算子类别 | Elementwise（逐元素一元运算） |
| 需求类型 | 特定用例（shape=[1,128]/[4,2048]/[32,4096], dtype=float16） |
| 支持数据类型 | float16 (half) |
| 支持芯片 | Ascend910B3 (DAV_2201) |
| 特殊约束 | 无 |
| 技术路线 | 通用 SIMD/MemBase（TPipe + 单缓冲 Sequential 模式） |

### 0.2 环境信息

| 项目 | 值 |
|-----|-----|
| CANN 版本 | 8.5.2 |
| 芯片型号 | Ascend910B3 (NpuArch DAV_2201) |
| UB 容量 | 192 KB (196608 bytes) |
| 架构目录 | aarch64-linux |

### 0.3 方案决策依据

| 决策项 | 选择 | 理由 |
|-------|------|------|
| 技术路线 | 通用 SIMD/MemBase（TPipe + 单缓冲 Sequential 模式） | 目标架构 DAV_2201（非 DAV_3510），走通用 Vector 路线；Abs 计算极轻量，单缓冲 in-place 模式可满足性能要求，无需 TQue 流水线 |
| 计算 API | `Abs< half >(dst, src, count)` Level 2 批量接口 | 最简单的 tensor 前 n 个数据计算接口，适合连续内存的 1D 场景 |
| 数据搬运 | `DataCopyPad` | 自动处理非对齐边界，避免 tile 边界对齐问题 |
| 精度策略 | 原 dtype 直接计算 | Abs 为逐元素操作，无"大数吃小数"风险，无需升精度 |

---

## 1. 算子设计

### 1.1 数学公式

```
// 输入输出定义
输入: x - shape=[D0, D1] 或展平 1D, dtype=float16
输出: y - shape=[D0, D1], dtype=float16

// 数学公式
y[i] = |x[i]|   (逐元素取绝对值)
```

### 1.2 API 映射

| 数学操作 | 对应 API | 关键参数 | 数据布局 | 官方文档 |
|---------|---------|---------|---------|---------|
| 绝对值计算 | `Abs< half >(dst, src, count)` | 前 count 个元素 | 连续 1D LocalTensor<half> | [Abs.md](asc-devkit/docs/api/SIMD-API/基础API/Memory矢量计算/基础算术/Abs.md) |
| GM → UB 搬运 | `DataCopyPad(dst, src, copyParams, padParams)` | blockLen=有效元素×2B | 连续 GM → LocalTensor | [api-datacopy.md] |
| UB → GM 搬运 | `DataCopyPad<dstType>(gm, ub, copyParams)` | 输出回写 | LocalTensor → GM | [api-datacopy.md] |

#### 1.2.1 API 语义验证

**API 验证表**：

| API | 数据布局 | 功能需求 | API选择 | 限制条件 | 匹配 | 文档 |
|-----|---------|---------|---------|---------|-----|------|
| `Abs< half >` | 连续 1D LocalTensor，32B 对齐 | dst[i] = \|src[i]\|, count 个元素 | `Abs(LocalTensor<T>& dst, LocalTensor<T>& src, const int32_t& count)` | dst/src 起始地址需要 32 字节对齐；half 支持 | ✅ | [Abs.md] |
| `DataCopyPad` | GM → UB，连续数据 | 搬运有效数据，处理非对齐 | `DataCopyPad(local, global[offset], copyParams, padParams)` | 无特殊对齐限制 | ✅ | [api-datacopy.md] |

**验证清单**：
- [x] 1. 数据布局确认：连续 1D 排列，UB Tensor 分配保证 32B 对齐
- [x] 2. 功能需求明确：一元逐元素绝对值，无跨元素依赖
- [x] 3. 已查阅官方文档：`Abs.md`（SIMD-API 基础算术）+ `asc_abs.md`（c_api vector_compute）
- [x] 4. 匹配验证：half 类型在 DAV_2201 完全支持，32B 对齐由 TPipe 分配保证
- [x] 5. 已记录验证过程

**验证方法 QA**：
```
问题 1：数据布局是什么？
    ├─ 内存如何排列？— 连续 1D，展平后线性访问
    ├─ 是否对齐？— TPipe AllocTensor 默认 32 字节对齐
    └─ 输入输出格式？— 向量（1D LocalTensor）

问题 2：需要什么操作？
    ├─ 操作类型？— elementwise 逐元素计算
    ├─ 操作维度？— 1D 线性处理
    └─ 特殊要求？— 无

问题 3：API 能实现吗？
    ├─ 查阅官方文档了吗？— ✅ Abs.md 确认
    ├─ API 适用场景对吗？— ✅ 连续 1D 精确匹配
    ├─ 满足 API 限制吗？— ✅ 32B 对齐由 TPipe 保证
    └─ 有更好的选择吗？— Level 2 最简单，无更好选择
```

### 1.3 数据流

```
输入 x (GlobalTensor<half>, GM)
    ↓ DataCopyPad (GM → VECIN)
输入 xLocal (LocalTensor<half>, VECIN)       // 单一共享 buffer (inQueue)
    ↓ Abs(xLocal, xLocal, count)              // in-place 计算（dst=src 别名，结果仍在 VECIN）
输出 yLocal (LocalTensor<half>, VECIN)        // 结果与输入共享同一 buffer
    ↓ DataCopyPad (VECIN → GM)                // 从 VECIN 读出结果回写 GM
输出 y (GlobalTensor<half>, GM)
```

**说明**：`Abs` Level 2 API 支持 dst 与 src 指向同一 LocalTensor（地址重叠），因此可以 in-place 计算。DataCopyPad 的 VECIN/VECOUT→GM 方向均支持，从 VECIN 读出不存在限制。单 buffer in-place 模式仅需一份 UB buffer，大幅简化内存管理。

### 1.4 核心计算步骤

**核心计算步骤**：
```
1. CopyIn — GM 数据搬运到 UB (DataCopyPad)
2. Compute — 对 UB 中的 half 数据执行 Abs 运算
3. CopyOut — UB 结果写回 GM (DataCopyPad)
```

**关键设计要点**：
1. **In-place 计算**：`Abs` Level 2 支持 dst 与 src 别名（地址重叠），只需 1 个输入/输出共享 Buffer + 1 个临时 Buffer（可选），节省 UB 空间
2. **DataCopyPad 统一搬运**：所有 GM↔UB 搬运统一使用 `DataCopyPad`，避免对 tile 边界的非对齐判断
3. **UB 切分保证 256B 对齐**：ubFormer 对齐到 256B，确保 Vector 指令效率

**参数使用规则**：
| 参数位置 | 规则 | 说明 |
|---------|------|------|
| CopyIn DataCopyPad blockLen | 用有效元素字节数 | 框架在 UB 侧自动补齐至 32B 对齐，可借助 padParams 指定填充值 |
| CopyOut DataCopyPad blockLen | 用有效元素字节数 | 框架在 UB 侧补齐至 32B 对齐，回写 GM 时自动剥离 padding |
| Abs count | 用有效元素个数 | 仅处理有效元素，尾部非对齐元素在 CopyIn 时已处理 |
| UB Buffer 分配大小 | 对齐到 256B (128元素×sizeof(half)) | 保证 LocalTensor 32B 对齐约束 |

### 1.5 内存管理（Buffer 规划）

| Buffer 名称 | 用途 | 大小计算 | TPosition |
|------------|------|---------|-----------|
| inQueue | 输入数据 + in-place 计算 + 输出回写 | `ubFormer * sizeof(half)` | VECIN |

**总 UB 使用量**：
- 单 Buffer in-place：`ubFormer * sizeof(half)` ≈ 最大 192 KB（ubFormer=98304 时），但实际分配时会预留对齐余量

> **说明**：采用单 buffer in-place 模式，仅需 1 份 UB buffer。输入、计算、输出共享同一 inQueue（TPosition=VECIN）。DataCopyPad 的 VECIN→GM 方向在 API 层面完全支持，框架自动处理 32B 补齐后剥离 padding。

---

## 2. 架构设计

### 2.1 多核切分策略

| 项目 | 说明 |
|-----|------|
| 切分维度 | 展平后的元素总数 dim0，沿 1D 均分 |
| 单核任务量 | `blockFormer` 元素（对齐到 128 元素，与 ubFormer 对齐粒度一致） |
| 使用的核数 | **强制动态计算**：`coreNum = min((dim0 * 16 + 32767) / 32768, availableCoreNum)` |
| 负载均衡方式 | 前 N-1 核各处理 blockFormer 元素，最后一核处理剩余元素 |

**各 Shape 切分结果**：

| Shape | dim0 | coreNum | blockFormer | blockNum |
|-------|------|---------|-------------|----------|
| [1,128] | 128 | 1 | 128 | 1 |
| [4,2048] | 8192 | 4 | 2048 | 4 |
| [32,4096] | 131072 | `min((dim0*16+32767)/32768, availCores)` → 动态 | `ceil(131072/coreNum/128)*128` → 动态 | 动态 |

> 注：[1,128] 仅 256B，单核即可处理，无需多核。[4,2048] 16KB，4 核每核 4KB 为最小粒度。blockFormer 对齐粒度改为 128 元素（与 ubFormer 一致），避免粗粒度对齐导致的尾核负值问题。Tiling 阶段仍需验证 `tailElements > 0`，若为负则 `coreNum -= 1` 重算。

### 2.2 UB 切分策略

| 项目 | 说明 |
|-----|------|
| UB 容量 | 192 KB (DAV_2201) |
| 单次处理数据量 | `ubFormer = (maxElemNum / 128) * 128` 个元素 |
| 是否需要分 chunk | 当 `blockFormer > ubFormer` 时需要 |
| 最大可容纳元素 | `maxElemNum = UB_size / sizeof(half) = 196608 / 2 = 98304`（单 buffer in-place，仅需 1 份 buffer） |
| ubFormer 对齐 | 对齐到 `256B / sizeof(half) = 128` 元素 |
| ubFormer 典型值 | 49152（可容纳全部小 shape 单次处理） |

### 2.3 分支场景覆盖

| 分支条件 | 处理策略 |
|---------|---------|
| 数据类型 | 仅 float16，单一分支 |
| shape [1,128] (小) | 单核单 tile，ubFormer 覆盖全部元素 |
| shape [4,2048] (中) | 4 核，每核 2048 元素 |
| shape [32,4096] (大) | 多核，按 blockFormer/ubFormer 循环处理 |
| 对齐 | DataCopyPad 统一处理 |
| 非对齐 | DataCopyPad 天然支持 |
| 边界情况 | 尾 block 特殊处理（blockTail ≠ blockFormer） |

### 2.4 Elementwise 特有设计

#### 2.4.1 Abs 基础分支（所有 shape 共用）

**适用场景**：所有 shape 场景，float16 数据，逐元素绝对值运算。

**Compute 核心流程伪代码**：
```cpp
// 对于每个 block：
//   1. 计算当前 block 的偏移
//   2. 循环处理每个 UB tile
//      2a. DataCopyPad: GM → UB (inQueue, VECIN)
//      2b. Abs: dst = |src|  (in-place, 结果仍在 VECIN)
//      2c. DataCopyPad: UB → GM (从 inQueue/VECIN 读出)
//   3. 处理尾 tile
//
// 架构说明：单缓冲（TBuf Sequential）模式，不使用 TQue 流水线。
//           Abs 计算极轻量，流水线并行收益有限，单缓冲 in-place 可简化代码并节约 UB。

// Tiling 参数准备（Host 侧计算，通过 TilingData 传入 Kernel）
//   blockFormer     — 每核处理元素数（128 对齐）
//   ubFormer        — 每次 tile 处理元素数（128 对齐）
//   loopNum         — 完整 tile 数
//   tailNum         — 尾 tile 元素数

// Kernel 伪代码（单 buffer in-place 模式）：
auto offset = blockIdx * blockFormer;
auto loopNum = isLastBlock ? ubLoopOfTailBlock : ubLoopOfFormerBlock;
auto tailNum = isLastBlock ? ubTailOfTailBlock : ubTailOfFormerBlock;

for (uint32_t i = 0; i < loopNum; i++) {
    // 完整 tile 处理（BlockLen = ubFormer * sizeof(half)）
    DataCopyPad(inQueue, xGm[offset], copyParams, padParams);
    Abs<half>(inQueue, inQueue, ubFormer);
    DataCopyPad(yGm[offset], inQueue, copyParams);
    offset += ubFormer;
}
// 尾 tile 处理（BlockLen = tailNum * sizeof(half)）
if (tailNum > 0) {
    DataCopyPad(inQueue, xGm[offset], copyParamsTail, padParamsTail);
    Abs<half>(inQueue, inQueue, tailNum);
    DataCopyPad(yGm[offset], inQueue, copyParamsTail);  // 从 inQueue(VECIN) 回写
}
```

**Buffer 需求**：

| Buffer 名称 | 用途 | 大小计算 |
|------------|------|---------|
| inQueue | 输入数据 + in-place 计算 + 输出回写 | `ubFormer * sizeof(half)` |

> **注意**：单 buffer in-place 模式中 CopyOut 从 `inQueue`（VECIN）读取。DataCopyPad API 文档明确支持 VECIN/VECOUT→GM 通路，无 VECIN 限制。

---

## 3. 确认清单

- [x] 多核切分策略已确定 — 1D 均分，按 128 元素对齐
- [x] UB 切分策略已确定 — 按 256B 对齐，最大 98304 元素（单 buffer in-place）
- [x] Buffer 规划已完成 — 单 buffer inQueue（VECIN），输入/计算/输出三合一共享
- [x] 分支场景已覆盖 — 仅 float16，三个典型 shape
- [x] Elementwise 特有设计已完成 — 1D 展平 + DataCopyPad + Abs Level 2
- [x] API 验证已完成 — Abs/DataCopyPad 均已验证
- [x] 精度策略已确认 — 原 dtype 直算，float16 社区标准 MERE < 2^-10
