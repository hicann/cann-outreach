# Abs 算子设计串讲 — 质疑清单

> **串讲人**：Developer (设计串讲模式)
> **日期**：2026-05-23
> **审查范围**：§1.2 (API 映射)、§1.3 (数据流)、§1.4 (核心步骤)、§1.5 (Buffer 规划)、§2.1 (多核切分)、§2.4 (伪代码)
> **跳过**：§0 (概述)、§4 (实施计划)、§5 (确认清单)

---

## 审查结论

- [ ] 设计可直接开发（无阻塞问题）
- [x] 设计需要修改后开发（有阻塞/讨论问题）
- [ ] 设计存在严重问题，无法开发

---

## 质疑清单

---

### 问题 1：伪代码中 Abs 结果与 CopyOut 源 Buffer 不匹配（VECIN ↔ VECOUT）

- **类别**：伪代码可实现性
- **严重程度**：🔴 阻塞
- **设计文档位置**：§2.4.1 (行 199-225)，§1.5 (行 138-146)

#### 问题描述

设计中的 Buffer 规划表和伪代码存在不可调和的矛盾：

**Buffer 规划表 (§1.5)**：
| Buffer | TPosition | 大小 |
|--------|-----------|------|
| `inQueue` | VECIN | `ubFormer * sizeof(half)` |
| `outQueue` | VECOUT | `ubFormer * sizeof(half)` |

**伪代码 (§2.4.1, 行 214-224)**：
```cpp
DataCopyPad(inQueue, xGm[offset], ...);  // CopyIn → VECIN (inQueue)     ✅
Abs<half>(inQueue, inQueue, ubFormer);   // 结果在 VECIN (inQueue)        ✅
DataCopyPad(yGm[offset], outQueue, ...); // CopyOut 从 VECOUT (outQueue)  ❌
```

**问题**：`Abs` in-place 计算后结果在 `inQueue` 中（逻辑位置 VECIN），但 `DataCopyPad` 输出时从 `outQueue`（逻辑位置 VECOUT）读取数据。`outQueue` 从未被写入有效数据，CopyOut 会搬运脏数据（未初始化）到 GM。

**API 依据**：
- `Abs< half >(dst, src, count)` — 当 `dst == src` 时，结果写入 dst 所指向的 LocalTensor。根据 §1.3 说明，Abs Level 2 API 支持 `dst == src` 别名（in-place），结果存放在 `dst`/`src` 所指的物理位置（这里是 VECIN 的 `inQueue`）。
- `DataCopyPad(GM, LocalTensor, ...)` — 对于 UB→GM 方向，从 `LocalTensor& src` 读取数据搬往 GM。`src` 可以是 VECIN/VECOUT，但必须包含有效数据。
- `DataCopyPad` 不会自动在 VECIN 和 VECOUT 之间搬运数据 — 二者是独立的物理 buffer。

#### Developer 视角

实现者看到这个伪代码会陷入两难：
1. **如果按照 Buffer 规划表实现**：inQueue 在 VECIN，outQueue 在 VECOUT。Abs 写入 VECIN，但 CopyOut 从 VECOUT 读 → 读到脏数据 → 精度全错。
2. **如果忽略 outQueue**：CopyOut 改用 inQueue（VECIN）→ 会导致 outQueue 分配浪费 UB 空间（约 96 KB），且与 Buffer 规划表不一致。
3. **如果想修正**：需要在 Abs 之后、CopyOut 之前增加一步 VECIN→VECOUT 的数据搬运 → 额外引入 MTE 开销，违背设计初衷。

#### 建议方案

**方案 A（推荐 — 单 buffer in-place）**：
修改伪代码，CopyOut 从 `inQueue` 读取，移除 `outQueue` 的分配：
```cpp
DataCopyPad(inQueue, xGm[offset], ...);   // GM → UB (VECIN)
Abs<half>(inQueue, inQueue, ubFormer);     // in-place (VECIN)
DataCopyPad(yGm[offset], inQueue, ...);    // UB → GM (从 VECIN 读取)
// 不再使用 outQueue
```
- 优势：节省 96 KB UB 空间，代码简单
- 代价：无法使用双缓冲流水线（但 Abs 极轻量，双缓冲收益不大）

**方案 B（双 buffer 非 in-place）**：
修改伪代码，将 Abs 改为非 in-place 写入 outQueue：
```cpp
DataCopyPad(inQueue, xGm[offset], ...);    // GM → UB (VECIN)
Abs<half>(outQueue, inQueue, ubFormer);     // 结果写入 VECOUT
DataCopyPad(yGm[offset], outQueue, ...);   // UB → GM (从 VECOUT 读取)
```
- 优势：满足双缓冲流水线设计，CopyIn/Compute/CopyOut 可在不同 buffer 上并行
- 代价：多占 ~96 KB UB 空间（DAG 场景下可能 UB 不够用），需核对 UB 预算

**无论选哪个方案，Buffer 规划表和伪代码必须同时更新以保持一致。**

---

### 问题 2：[32,4096] 场景的 coreNum 不一致（公式 vs 表格）

- **类别**：多核策略可行性
- **严重程度**：🟡 需讨论
- **设计文档位置**：§2.1 (行 157-167)

#### 问题描述

**§2.1 核心公式 (行 157)**：
```
coreNum = min((dim0 * 16 + 32767) / 32768, availableCoreNum)
```
对 [32,4096]，`dim0=131072`，代入公式得：
```
(131072 * 16 + 32767) / 32768 = 2129919 / 32768 = 64
coreNum = min(64, availableCoreNum)  // 若 availCores=28 → 28
```

**§2.1 表格 (行 166)**：
| [32,4096] | 131072 | `min(availableCoreNum, 4)` → 动态 |

**问题**：表格中核心数上限写死为 `4`，但公式计算出 `64`（或 min(64, availCores) ≈ 28）。两者不一致。

**影响分析**：
- 如果按公式用 28 核：`blockFormer = ceil(131072/28) = 4681`，对齐到 512 得 `5120`。前 27 核处理 `27×5120 = 138240` > 总元素 `131072` → 尾核分配为负数，**越界崩溃**。
- 如果按表格用 4 核：`blockFormer = ceil(131072/4) = 32768`（已对齐），各核分配正常。但浪费了多核资源。

**Developer 视角**：实现者不知道应该按公式还是按表格实现。如果按公式实现，[32,4096] 场景会触发越界。

#### 建议方案

1. **明确采用哪个方案**：
   - **选项 A（推荐 — 限制核心数）**：将 [32,4096] 的核心数上限设为 4~8，避免 blockFormer 对齐后的尾核溢出。
   - **选项 B（用公式但改 blockFormer 计算）**：保持公式 `ceil(dim0 / coreNum)` 然后对齐到 512，但尾核用精确剩余数而非 round-up 后的 blockFormer。

2. **不论选哪个，都需要**：
   - 统一公式和表格的数字
   - 在 Tiling 代码中处理尾核的边界检查：`tailElements = totalElements - (coreNum-1) * blockFormer`
   - 验证 `tailElements > 0`，否则 `coreNum -= 1` 重算

---

### 问题 3：Pipeline（TQue/TPipe）架构未在伪代码中落地

- **类别**：伪代码可实现性
- **严重程度**：🟡 需讨论
- **设计文档位置**：§0.3 (行 39-42)、§1.4 (行 119-123)、§2.4.1 (行 199-225)

#### 问题描述

**§0.3 技术路线**声明：
> 技术路线：通用 SIMD/MemBase（TPipe + TQue 流水线）

**§2.4.1 伪代码**中实际呈现的是裸 `DataCopyPad` 调用，没有任何 `TQue` 的 `AllocTensor` / `EnQue` / `DeQue` / `FreeTensor` 操作：

```cpp
// 设计伪代码（简化版）
DataCopyPad(inQueue, xGm[offset], ...);  // inQueue 是 LocalTensor 还是 TQue？
Abs<half>(inQueue, inQueue, ubFormer);
DataCopyPad(yGm[offset], outQueue, ...);
```

**问题**：
1. `inQueue` / `outQueue` 在伪代码中被当作 `LocalTensor` 直接使用，但在 TQue 流水线模式中，它们应该是 `TQue` 类型，需要通过 `AllocTensor` 获取真正的 `LocalTensor`。
2. 缺少 `EnQue` / `DeQue` 操作，无法实现 MTE2/MTE3 与 Vector 计算的流水线并行。

**API 依据**：
- [api-buffer.md] 明确说明：MTE2/MTE3 搬运缓冲区应使用 `TQue<VECIN/VECOUT>`，并经过 `EnQue/DeQue` 管理。
- 裸 `LocalTensor` 直接传入 `DataCopyPad` 是 TBuf 模式（非流水线），无法利用 MTE+Vector 并行。

**影响分析**：
- 实现者无法判断该用 TBuf 还是 TQue 模式——决定影响代码结构差异很大
- 如果设计意图是用 TQue 双缓冲流水线，伪代码缺少 EnQue/DeQue 步骤可能导致实现者遗漏同步，引入数据竞争
- 如果设计意图是简单单缓冲（无流水线），则 §0.3 的技术路线声明有误导性

#### 建议方案

**在伪代码中补充 TQue 的完整生命周期操作**，或明确声明使用 TBuf 简化模式。两种模式各有利弊：

| 模式 | 适用场景 | 收益 | 代价 |
|------|---------|------|------|
| **TQue 双缓冲流水线** | MTE 开销大的场景 | 搬入/计算/搬出三阶段并行 | 代码复杂度高，至少需要 3 段函数 |
| **TBuf 单缓冲** | 计算极轻量的算子（如 Abs） | 代码简单，UB 占用少 | 不能掩盖搬运延迟 |

推荐 Abs 这种极轻量算子使用 **TBuf 单缓冲**模式（DataCopyPad + 裸 LocalTensor），去掉 TQue 描述以降低复杂度。如果坚持使用流水线，需要补充：
```cpp
// CopyIn 段
auto xLocal = inQueueX.AllocTensor<half>();
DataCopyPad(xLocal, xGm[offset], copyParams, padParams);
inQueueX.EnQue(xLocal);

// Compute 段
auto xCompute = inQueueX.DeQue<half>();
auto yLocal = outQueueY.AllocTensor<half>();
Abs<half>(yLocal, xCompute, ubFormer);  // 非 in-place
outQueueY.EnQue(yLocal);
inQueueX.FreeTensor(xCompute);

// CopyOut 段
auto yOut = outQueueY.DeQue<half>();
DataCopyPad(yGm[offset], yOut, copyParams);
outQueueY.FreeTensor(yOut);
```

---

### 问题 4：ubFormer 计算假设双缓冲，但 Compute 用 in-place 模式，UB 空间浪费

- **类别**：内存规划合理性
- **严重程度**：🟢 建议
- **设计文档位置**：§1.5 (行 138-146)、§2.2 (行 174-180)

#### 问题描述

**§2.2 (行 177)**：
```cpp
maxElemNum = UB_size / (2 * sizeof(half)) = 196608 / 4 = 49152
```
这里的 `(2 * sizeof(half))` = `4`，隐含假设：UB 空间被 2 份 buffer（in + out）均分，因此每个 buffer 最多使用 UB 的 1/2。

**§1.5 (行 140-141)** 规划了 inQueue + outQueue 两个 buffer，每个 `ubFormer * sizeof(half)` = 98304 字节，合计 196608 字节 = 192 KB = 全部 UB。

**§2.4.1 伪代码**中 Compute 使用 in-place 模式 (`inQueue` 既是输入又是输出)，不需要独立的输入和输出 buffer。

**问题**：
- In-place 模式下，实际只需要 1 份 buffer 持有有效数据，另一份 `outQueue` 完全闲置但仍占用 96 KB UB
- `ubFormer` 计算基于 2 份 buffer 假设，将每个 buffer 的容量限制为 UB/2
- 如果改用 in-place 单 buffer，每个 tile 可以处理最多 `196608 / 2 = 98304` 个元素（而不是 49152），提升 2× 吞吐

#### 建议方案

- 明确决定使用**单 buffer in-place 模式**（问题 1 的方案 A），然后：
  - `maxElemNum = UB_size / sizeof(half) = 98304`（仅 1 个 buffer 的约束）
  - `ubFormer = min(98304, blockFormer)`，对齐到 128 元素
  - 删除 `outQueue` 分配，只用 `inQueue`
- 或明确决定使用**双 buffer 非 in-place 模式**（问题 1 的方案 B），保持当前规划但更新伪代码

---

### 问题 5：blockFormer 对齐到 512 元素的理由不充分

- **类别**：多核策略可行性
- **严重程度**：🟢 建议
- **设计文档位置**：§2.1 (行 155-156)

#### 问题描述

**§2.1 (行 156)**：
> 单核任务量：`blockFormer` 元素（对齐到 512 元素）

但 **§2.2 (行 178)** 的 ubFormer 对齐是 128 元素（256B）。

**问题**：blockFormer 对齐到 512 元素的理由是什么？为什么不直接用 ubFormer 的对齐粒度（128 元素）？

**影响分析**：
- 512 元素 = 1024 字节。更大的对齐粒度意味着更粗粒度的核心任务分配。
- 对于 [32,4096] 场景用了较多核心时（如 28 核），512 对齐导致 blockFormer 从 4681 向上取整到 5120，每核浪费 `5120-4681 = 439` 个元素（约 9%），累积 27 核浪费 11853 个元素。
- 如果使用 128 元素对齐，blockFormer = `ceil(4681/128) * 128 = 4736`，浪费 `4736-4681 = 55` 个元素（约 1.2%），浪费大幅降低。

#### 建议方案

- 如果 512 对齐有特定硬件理由（如 L1 cache line 大小），请在文档中标注
- 如果没有硬件约束，建议改为 128 元素对齐（与 ubFormer 一致），减少多核负载不均

---

### 问题 6：伪代码中 DataCopyPad 的 CopyOut 方向缺少 padParams 参数

- **类别**：API 可行性
- **严重程度**：🟢 建议
- **设计文档位置**：§2.4.1 (行 218, 224)

#### 问题描述

**设计文档 (行 218)**：
```cpp
DataCopyPad(yGm[offset], outQueue, copyParams);
```

**设计文档 (行 224)**：
```cpp
DataCopyPad(yGm[offset], outQueue, copyParamsTail);
```

**API 依据**：
查阅`DataCopyPad(ISASI).md`：
```
// UB→GM 方向 — DataCopyParams 类型
__aicore__ inline void DataCopyPad(
    const GlobalTensor<T>& dst, 
    const LocalTensor<T>& src,
    const DataCopyParams& dataCopyParams);
```
UB→GM 方向确实没有 `padParams`，这与设计一致。所以这不是 bug。

但问题在于：`copyParams` 和 `copyParamsTail` 的 `blockLen` 参数必须传入**有效元素**的字节数（非对齐也可以，DataCopyPad 会自动补齐假数据并在搬到 GM 时丢弃）。如果传入了对齐后的字节数，会出现数据错位。

**Developer 视角**：实现者需要知道 `copyParams.blockLen` 到底传什么值——是有效数据长度还是对齐后长度。设计文档应在参数规则（§1.4 行 131-135）中补充说明 CopyOut 方向的有效长度规则。

#### 建议方案

在 §1.4 参数使用规则表中，补充 CopyOut 方向的说明：

| 参数位置 | 用有效长度 | 用对齐长度 |
|---------|-----------|-----------|
| CopyIn DataCopyPad blockLen / Abs count | ✓ | ✗ |
| **CopyOut DataCopyPad blockLen** | **✓** | **✗** |
| UB 数据偏移 / Buffer 大小 | ✗ | ✓ |

---

## 汇总

| # | 严重程度 | 问题 | 类别 |
|---|---------|------|------|
| 1 | 🔴 阻塞 | 伪代码中 Abs 结果在 VECIN，CopyOut 从 VECOUT 读 — 数据错误 | 伪代码可实现性 |
| 2 | 🟡 需讨论 | [32,4096] coreNum 公式(64) vs 表格(4) 不一致，尾核可能越界 | 多核策略 |
| 3 | 🟡 需讨论 | Pipeline(TQue)架构声明但伪代码是裸 DataCopyPad，无 EnQue/DeQue | 伪代码可实现性 |
| 4 | 🟢 建议 | ubFormer 按双缓冲规划但不能与 in-place 计算共存，UB 浪费 50% | 内存规划 |
| 5 | 🟢 建议 | blockFormer 512 对齐粒度缺乏理由，导致多核负载不均 | 多核策略 |
| 6 | 🟢 建议 | CopyOut 方向参数规则需在 §1.4 明确补充 | API 可行性 |

### 关键决策路径

建议 Architect 优先解决以下依赖链：

1. **决定 Buffer 模式**（单 buffer in-place vs 双 buffer 非 in-place）→ 解决问题 1、4
2. **决定 Pipeline 架构**（TQue 流水线 vs TBuf 单缓冲）→ 解决问题 3
3. **决定 [32,4096] 核心数上限**（公式 vs 表格）→ 解决问题 2
4. 对齐参数文档化 → 解决问题 5、6

---

## 设计串讲仲裁

> **仲裁人**：CANNBot（基于官方文档）
> **日期**：2026-05-23

### Architect 回应

#### 问题 1：伪代码中 Abs 结果在 VECIN，CopyOut 从 VECOUT 读 — 数据错误

- **回应**：✅ **接受（采纳方案 A — 单 buffer in-place）**
- **理由**：Developer 的分析完全正确。原设计伪代码中的 `DataCopyPad(yGm[offset], outQueue, ...)` 与 `Abs` 写入 `inQueue` 存在 buffer 不匹配，会导致 CopyOut 读出脏数据。
- **文档依据**：
  - `DataCopyPad(ISASI).md` 明确标注 UB→GM 方向签名：`DataCopyPad(const GlobalTensor<T>& dst, const LocalTensor<T>& src, const DataCopyParams& dataCopyParams)`，`src`可以是 VECIN 或 VECOUT，无限制（[DataCopyPad(ISASI).md #L36-L44]）。
  - `Abs.md` 明确展示 `Abs(dstLocal, srcLocal, 512)` 的 tensor前n个数据计算接口支持 `dst==src` 别名（[Abs.md #L148-L152]）。
- **DESIGN.md 变更**：
  1. §1.3 数据流：删除 VECOUT 路径，改为单一 VECIN 流（CopyIn→Abs(in-place)→CopyOut 全在 VECIN）
  2. §1.5 Buffer 规划：删除 `outQueue`，仅保留单 buffer `inQueue`（VECIN）
  3. §2.4.1 伪代码：所有 `DataCopyPad(yGm[offset], outQueue, ...)` 改为 `DataCopyPad(yGm[offset], inQueue, ...)`
  4. Buffer 需求表：删除 outQueue 行

---

#### 问题 2：[32,4096] 场景的 coreNum 不一致（公式 vs 表格）

- **回应**：✅ **接受 + 改进**
- **理由**：Developer 指出的不一致确实存在。公式计算出 64（min with availCores 后 28），但表格写的是 4。同时 512 元素对齐导致尾核越界（27×5120=138240>131072→tail=-7168）。
- **解决方案**：
  1. blockFormer 对齐从 512 改为 **128 元素**（与 ubFormer 一致，见 Issue 5）
  2. 128 对齐后，28 核场景：blockFormer=4736, 27×4736=127872<131072, tail=3200 ✓（**无越界**）
  3. 表格更新为反映公式计算结果
  4. Tiling 仍保留 `tailElements > 0` 安全校验，若为负则 `coreNum -= 1` 重算
- **文档依据**：无硬件约束要求 512 对齐；ubFormer 对齐 256B=128元素是 Vector 指令的标准粒度。
- **DESIGN.md 变更**：
  1. §2.1 blockFormer 对齐改为 128 元素
  2. §2.1 表格更新：[1,128] blockFormer=128, [32,4096] coreNum 改回公式值
  3. §2.1 增加安全校验标注

---

#### 问题 3：Pipeline（TQue/TPipe）架构未在伪代码中落地

- **回应**：✅ **接受 — 技术路线从 TQue 流水线改为 TBuf 单缓冲**
- **理由**：Developer 的质疑合理。原设计声明 TQue 流水线但伪代码无 EnQue/DeQue。重新评估后：
  - Abs 是**极轻量计算**（~1 cycle/element），Vector 计算时间 << MTE 搬运时间
  - TQue 双缓冲流水线的复杂度远大于其对 Abs 的性能收益
  - TBuf 单缓冲（Direct LocalTensor + DataCopyPad）更匹配 Abs 的简单特性
  - UB 占用更少，代码结构更清晰
- **文档依据**：示例代码 `basic_api_memory_allocator_add/` 展示了 TQue 流水线 vs 非流水线的模式差异。对于轻量计算，单缓冲模式更简洁高效。
- **DESIGN.md 变更**：
  1. §0.1 / §0.3：技术路线从"Pipeline + TQue/TPipe"改为"TPipe + 单缓冲 Sequential 模式"
  2. §0.3 理由栏：标注"Abs 计算极轻量，单缓冲 in-place 即可，无需 TQue 流水线"
  3. §2.4.1 伪代码添加注释说明架构选择

---

#### 问题 4：ubFormer 计算假设双缓冲，但 Compute 用 in-place 模式，UB 浪费

- **回应**：✅ **接受**
- **理由**：Developer 正确指出 in-place 模式下只需要 1 份 buffer。原设计 `maxElemNum = UB_size / (2 * sizeof(half)) = 49152` 是基于双 buffer 假设。改用单 buffer in-place 后，`maxElemNum = UB_size / sizeof(half) = 98304`，提升 2× 单 tile 吞吐。
- **影响分析**：对于三个目标 shape（[1,128], [4,2048], [32,4096]），blockFormer 均不超过 98304（最大 32768/4736），每个 core 仅需 1 个 tile，**不再需要循环分 chunk**。
- **DESIGN.md 变更**：
  1. §2.2：maxElemNum 从 `49152` 改为 `98304`
  2. §1.5：总 UB 使用量从 `ubFormer * sizeof(half) * 2` 改为 `ubFormer * sizeof(half)`

---

#### 问题 5：blockFormer 对齐到 512 元素的理由不充分

- **回应**：✅ **接受**
- **理由**：Developer 的分析正确。512 元素对齐（1024B）缺乏硬件约束依据，且对大核心数场景导致负载不均（[32,4096] 28 核时每核浪费 439 元素 ≈ 9%）。改为 128 元素对齐（256B，与 ubFormer 一致），每核浪费降至 55 元素 ≈ 1.2%。
- **同步收益**：128 对齐后，28 核场景下尾核不再越界（见 Issue 2）。
- **DESIGN.md 变更**：
  1. §2.1 单核任务量对齐粒度：512 → 128 元素
  2. §2.1 表格 [1,128] blockFormer: 512 → 128
  3. §2.1 附注消除

---

#### 问题 6：伪代码中 DataCopyPad 的 CopyOut 方向缺少 padParams 参数

- **回应**：✅ **接受 — 补充参数规则说明**
- **理由**：Developer 指出实现者需要明确 CopyOut 方向 `blockLen` 的传值规则。原 §1.4 参数规则表已暗示 `DataCopyPad blockLen` 用有效长度，但未显式区分 CopyIn/CopyOut。现已扩展为详细表格，明确标注：
  - CopyIn/CopyOut DataCopyPad blockLen 均使用 **有效元素字节数**
  - UB Buffer 分配大小使用**对齐长度**（256B=128元素）
- **文档依据**：DataCopyPad(ISASI).md 对 VECIN/VECOUT→GM 方向说明："框架在搬出时会自动补充一些假数据来保证对齐，搬到 GM 时会自动将填充的假数据丢弃掉"（[DataCopyPad(ISASI).md #L234]）。
- **DESIGN.md 变更**：
  1. §1.4 参数使用规则表：从 2 行扩展为 4 行，分别明确 CopyIn blockLen/CopyOut blockLen/Abs count/UB Buffer 分配规则

---

### 回应统计

| # | 严重程度 | 问题 | 回应 | DESIGN.md 变更 |
|---|---------|------|------|---------------|
| 1 | 🔴 阻塞 | Abs 结果在 VECIN，CopyOut 从 VECOUT 读 | ✅ 接受（方案 A） | §1.3, §1.5, §2.4.1 |
| 2 | 🟡 讨论 | [32,4096] coreNum 公式 vs 表格不一致 | ✅ 接受（128对齐修复越界） | §2.1 |
| 3 | 🟡 讨论 | Pipeline 架构与伪代码不匹配 | ✅ 接受（改为 TBuf 单缓冲） | §0.1, §0.3, §2.4.1 |
| 4 | 🟢 建议 | ubFormer 按双缓冲但 in-place 浪费 | ✅ 接受（maxElemNum 翻倍） | §2.2, §1.5 |
| 5 | 🟢 建议 | blockFormer 512 对齐缺乏理由 | ✅ 接受（改为 128 对齐） | §2.1 |
| 6 | 🟢 建议 | CopyOut 参数规则需补充 | ✅ 接受（表格扩展） | §1.4 |

- **接受**：6 项 | **保留原设计**：0 项 | **部分修改**：0 项
- **DESIGN.md 已更新**：是（§0.1, §0.3, §1.3, §1.4, §1.5, §2.1, §2.2, §2.4.1, §3）
