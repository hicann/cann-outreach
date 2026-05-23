# abs 算子技术设计文档

---

## 0. 概述

### 0.0 需求类型判断

**判断结果**：通用

**判断依据**：
- 用户未明确指定具体的 shape 和 dtype
- 明确要求支持 FP16、FP32 多种数据类型
- 明确要求支持任意形状的输入张量

### 0.1 基本信息

| 项目 | 内容 |
|-----|------|
| 算子名称 | abs |
| 算子类别 | Elementwise |
| 需求类型 | 通用 |
| 支持数据类型 | FP16 (half), FP32 (float) |
| 支持芯片 | Atlas A2/A3 训练系列产品, Atlas 200I/500 A2 推理产品, Ascend 950PR/Ascend 950DT, Kirin X90/Kirin 9030 |
| 特殊约束 | 输入输出 Shape 完全相同 |

---

## 1. 算子设计

### 1.1 数学公式

```
// 输入输出定义
输入: x - shape: [任意], dtype: half/float
输出: y - shape: [与x相同], dtype: half/float

// 数学公式
y = |x|
```

**数值稳定性说明**：
- abs 操作本身数值稳定，不涉及累加，无需升精度处理
- 直接按原 dtype 计算即可

### 1.2 API 映射

| 数学操作 | 对应 API | 关键参数 | 数据布局 | 官方文档 |
|---------|---------|---------|---------|---------|
| 数据搬运 GM→UB | DataCopyPad | dst(LocalTensor), src(GlobalTensor), copyParams, padParams | 连续布局 | [DataCopyPad](asc-devkit/docs/api/SIMD-API/基础API/Memory数据搬运/DataCopy/DataCopyPad.md) |
| 绝对值计算 | Abs | dst(LocalTensor), src(LocalTensor), count | 连续布局 | [Abs](asc-devkit/docs/api/SIMD-API/基础API/Memory矢量计算/基础算术/Abs.md) |
| 数据搬运 UB→GM | DataCopyPad | dst(GlobalTensor), src(LocalTensor), copyParams | 连续布局 | [DataCopyPad](asc-devkit/docs/api/SIMD-API/基础API/Memory数据搬运/DataCopy/DataCopyPad.md) |

**API 选择原则**：优先使用 DataCopyPad（来自 [api-datacopy.md §选择规则](.opencode/skills/ascendc-api-best-practices/references/api-datacopy.md#选择规则)）
- 自动处理非对齐，无需手动判断 32 字节对齐
- CopyIn 和 CopyOut 都适用
- Tiling 设计时可能产生非对齐的 tile 大小（尾块处理）
- 对齐场景下性能差异可忽略

#### 1.2.1 API 语义验证

**验证表格**（所有 API 必须填写）：

| API | 数据布局 | 功能需求 | API选择 | 限制条件 | 匹配 | 文档 |
|-----|---------|---------|---------|---------|-----|------|
| DataCopyPad (GM→UB) | 连续布局 | GM到UB的数据搬运，支持非对齐 | `DataCopyPad(LocalTensor<T>& dst, GlobalTensor<T>& src, const DataCopyParams& copyParams, const DataCopyPadParams& padParams)` | blockLen单位为字节，支持非对齐搬运 | ✅ | [链接](asc-devkit/docs/api/SIMD-API/基础API/Memory数据搬运/DataCopy/DataCopyPad.md) |
| Abs | 连续布局，32B对齐要求 | 逐元素绝对值计算 | `Abs(LocalTensor<T>& dst, LocalTensor<T>& src, int32_t count)` | 地址需32B对齐，支持half/float | ✅ | [链接](asc-devkit/docs/api/SIMD-API/基础API/Memory矢量计算/基础算术/Abs.md) |
| DataCopyPad (UB→GM) | 连续布局 | UB到GM的数据搬运，支持非对齐 | `DataCopyPad(GlobalTensor<T>& dst, LocalTensor<T>& src, const DataCopyParams& copyParams)` | blockLen单位为字节，自动处理非对齐 | ✅ | [链接](asc-devkit/docs/api/SIMD-API/基础API/Memory数据搬运/DataCopy/DataCopyPad.md) |

**验证清单**（每个 API 必须完成）：
- [x] 1. 数据布局确认（连续布局）
- [x] 2. 功能需求明确（GM搬运、绝对值计算、UB搬运）
- [x] 3. 已查阅官方文档（提供链接）
- [x] 4. 匹配验证（数据布局与 API 能力匹配、限制条件满足）
- [x] 5. 已记录验证过程

**API 约束详情**：

1. **Abs API 约束**：
   - 地址需 32 字节对齐
   - 支持数据类型：half、float（符合需求）
   - 计算公式：y = |x|

2. **DataCopyPad API 约束**：
   - CopyIn: `DataCopyPad(LocalTensor, GlobalTensor, DataCopyParams, DataCopyPadParams)`
   - CopyOut: `DataCopyPad(GlobalTensor, LocalTensor, DataCopyParams)`
   - blockLen 单位为字节，支持非对齐搬运
   - isPad=false 时框架自动填充（非对齐场景推荐）
   - 支持数据类型：half、float

### 1.3 数据流

```
输入 x (Global Tensor)
    ↓ DataCopyPad (GM → UB, CopyIn)
输入 x (Local Tensor - inQueueX)
    ↓ Abs (逐元素计算, Compute)
输出 y (Local Tensor - outQueueY)
    ↓ DataCopyPad (UB → GM, CopyOut)
输出 y (Global Tensor)
```

**数据流说明**：
- **CopyIn**：输入数据从 Global Memory 搬运到 Unified Buffer (UB)，使用 DataCopyPad
- **Compute**：在 UB 中进行逐元素绝对值计算
- **CopyOut**：计算结果从 UB 搬运回 Global Memory，使用 DataCopyPad

**Double Buffer 流水线并行时间线**：
```
Row 0: [MTE2-B0][Vector-B0][MTE3-B0]
Row 1:          [MTE2-B1][Vector-B1][MTE3-B1]
          ↑ MTE2 与 Vector 并行！
```
- MTE2: GM → UB（CopyIn）
- Vector: Abs 计算（Compute）
- MTE3: UB → GM（CopyOut）

### 1.4 核心计算步骤

**核心计算步骤概览**：
```
1. GM→UB数据搬运 - 将输入数据搬运到 UB
2. Abs 计算 - 在 UB 中逐元素计算绝对值
3. UB→GM数据搬运 - 将结果搬运回 GM
```

**关键设计要点**：
1. **Buffer 使用**：
   - inQueueX：输入数据缓存（VECIN）
   - outQueueY：输出数据缓存（VECOUT）
   - 无需中间 Buffer（abs 不需要临时存储）

2. **API 选择**：
   - 使用 Memory 矢量计算的 Abs API（适用于所有架构）
   - 使用基础 DataCopy API（连续搬运模式）

3. **参数含义**：
   - count：参与计算/搬运的元素个数
   - 需确保 count 对齐到 32B（实际处理时按 ubFormer 切分）

4. **优化技巧**：
   - 按 Elementwise Tiling 指南进行多核切分和 UB 切分
   - 区分首 block 和尾 block 的处理逻辑

### 1.5 内存管理 (Buffer 规划)

| Buffer 名称 | 用途 | 大小计算 | TPosition |
|------------|------|---------|-----------|
| inQueueX | 输入数据缓存（MTE2 搬运） | ubFormer * sizeof(T) * 2 | VECIN |
| outQueueY | 输出数据缓存（MTE3 搬运） | ubFormer * sizeof(T) * 2 | VECOUT |

**总 UB 使用量**（Double Buffer 开启，每个 TQue 占用 2 个 buffer）：
- FP16：`2 * 2 * ubFormer * 2 bytes` = `8 * ubFormer bytes`
- FP32：`2 * 2 * ubFormer * 4 bytes` = `16 * ubFormer bytes`
- 典型值（FP32，ubFormer ≈ 12KB）：约 192KB（满 UB）

**Buffer 规划说明**：
- 采用 Double Buffer 策略：`InitBuffer(que, 2, size)` 开启 Double Buffer
- 输入输出各占用一个 TQue，每个 TQue 在 Double Buffer 下占用 2 个 buffer
- 无需中间计算 Buffer（abs 操作无跨元素依赖）
- Double Buffer 使 MTE2/MTE3 与 Vector 计算并行，掩盖搬运延迟

---

## 2. 架构设计

### 2.1 多核切分策略

| 项目 | 说明 |
|-----|------|
| 切分维度 | 按元素总数 dim0 切分，展平为 1D 线性处理 |
| 单核任务量 | blockFormer（对齐到 512 元素） |
| 使用的核数 | 动态计算：`min(计算核数, 最大核数)`，保证每个核至少处理 4KB 数据 |
| 负载均衡方式 | 均分 + 尾块处理（最后一个 block 处理剩余数据） |

**多核切分公式**（来自 Elementwise Tiling 指南）：

```cpp
// Step 1: 计算核数（保证每个核至少处理 4KB 数据）
// MIN_TILING_BITS_SIZE_PER_CORE = 32768 bits = 4KB
// minDtypeBits = sizeof(T) * 8
coreNum = (dim0 * minDtypeBits + MIN_TILING_BITS_SIZE_PER_CORE - 1) / 
           MIN_TILING_BITS_SIZE_PER_CORE;
coreNum = min(coreNum, availableCoreNum);

// Step 2: 每个核的基础元素数，对齐到 512 元素
// ELEM_ALIGN_FACTOR = 512
blockFormer = ((dim0 + coreNum - 1) / coreNum + ELEM_ALIGN_FACTOR - 1) / 
               ELEM_ALIGN_FACTOR * ELEM_ALIGN_FACTOR;

// Step 3: 计算虚拟 block 数量
blockNum = (dim0 + blockFormer - 1) / blockFormer;
```

**小 shape 场景特殊处理**（当 dim0 < 512 时）：

```cpp
// 小 shape 场景处理（dim0 < ELEM_ALIGN_FACTOR）
if (dim0 < ELEM_ALIGN_FACTOR) {
    // 单核处理，不对齐
    coreNum = 1;
    blockFormer = dim0;
    blockNum = 1;
} else {
    // 正常多核切分（上述公式）
    // ...
}
```

**说明**：小 shape 场景（如 dim0=64）下，直接单核处理全部数据，避免对齐计算导致负数问题。

### 2.2 UB 切分策略

| 项目 | 说明 |
|-----|------|
| UB 容量 | 192KB (DAV_2201) / 248KB (DAV_3510)，运行时通过 `GetCoreMemSize` 获取 |
| 单次处理数据量 | ubFormer（256B 对齐） |
| 是否需要分 chunk | 是，当 blockFormer > ubFormer 时需要多次 UB 循环 |
| chunk 大小计算公式 | ubFormer = 对齐后的 UB 单次处理量 |

**UB 切分公式**：

```cpp
// Step 1: 计算 UB 能容纳的最大元素数
// Double Buffer 开启，每个 TQue 占用 2 个 buffer
// 总 buffer 数 = 2 * 2 = 4（inQueueX + outQueueY，各 2 个）
bufferDivisor = 4 * sizeof(T);  // 输入输出各 2 个 buffer
maxElemNum = ubSize / bufferDivisor;

// Step 2: 按 256B 对齐（REPEAT_BYTES = 256）
alignFactor = REPEAT_BYTES / sizeof(T);  // FP32=64元素, FP16=128元素
ubFormer = (maxElemNum / alignFactor) * alignFactor;

// Step 3: 计算循环次数和尾部大小
// 区分首 block 和尾 block
ubLoopOfFormerBlock = (blockFormer + ubFormer - 1) / ubFormer;
ubTailOfFormerBlock = blockFormer - (ubLoopOfFormerBlock - 1) * ubFormer;

blockTail = dim0 - (blockNum - 1) * blockFormer;
ubLoopOfTailBlock = (blockTail + ubFormer - 1) / ubFormer;
ubTailOfTailBlock = blockTail - (ubLoopOfTailBlock - 1) * ubFormer;
```

**具体数值计算示例**（Atlas A2/A3 UB = 192KB）：

```cpp
// FP32 示例
constexpr uint64_t UB_SIZE = 192 * 1024;  // 192KB
uint32_t maxElemNumFP32 = UB_SIZE / (4 * sizeof(float));  // = 48KB / 4 bytes = 12288 元素
uint32_t alignFactorFP32 = 256 / sizeof(float);           // = 64 元素
uint32_t ubFormerFP32 = (12288 / 64) * 64;                // = 12288 元素

// FP16 示例
uint32_t maxElemNumFP16 = UB_SIZE / (4 * sizeof(half));   // = 48KB / 2 bytes = 24576 元素
uint32_t alignFactorFP16 = 256 / sizeof(half);            // = 128 元素
uint32_t ubFormerFP16 = (24576 / 128) * 128;              // = 24576 元素

// Double Buffer 说明
// InitBuffer(inQueueX, 2, ubFormerFP32 * sizeof(float));  // 每个 buffer 48KB
// InitBuffer(outQueueY, 2, ubFormerFP32 * sizeof(float)); // 每个 buffer 48KB
// 总占用：4 * 48KB = 192KB（满 UB）
```

### 2.3 分支场景覆盖

| 分支条件 | 处理策略 |
|---------|---------|
| 数据类型 FP16 | 使用 half 类型模板参数，按 FP16 精度标准验证 |
| 数据类型 FP32 | 使用 float 类型模板参数，按 FP32 精度标准验证 |
| 大 shape | 正常多核切分，充分利用核数 |
| 小 shape | 可能只用少量核，甚至单核处理 |
| 对齐情况 | 正常处理，使用完整 count |
| 边界情况 | 尾块处理，使用 tail count |

### 2.4 类别特有设计

#### 2.4.1 Elementwise 标准流程

**适用场景**：所有 Elementwise 算子（输入输出 Shape 相同）

**标准三段式结构**（来自 [api-buffer.md §Double Buffer 流水线并行](.opencode/skills/ascendc-api-best-practices/references/api-buffer.md#double-buffer-流水线并行)）：

```
CopyIn → Compute → CopyOut
   ↓         ↓         ↓
  MTE2     Vector     MTE3
```

**Double Buffer 流水线并行时间线**：
```
Row 0: [MTE2-B0][Vector-B0][MTE3-B0]
Row 1:          [MTE2-B1][Vector-B1][MTE3-B1]
          ↑ MTE2 与 Vector 并行！
```

**核心流程伪代码**：

```cpp
// ========== CopyIn (GM → UB) ==========
template<typename T>
__aicore__ inline void CopyIn(int64_t offset, int64_t count) {
    // 1. 从 inQueueX 分配 buffer（Double Buffer 自动轮转 Buffer 0/1）
    LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
    
    // 2. 使用 DataCopyPad 搬运数据（支持非对齐）
    AscendC::DataCopyPad(xLocal, xGm[offset], 
        {1, (uint32_t)(count * sizeof(T)), 0, 0},  // copyParams: blockLen 单位为字节
        {false, 0, 0, 0});                         // padParams: 自动处理非对齐
    
    // 3. 入队，标记数据就绪（非阻塞）
    inQueueX.EnQue<T>(xLocal);
}

// ========== Compute (UB 计算) ==========
template<typename T>
__aicore__ inline void Compute(int64_t count) {
    // 1. 从 inQueueX 获取输入数据（阻塞，等待 CopyIn 就绪）
    LocalTensor<T> xLocal = inQueueX.DeQue<T>();
    
    // 2. 从 outQueueY 分配输出 buffer（Double Buffer 自动轮转）
    LocalTensor<T> yLocal = outQueueY.AllocTensor<T>();
    
    // 3. 计算绝对值
    AscendC::Abs(yLocal, xLocal, count);
    
    // 4. 入队输出数据，标记计算完成
    outQueueY.EnQue<T>(yLocal);
    
    // 5. 释放输入 buffer
    inQueueX.FreeTensor(xLocal);
}

// ========== CopyOut (UB → GM) ==========
template<typename T>
__aicore__ inline void CopyOut(int64_t offset, int64_t count) {
    // 1. 从 outQueueY 获取输出数据（阻塞，等待 Compute 就绪）
    LocalTensor<T> yLocal = outQueueY.DeQue<T>();
    
    // 2. 使用 DataCopyPad 搬运数据到 GM（支持非对齐）
    AscendC::DataCopyPad(yGm[offset], yLocal, 
        {1, (uint32_t)(count * sizeof(T)), 0, 0});  // copyParams: 自动处理非对齐
    
    // 3. 释放输出 buffer
    outQueueY.FreeTensor(yLocal);
}

// ========== Process (主流程) ==========
template<typename T>
__aicore__ inline void Process() {
    // 1. 判断当前处理的是否是最后一个 block
    bool isLastBlock = (blockIdx == blockNum - 1);
    
    // 2. 获取当前 block 的循环次数和尾部大小
    int64_t loopNum = isLastBlock ? ubLoopOfTailBlock : ubLoopOfFormerBlock;
    int64_t tailNum = isLastBlock ? ubTailOfTailBlock : ubTailOfFormerBlock;
    
    // 3. 计算 GM 偏移
    int64_t offset = blockIdx * blockFormer;
    
    // 4. 主循环（处理完整的 UB 块）
    // 单循环结构，TQue 自动轮转 Buffer 0 和 Buffer 1
    // MTE2 (CopyIn) 与 Vector (Compute) 并行执行
    for (int64_t i = 0; i < loopNum - 1; i++) {
        CopyIn(offset, ubFormer);   // MTE2 异步搬运到 Buffer 0/1
        Compute(ubFormer);          // Vector 计算，同时 MTE2 搬运下一块
        CopyOut(offset, ubFormer);  // MTE3 异步搬出，同时 Vector 计算下一块
        offset += ubFormer;
    }
    
    // 5. 尾部处理（处理最后一个不完整的 UB 块）
    if (tailNum > 0) {
        CopyIn(offset, tailNum);
        Compute(tailNum);
        CopyOut(offset, tailNum);
    }
}
```

**Buffer 需求**：

| Buffer 名称 | 用途 | 大小计算 | Double Buffer |
|------------|------|---------|--------------|
| inQueueX | 输入数据缓存（MTE2 搬运） | ubFormer * sizeof(T) * 2 | num=2 |
| outQueueY | 输出数据缓存（MTE3 搬运） | ubFormer * sizeof(T) * 2 | num=2 |

**Double Buffer 实现说明**：
- `InitBuffer(inQueueX, 2, ubFormer * sizeof(T))`：开启 Double Buffer，占用 2 个 buffer
- `InitBuffer(outQueueY, 2, ubFormer * sizeof(T))`：开启 Double Buffer，占用 2 个 buffer
- 单循环结构 + TQue 自动轮转 Buffer 0 和 Buffer 1
- MTE2/MTE3（搬运）与 Vector（计算）并行执行，掩盖搬运延迟

---

## 3. 确认清单

- [x] 多核切分策略已确定（按 dim0 切分，对齐到 512 元素，含小 shape 场景处理）
- [x] UB 切分策略已确定（按 256B 对齐，区分首/尾 block，含具体数值示例）
- [x] Buffer 规划已完成（2 个 TQue，Double Buffer 开启，共 4 个 buffer）
- [x] 分支场景已覆盖（dtype 分支、shape 分支、对齐分支）
- [x] 类别特有设计已完成（Elementwise 标准三段式流程 + Double Buffer 流水线并行）
- [x] API 已验证（Abs、DataCopyPad 均已查阅官方文档并确认约束）