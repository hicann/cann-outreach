# abs 算子设计串讲质疑清单

---

## 设计串讲

### 审查结论
- [ ] 设计可直接开发（无阻塞问题）
- [x] 设计需要修改后开发（有阻塞/讨论问题）
- [ ] 设计存在严重问题，无法开发

### 质疑清单

#### 问题 1：DataCopy API 选择不符合最佳实践
- **类别**：API 可行性
- **严重程度**：🔴 阻塞
- **设计文档位置**：DESIGN.md §1.2（API 映射）和 §1.2.1（API 语义验证）
- **问题描述**：设计文档中选择了 `DataCopy` API 进行 GM↔UB 数据搬运，并提到"count * sizeof(T) 需要 32 字节对齐，若不对齐，搬运量将向下取整"。根据最佳实践文档（ascendc-api-best-practices/references/api-datacopy.md），**应该优先使用 DataCopyPad**，因为：
  - 非对齐或不确定对齐场景必须使用 DataCopyPad
  - 尾块处理时 `tailNum` 很可能不对齐（如 shape=[1, 513]）
  - DataCopy 非对齐会导致数据错误，而不是简单的"向下取整"
- **Developer 视角**：从开发者角度看，使用 DataCopy 会导致：
  1. 需要手动判断每次搬运是否 32 字节对齐
  2. 尾块处理复杂化，增加边界 bug 风险
  3. 违反"优先使用 DataCopyPad"的最佳实践原则
- **建议方案**：将所有 `DataCopy` API 替换为 `DataCopyPad` API，参数配置为：
  ```cpp
  // CopyIn (GM → UB)
  AscendC::DataCopyPad(xLocal, xGm[offset], 
      {1, (uint32_t)(count * sizeof(T)), 0, 0},  // copyParams
      {false, 0, 0, 0});                         // padParams: 自动处理
  
  // CopyOut (UB → GM)
  AscendC::DataCopyPad(yGm[offset], yLocal, 
      {1, (uint32_t)(count * sizeof(T), 0, 0});  // copyParams
  ```
- **文档依据**：
  - 最佳实践：[api-datacopy.md §选择规则](ascendc-api-best-practices/references/api-datacopy.md#选择规则)："原则：优先使用 DataCopyPad"
  - 官方文档：[基础数据搬运.md](asc-devkit/docs/api/SIMD-API/基础API/Memory数据搬运/DataCopy/基础数据搬运.md)："count * sizeof(T) 需要 32 字节对齐，若不对齐，搬运量将对 32 字节做向下取整"

---

#### 问题 2：伪代码 ProcessTile 函数中 Buffer 流水线逻辑不完整
- **类别**：伪代码可实现性
- **严重程度**：🔴 阻塞
- **设计文档位置**：DESIGN.md §2.4.1（Elementwise 标准流程）ProcessTile 函数
- **问题描述**：ProcessTile 函数中的 Buffer 流水线逻辑不完整：
  1. **缺少 CopyIn 步骤**：ProcessTile 函数直接 DeQue 输入 buffer（第 230 行），但对应的 EnQue 操作在 Compute 函数中（第 269 行）
  2. **缺少 CopyOut 步骤**：ProcessTile 函数完成计算后，只执行了 EnQue（第 240 行），但缺少独立的 CopyOut 函数来处理 DataCopy(yGm, yLocal, count)
  3. **Buffer 流水线顺序混乱**：
     - Compute 函数第 267-269 行执行了 CopyIn（AllocTensor → DataCopy → EnQue）
     - ProcessTile 函数第 230 行直接 DeQue 输入
     - ProcessTile 函数第 246 行执行 DataCopy(yGm, yLocal, count)
     - Compute 函数没有调用 ProcessTile 前的 CopyIn 逻辑，直接在循环内处理
- **Developer 视角**：从开发者角度看，这个伪代码会导致：
  1. 难以按照标准的 CopyIn-Compute-CopyOut 三段式结构实现
  2. Buffer 流水线管理混乱，容易导致 EnQue/DeQue 配对错误
  3. 无法清晰地实现 Double Buffer 流水线并行
- **建议方案**：将伪代码重构为标准的 Elementwise 流程：
  ```cpp
  // 标准三段式结构
  template<typename T>
  __aicore__ inline void CopyIn(int64_t offset, int64_t count) {
      LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
      AscendC::DataCopyPad(xLocal, xGm[offset], 
          {1, (uint32_t)(count * sizeof(T)), 0, 0}, {false, 0, 0, 0});
      inQueueX.EnQue<T>(xLocal);
  }
  
  template<typename T>
  __aicore__ inline void Compute(int64_t count) {
      LocalTensor<T> xLocal = inQueueX.DeQue<T>();
      LocalTensor<T> yLocal = outQueueY.AllocTensor<T>();
      AscendC::Abs(yLocal, xLocal, count);
      outQueueY.EnQue<T>(yLocal);
      inQueueX.FreeTensor(xLocal);
  }
  
  template<typename T>
  __aicore__ inline void CopyOut(int64_t offset, int64_t count) {
      LocalTensor<T> yLocal = outQueueY.DeQue<T>();
      AscendC::DataCopyPad(yGm[offset], yLocal, 
          {1, (uint32_t)(count * sizeof(T)), 0, 0});
      outQueueY.FreeTensor(yLocal);
  }
  
  template<typename T>
  __aicore__ inline void Process() {
      // 标准循环结构，实现 Double Buffer 流水线并行
      for (int64_t i = 0; i < loopNum - 1; i++) {
          CopyIn(offset, ubFormer);
          Compute(ubFormer);
          CopyOut(offset, ubFormer);
          offset += ubFormer;
      }
      // 尾块处理
      if (tailNum > 0) {
          CopyIn(offset, tailNum);
          Compute(tailNum);
          CopyOut(offset, tailNum);
      }
  }
  ```
- **文档依据**：
  - 最佳实践：[api-buffer.md §Double Buffer 流水线并行](ascendc-api-best-practices/references/api-buffer.md#double-buffer-流水线并行)：标准的 CopyIn-Compute-CopyOut 三段式结构
  - 最佳实践：[api-buffer.md §TQue 正确用法](ascendc-api-best-practices/references/api-buffer.md#tque-正确用法)：EnQue/DeQue/FreeTensor 的正确配对

---

#### 问题 3：UB 切分对齐计算的具体值未明确
- **类别**：内存规划
- **严重程度**：🟡 需讨论
- **设计文档位置**：DESIGN.md §2.2（UB 切分策略）
- **问题描述**：设计文档中给出了 UB 切分的计算公式，但没有给出具体的 `ubFormer` 值计算示例：
  - FP32: alignFactor = 64 元素（256B / 4 bytes）
  - FP16: alignFactor = 128 元素（256B / 2 bytes）
  - UB 容量：192KB（Atlas A2/A3）
  - 但没有给出实际的 `ubFormer` 数值计算
- **Developer 视角**：从开发者角度看，需要明确：
  1. FP32 下 `ubFormer` 的典型值是多少元素？（如 24KB / 4 bytes = 6144 元素）
  2. FP16 下 `ubFormer` 的典型值是多少元素？（如 48KB / 2 bytes = 24576 元素）
  3. Double Buffer 开启后，每个 buffer 实际大小是多少？
- **建议方案**：补充 UB 切分的具体数值计算示例：
  ```cpp
  // Atlas A2/A3 UB = 192KB
  constexpr uint64_t UB_SIZE = 192 * 1024;
  
  // FP32 示例
  // bufferNum = 2（Double Buffer），所以每个 buffer 分配 192KB / 4 = 48KB
  uint32_t maxElemNumFP32 = UB_SIZE / 4 / sizeof(float);  // = 48KB / 4 bytes = 12288 元素
  uint32_t alignFactorFP32 = 256 / sizeof(float);         // = 64 元素
  uint32_t ubFormerFP32 = (maxElemNumFP32 / alignFactorFP32) * alignFactorFP32;  // = 12288 元素
  
  // FP16 示例
  uint32_t maxElemNumFP16 = UB_SIZE / 4 / sizeof(half);   // = 48KB / 2 bytes = 24576 元素
  uint32_t alignFactorFP16 = 256 / sizeof(half);          // = 128 元素
  uint32_t ubFormerFP16 = (maxElemNumFP16 / alignFactorFP16) * alignFactorFP16;  // = 24576 元素
  ```
- **文档依据**：
  - 设计文档：DESIGN.md §2.2 UB 切分公式
  - 最佳实践：[api-buffer.md §TQue Buffer 数量限制](ascendc-api-best-practices/references/api-buffer.md#tque-buffer-数量限制)：Double Buffer 开启时每个 TQue 占用 2 个 buffer

---

#### 问题 4：伪代码中未体现 Double Buffer 流水线并行的完整逻辑
- **类别**：伪代码可实现性
- **严重程度**：🟡 需讨论
- **设计文档位置**：DESIGN.md §1.5（内存管理）和 §2.4.1（Elementwise 标准流程）
- **问题描述**：设计文档中提到采用 Double Buffer 策略（bufferNum = 2），但伪代码中没有体现 Double Buffer 的流水线并行逻辑：
  1. Compute 函数中的循环结构没有体现 Double Buffer 的轮转机制
  2. 没有说明如何通过 EnQue/DeQue 自动实现 Buffer 0 和 Buffer 1 的轮转
  3. 缺少对 Double Buffer 流水线并行的说明（MTE2/MTE3 与 Vector 计算并行）
- **Developer 觖角**：从开发者角度看，需要明确：
  1. Double Buffer 的流水线并行时间线（如 Row 0: [MTE2-B0][Vector-B0][MTE3-B0], Row 1: [MTE2-B1][Vector-B1][MTE3-B1]）
  2. 单循环结构如何自动实现 Buffer 轮转（通过 TQue 的 EnQue/DeQue）
  3. 如何通过标准的三段式结构（CopyIn-Compute-CopyOut）实现流水线并行
- **建议方案**：在伪代码中明确说明 Double Buffer 的流水线并行逻辑：
  ```cpp
  // Double Buffer 流水线并行时间线：
  // Row 0: [MTE2-B0][Vector-B0][MTE3-B0]
  // Row 1:          [MTE2-B1][Vector-B1][MTE3-B1]
  //          ↑ MTE2 与 Vector 并行！
  
  template<typename T>
  __aicore__ inline void Process() {
      // 单循环结构，TQue 自动轮转 Buffer 0 和 Buffer 1
      for (int64_t i = 0; i < loopNum - 1; i++) {
          CopyIn(offset, ubFormer);   // MTE2 异步搬运到 Buffer 0/1
          Compute(ubFormer);          // Vector 计算，同时 MTE2 搬运下一块
          CopyOut(offset, ubFormer);  // MTE3 异步搬出，同时 Vector 计算下一块
          offset += ubFormer;
      }
      // 尾块处理
      if (tailNum > 0) {
          CopyIn(offset, tailNum);
          Compute(tailNum);
          CopyOut(offset, tailNum);
      }
  }
  ```
- **文档依据**：
  - 最佳实践：[api-buffer.md §Double Buffer 流水线并行](ascendc-api-best-practices/references/api-buffer.md#double-buffer-流水线并行)：详细说明流水线并行原理和实现
  - 最佳实践：[api-buffer.md §为什么能并行？](ascendc-api-best-practices/references/api-buffer.md#为什么能并行)：DataCopy 异步 DMA、EnQue 非阻塞、DeQue 阻塞

---

#### 问题 5：多核切分策略的小 shape 场景处理不明确
- **类别**：多核策略
- **严重程度**：🟢 建议
- **设计文档位置**：DESIGN.md §2.1（多核切分策略）
- **问题描述**：多核切分公式中对齐到 512 元素可能导致小 shape 场景的处理不明确：
  - 当 dim0 < 512 时，`blockFormer` 会被对齐到 512，但实际数据量小于 512
  - 测试用例中有 shape=[1, 64] 的情况，此时 dim0=64 < 512
- **Developer 视角**：从开发者角度看，需要明确：
  1. 小 shape 场景下的核数计算：coreNum = min(1, availableCoreNum) = 1
  2. blockFormer 的处理：对齐到 512，但实际只有 64 个元素
  3. 尾块处理：blockTail = dim0 - blockFormer = 64 - 512 = -448（负数？）
- **建议方案**：补充小 shape 场景的处理说明：
  ```cpp
  // 小 shape 场景处理（dim0 < 512）
  if (dim0 < ELEM_ALIGN_FACTOR) {
      // 单核处理，不对齐
      coreNum = 1;
      blockFormer = dim0;
      blockNum = 1;
  } else {
      // 正常多核切分
      coreNum = (dim0 * minDtypeBits + MIN_TILING_BITS_SIZE_PER_CORE - 1) / 
                MIN_TILING_BITS_SIZE_PER_CORE;
      coreNum = min(coreNum, availableCoreNum);
      blockFormer = ((dim0 + coreNum - 1) / coreNum + ELEM_ALIGN_FACTOR - 1) / 
                   ELEM_ALIGN_FACTOR * ELEM_ALIGN_FACTOR;
      blockNum = (dim0 + blockFormer - 1) / blockFormer;
  }
  ```
- **文档依据**：
  - 设计文档：DESIGN.md §2.1 多核切分公式
  - 测试用例：PLAN.md §3.3 T9/P9：小 shape FP16 [1, 64]

---

### 审查总结

**阻塞级问题（必须修改）**：
1. DataCopy API 选择不符合最佳实践（应使用 DataCopyPad）
2. 伪代码 ProcessTile 函数的 Buffer 流水线逻辑不完整（需重构为标准三段式）

**讨论级问题（建议明确）**：
3. UB 切分对齐计算的具体值未明确
4. Double Buffer 流水线并行逻辑未体现

**建议级问题（可选优化）**：
5. 多核切分策略的小 shape 场景处理不明确

**总体评价**：设计方案的核心思路正确（Elementwise、多核切分、UB 切分），但在 API 选择和伪代码实现细节上存在关键问题，需要修改后方可开发。特别是 DataCopy API 的选择和 Buffer 流水线的实现，直接影响代码的正确性和性能。

---

## Developer 审查签名

- **审查时间**：2026-05-23
- **审查角色**：Developer（串讲模式）
- **下一步**：等待 Architect 回应质疑，或直接进入开发阶段（如果阻塞问题已解决）

---

### Architect 回应

#### 问题 1：DataCopy API 选择不符合最佳实践
- **回应**：✅ 已修改
- **理由**：Developer 质疑完全正确。根据最佳实践文档，"原则：优先使用 DataCopyPad"，原因包括：
  1. 自动处理非对齐，无需手动判断每次搬运是否 32 字节对齐
  2. CopyIn 和 CopyOut 都适用
  3. Tiling 设计时可能产生非对齐的 tile 大小（尾块处理）
  4. 对齐场景下性能差异可忽略
- **文档依据**：
  - 最佳实践：[api-datacopy.md §选择规则](.opencode/skills/ascendc-api-best-practices/references/api-datacopy.md#选择规则)
  - 官方文档：[基础数据搬运.md](asc-devkit/docs/api/SIMD-API/基础API/Memory数据搬运/DataCopy/基础数据搬运.md)：确认 DataCopy 的 32 字节对齐要求
- **DESIGN.md 变更**：
  - §1.2 API 映射表：将 DataCopy 替换为 DataCopyPad
  - §1.2.1 API 语义验证表：更新为 DataCopyPad 的参数签名
  - §1.3 数据流：更新为使用 DataCopyPad
  - §2.4.1 伪代码：重构为标准三段式，使用 DataCopyPad

---

#### 问题 2：伪代码 ProcessTile 函数中 Buffer 流水线逻辑不完整
- **回应**：✅ 已修改
- **理由**：Developer 质疑正确。原伪代码的 Buffer 流水线逻辑混乱：
  1. ProcessTile 缺少独立的 CopyIn/CopyOut 步骤
  2. Buffer 流水线顺序不符合标准的 CopyIn-Compute-CopyOut 三段式结构
  3. 难以按照最佳实践实现 Double Buffer 流水线并行
- **文档依据**：
  - 最佳实践：[api-buffer.md §Double Buffer 流水线并行](.opencode/skills/ascendc-api-best-practices/references/api-buffer.md#double-buffer-流水水线并行)：标准三段式结构
  - 最佳实践：[api-buffer.md §TQue 正确用法](.opencode/skills/ascendc-api-best-practices/references/api-buffer.md#tque-正确用法)：EnQue/DeQue/FreeTensor 配对
- **DESIGN.md 变更**：
  - §2.4.1 Elementwise 标准流程：完全重构伪代码为标准三段式结构（CopyIn、Compute、CopyOut、Process）

---

#### 问题 3：UB 切分对齐计算的具体值未明确
- **回应**：✅ 部分修改
- **理由**：Developer 质疑合理。设计文档给出了公式但缺乏具体数值示例，开发者需要明确：
  1. FP32/FP16 下 ubFormer 的典型值
  2. Double Buffer 开启后每个 buffer 的实际大小
- **文档依据**：
  - 设计文档：DESIGN.md §2.2 UB 切分公式
  - 最佳实践：[api-buffer.md §TQue Buffer 数量限制](.opencode/skills/ascendc-api-best-practices/references/api-buffer.md#tque-buffer-数量限制)
- **DESIGN.md 变更**：
  - §2.2 UB 切分策略：补充具体数值计算示例（Atlas A2/A3 UB = 192KB）

---

#### 问题 4：伪代码中未体现 Double Buffer 流水线并行的完整逻辑
- **回应**：✅ 部分修改
- **理由**：Developer 质疑合理。设计文档提到 Double Buffer（bufferNum = 2），但伪代码中未体现：
  1. Double Buffer 的流水线并行时间线
  2. 单循环结构如何自动实现 Buffer 轮转
  3. 如何通过三段式结构实现 MTE2/MTE3 与 Vector 并行
- **文档依据**：
  - 最佳实践：[api-buffer.md §Double Buffer 流水线并行](.opencode/skills/ascendc-api-best-practices/references/api-buffer.md#double-buffer-流水线并行)：详细说明流水线并行原理
  - 最佳实践：[api-buffer.md §为什么能并行？](.opencode/skills/ascendc-api-best-practices/references/api-buffer.md#为什么能并行)：DataCopy 异步 DMA、EnQue 非阻塞、DeQue 阻塞
- **DESIGN.md 变更**：
  - §2.4.1 Elementwise 标准流程：补充 Double Buffer 流水线并行时间线说明

---

#### 问题 5：多核切分策略的小 shape 场景处理不明确
- **回应**：✅ 部分修改
- **理由**：Developer 质疑合理。多核切分公式中对齐到 512 元素可能导致小 shape（如 dim0=64 < 512）的处理不明确：
  1. 小 shape 场景下的核数计算
  2. blockFormer 的处理逻辑
  3. 防止尾块计算出现负数
- **文档依据**：
  - 设计文档：DESIGN.md §2.1 多核切分公式
  - 测试用例：PLAN.md §3.3 T9/P9：小 shape FP16 [1, 64]
- **DESIGN.md 变更**：
  - §2.1 多核切分策略：补充小 shape 场景的特殊处理逻辑

---

### 回应统计
- ✅ 接受 2 项（问题 1、问题 2）
- ✅ 部分修改 3 项（问题 3、问题 4、问题 5）
- ❌ 保留原设计 0 项

---

## Architect 审查签名

- **回应时间**：2026-05-23
- **回应角色**：Architect（串讲回应模式）
- **下一步**：Developer 可基于更新后的 DESIGN.md 进入开发阶段