# sparse_gemm 设计串讲审查

> 审查时间：2026-05-22
> 审查人：Developer（串讲模式）
> 审查范围：DESIGN.md §1（算子设计）、§2.4（伪代码）

---

## 审查结论

- [ ] 设计可直接开发（无阻塞问题）
- [x] 设计需要修改后开发（有阻塞/讨论问题）
- [ ] 设计存在严重问题，无法开发

---

## 质疑清单

### 问题 1：API 混用冲突 — 基础 API 与高阶 API 不兼容

- **类别**：API 可行性
- **严重程度**：🔴 阻塞
- **设计文档位置**：DESIGN.md §1.2 API 映射、§2.4 伪代码
- **问题描述**：

  设计文档同时使用了两类互斥的 API：
  - **基础 API**：`LoadDataWithSparse`、`MmadWithSparse`（需手动管理 A1/A2/B1/B2/CO1 Buffer）
  - **高阶 API**：`SetSparseIndex`、`SetTensorA`、`SetTensorB`、`IterateAll`（Matmul 对象封装接口）

  这两种 API 体系**不能在同一个算子中混用**。基础 API 是直接操作 Buffer 的底层接口，需要手动调用 LoadData/LoadDataWithSparse/MmadWithSparse；高阶 API 通过 Matmul 对象自动管理数据搬运和计算。

- **Developer 视角**：

  伪代码中：
  ```cpp
  // 使用高阶 API
  matmulObj.SetSparseIndex(indexGlobal);
  matmulObj.SetTensorA(aGlobal, IS_TRANS_A);
  matmulObj.SetTensorB(bGlobal, IS_TRANS_B);
  matmulObj.IterateAll(cGlobal);
  ```
  
  但数据流部分描述的是基础 API 的手动 Buffer 管理流程（A1→A2、B1→B2）。开发者无法同时实现两套 API，必须选择其一。

- **文档依据**：

  - LoadDataWithSparse.md：明确是基础 API，操作 B1→B2 的 Buffer 搬运
  - SetSparseIndex.md：约束条件"仅支持在纯Cube模式（只有矩阵计算）且MDL模板的场景使用"
  - MmadWithSparse.md 示例：使用基础 API 手动管理 Buffer

- **建议方案**：

  **方案 A（推荐）**：使用基础 API 路径
  - 移除 `SetSparseIndex`、`SetTensorA`、`SetTensorB`、`IterateAll` 的使用
  - 改为手动调用 `LoadDataWithSparse` + `MmadWithSparse`
  - 参考 `asc-devkit/examples/01_simd_cpp_api/02_features/03_basic_api/01_matrix_compute/mmad_with_sparse/` 示例

  **方案 B**：使用高阶 API 路径
  - 移除 `LoadDataWithSparse`、`MmadWithSparse` 的直接调用
  - 使用 Matmul 对象的 `SetSparseIndex` + `SetTensorA` + `SetTensorB` + `IterateAll`
  - 需要确认高阶 API 是否支持稀疏矩阵场景

---

### 问题 2：数据类型不匹配 — MmadWithSparse 输出必须是 int32_t

- **类别**：精度风险
- **严重程度**：🟡 需讨论
- **设计文档位置**：DESIGN.md §1.1 数学公式、§1.5 内存管理
- **问题描述**：

  设计文档定义输出为 `float32`，但 `MmadWithSparse` API 的输出必须是 `int32_t`（CO1 位置）。

  API 文档明确要求：
  > T 必须为 int32_t 类型，否则编译失败

- **Developer 视角**：

  设计文档说输出是 float32，但实际计算结果是 int32_t。需要在 CopyOut 阶段通过 Fixpipe 或 Cast 进行类型转换。设计文档没有说明这个转换步骤。

- **文档依据**：

  MmadWithSparse.md：
  > dst: 目的操作数，结果矩阵，类型为LocalTensor，支持的TPosition为CO1。
  > T 必须为 int32_t 类型

- **建议方案**：

  1. 明确 L0C/CO1 存储 int32_t 类型数据
  2. 在 CopyOut 阶段使用 Fixpipe 进行 int32_t → float32 转换（Fixpipe 支持数据类型转换）
  3. 或者在 Kernel 末尾使用 Cast API 进行转换
  4. 更新 Buffer 规划中 L0C 的数据类型为 int32_t

---

### 问题 3：稀疏矩阵对象描述混乱

- **类别**：伪代码可实现性
- **严重程度**：🔴 阻塞
- **设计文档位置**：DESIGN.md §1.1 数学公式、§1.2 API 映射
- **问题描述**：

  设计文档中的稀疏矩阵描述存在矛盾：

  - §1.1 说："A - 稀疏矩阵...（2:4稀疏格式）"
  - §1.2 API 映射表说："LoadDataWithSparse — 搬运稠密权重矩阵到B2，同时加载索引矩阵"
  - §1.4 说："稀疏矩阵B的大小为K/2*N"

  根据 API 文档和示例代码：
  - **B 矩阵是稀疏矩阵**（K/2*N，因为 2:4 稀疏格式压缩了一半）
  - **A 矩阵是稠密矩阵**（M*K）
  - `LoadDataWithSparse` 用于加载 B 矩阵（稀疏）和索引矩阵

- **Developer 视角**：

  设计文档的数学公式说 A 是稀疏的，但 API 映射和数据流描述的是 B 是稀疏的。开发者不知道应该对哪个矩阵应用稀疏格式。

- **文档依据**：

  MmadWithSparse.md：
  > "完成矩阵乘加操作，传入的左矩阵A为稀疏矩阵，右矩阵B为稠密矩阵"
  
  但示例代码：
  ```cpp
  // bSize = k / 2 * n;  // B 矩阵是稀疏的，大小减半
  // aSize = m * k;      // A 矩阵是稠密的
  ```

  **注意**：API 文档说"A是稀疏的"，但示例代码显示"B是稀疏的"。这可能是文档错误，需要以示例代码为准。

- **建议方案**：

  1. 明确哪个矩阵是稀疏的（建议以示例代码为准：B 是稀疏矩阵）
  2. 更新数学公式：`C = A * B_sparse^T`
  3. 更新输入定义：
     - A: shape=[batch, H_Q, N, D_H], dtype=int8（稠密矩阵）
     - B: shape=[batch, H_K, M, D_H/2], dtype=int8（稀疏矩阵，2:4 格式压缩）
     - 索引矩阵: shape=[batch, H_K, M, D_H/8], dtype=uint8
  4. 更新数据流描述

---

### 问题 4：输入数据类型错误 — 应为 int8_t 而非 float16

- **类别**：精度风险
- **严重程度**：🔴 阻塞
- **设计文档位置**：DESIGN.md §0.1 基本信息、§1.1 数学公式
- **问题描述**：

  设计文档定义输入为 `float16`，但 `MmadWithSparse` 和 `LoadDataWithSparse` 要求输入数据类型为 `int8_t`。

- **Developer 视角**：

  设计文档说输入是 float16，但 API 要求 int8_t。如果输入真的是 float16，需要先转换为 int8_t 才能使用这些 API。但这个转换会丢失精度，设计文档没有说明。

- **文档依据**：

  LoadDataWithSparse.md：
  > dst: 支持的数据类型为int8_t
  > src: 支持的数据类型为int8_t
  
  MmadWithSparse.md：
  > U 必须为int8_t类型

  示例代码：
  ```cpp
  AscendC::GlobalTensor<int8_t> aGM, bGM;  // 输入是 int8_t
  ```

- **建议方案**：

  **方案 A**：如果输入确实是 int8_t
  - 更新设计文档，将输入 dtype 改为 int8_t
  - 更新 shape 描述（int8_t 稀疏矩阵大小是 float16 的一半）

  **方案 B**：如果输入必须是 float16
  - 需要在 Kernel 开始时将 float16 转换为 int8_t
  - 这会增加计算开销和精度损失
  - 需要在设计文档中明确说明这个转换步骤

---

### 问题 5：Buffer 数据类型错误

- **类别**：内存规划
- **严重程度**：🟡 需讨论
- **设计文档位置**：DESIGN.md §1.5 内存管理
- **问题描述**：

  设计文档中 L1_B 的大小计算使用了 `sizeof(float16)`，但根据 API 要求，B 矩阵应该是 `int8_t` 类型。

  ```
  L1_B: depthB1 × baseN × baseK/2 × sizeof(float16)  // 错误
  L1_B: depthB1 × baseN × baseK/2 × sizeof(int8_t)   // 正确
  ```

- **Developer 视角**：

  Buffer 大小计算错误会导致内存分配不足或浪费。如果按 float16（2字节）计算，实际只需要 int8_t（1字节），会浪费一半空间。

- **建议方案**：

  更新 Buffer 规划中所有使用 `sizeof(float16)` 的地方，根据实际数据类型使用正确的 sizeof。

---

### 问题 6：SetSparseIndex 使用约束未满足

- **类别**：API 可行性
- **严重程度**：🟡 需讨论
- **设计文档位置**：DESIGN.md §1.2 API 映射、§2.4 伪代码
- **问题描述**：

  `SetSparseIndex` 有严格约束：
  > 本接口仅支持在纯Cube模式（只有矩阵计算）且MDL模板的场景使用

  设计文档没有说明是否满足这些约束条件。

- **Developer 视角**：

  如果不满足纯 Cube 模式和 MDL 模板约束，`SetSparseIndex` 会编译失败或运行时错误。开发者需要知道如何启用这些模式。

- **文档依据**：

  SetSparseIndex.md：
  > 索引矩阵的Format格式要求为NZ格式。
  > 本接口仅支持在纯Cube模式（只有矩阵计算）且MDL模板的场景使用。

- **建议方案**：

  1. 如果使用基础 API 路径（方案 A），不需要 SetSparseIndex，索引通过 LoadDataWithSparse 的 idx 参数传入
  2. 如果使用高阶 API 路径，需要：
     - 在 Kernel 头文件定义 `#define ASCENDC_CUBE_ONLY`
     - 使用 `REGIST_MATMUL_OBJ` 注册 Matmul 对象
     - 确认 MDL 模板配置

---

### 问题 7：尾块处理的稀疏格式约束

- **类别**：多核策略
- **严重程度**：🟡 需讨论
- **设计文档位置**：DESIGN.md §2.4.2 尾块分支
- **问题描述**：

  稀疏格式要求"每4个元素中最多2个非零元素"，但设计文档的尾块处理没有考虑这个约束。

  当尾块 K 维度不是 4 的倍数时：
  - 稀疏格式无法正确应用
  - 索引矩阵的计算会出错

- **Developer 视角**：

  如果 K=128，尾块可能是 K_tail=50（不是 4 的倍数）。这种情况下，稀疏格式的边界处理需要特殊逻辑。

- **建议方案**：

  1. 确保 K 维度的 tile 大小是 4 的倍数（baseK % 4 == 0）
  2. 如果尾块 K 不是 4 的倍数，需要 padding 到 4 的倍数
  3. 在 Tiling 策略中明确这个约束

---

### 问题 8：稀疏矩阵形状描述错误

- **类别**：伪代码可实现性
- **严重程度**：🟡 需讨论
- **设计文档位置**：DESIGN.md §1.1 数学公式
- **问题描述**：

  设计文档说：
  > 稀疏矩阵B的大小为K/2*N（因为2:4稀疏格式）

  但输入定义中 B 的 shape 是 `[batch, H_K, M, D_H]`，没有除以 2。

  如果 B 是稀疏矩阵，应该是 `[batch, H_K, M, D_H/2]`（因为 2:4 稀疏压缩了一半）。

- **Developer 视角**：

  shape 定义和稀疏格式说明不一致，开发者不知道实际输入数据的 shape 是什么。

- **建议方案**：

  更新 B 矩阵的 shape 定义：
  - 如果 B 是稀疏矩阵：`shape=[batch, H_K, M, D_H/2]`
  - 如果 B 是稠密矩阵：`shape=[batch, H_K, M, D_H]`（但需要在 Kernel 中进行稀疏化处理）

---

### 问题 9：Fixpipe 数据类型转换未说明

- **类别**：精度风险
- **严重程度**：🟢 建议
- **设计文档位置**：DESIGN.md §1.3 数据流
- **问题描述**：

  数据流显示：
  ```
  输出 C (Local Tensor, CO1位置) → Fixpipe → 输出 C (Global Tensor)
  ```

  但没有说明 Fixpipe 是否进行 int32_t → float32 的类型转换。

- **Developer 视角**：

  Fixpipe 支持数据类型转换，但需要正确配置参数。设计文档应该明确说明这个转换步骤。

- **建议方案**：

  在数据流中明确说明：
  ```
  CO1 (int32_t) → Fixpipe (int32_t→float32) → GM (float32)
  ```

---

### 问题 10：SetSparse 调用位置未明确

- **类别**：API 可行性
- **严重程度**：🟢 建议
- **设计文档位置**：DESIGN.md §1.2 API 映射
- **问题描述**：

  `SetSparse` API 有约束：
  > 本接口必须在 GetTiling 接口前调用

  设计文档只列出了这个 API，但没有说明在 Host 侧的调用位置。

- **Developer 视角**：

  如果不在 GetTiling 前调用 SetSparse，Tiling 计算不会考虑稀疏场景的特殊需求。

- **建议方案**：

  在 Host 侧 Tiling 计算流程中明确：
  ```cpp
  MatmulApiTiling tiling(ascendcPlatform);
  tiling.SetSparse(true);  // 必须在 GetTiling 前
  tiling.SetAType(...);
  tiling.SetBType(...);
  // ...
  tiling.GetTiling(tilingData);
  ```

---

## 总结

### 阻塞问题（必须解决）

| 序号 | 问题 | 核心矛盾 |
|------|------|---------|
| 1 | API 混用冲突 | 基础 API 与高阶 API 不兼容 |
| 3 | 稀疏矩阵对象混乱 | A/B 哪个是稀疏的？ |
| 4 | 输入数据类型错误 | float16 vs int8_t |

### 需讨论问题（建议解决）

| 序号 | 问题 | 核心矛盾 |
|------|------|---------|
| 2 | 输出数据类型 | int32_t vs float32 |
| 5 | Buffer 数据类型 | sizeof(float16) vs sizeof(int8_t) |
| 6 | SetSparseIndex 约束 | 纯 Cube 模式要求 |
| 7 | 尾块稀疏约束 | K 维度 4 对齐 |
| 8 | 稀疏矩阵 shape | D_H vs D_H/2 |

### 建议问题（可选优化）

| 序号 | 问题 |
|------|------|
| 9 | Fixpipe 类型转换说明 |
| 10 | SetSparse 调用位置 |

---

## Architect 回应

### 问题 1：API 混用冲突 — 基础 API 与高阶 API 不兼容

- **回应**：已修改
- **理由**：Developer 质疑完全正确。设计文档确实混用了基础 API（LoadDataWithSparse、MmadWithSparse）和高阶 API（SetSparseIndex、SetTensorA、SetTensorB、IterateAll），这两套 API 体系不能在同一个算子中混用。基础 API 是直接操作 Buffer 的底层接口，高阶 API 通过 Matmul 对象自动管理数据搬运和计算。
- **文档依据**：
  - `asc-devkit/examples/01_simd_cpp_api/02_features/03_basic_api/01_matrix_compute/mmad_with_sparse/mmad_with_sparse.asc` — 官方示例使用基础 API
  - LoadDataWithSparse.md：明确是基础 API，操作 B1→B2 的 Buffer 搬运
  - SetSparseIndex.md：约束条件"仅支持在纯Cube模式（只有矩阵计算）且MDL模板的场景使用"
- **DESIGN.md 变更**：
  - 选择基础 API 路线（LoadDataWithSparse + MmadWithSparse）
  - 移除所有高阶 API 引用（SetSparseIndex、SetTensorA、SetTensorB、IterateAll）
  - 更新 API 映射表，只保留基础 API
  - 更新伪代码，使用手动 Buffer 管理流程

### 问题 2：数据类型不匹配 — MmadWithSparse 输出必须是 int32_t

- **回应**：已修改
- **理由**：Developer 质疑完全正确。MmadWithSparse API 文档明确要求"T 必须为 int32_t 类型，否则编译失败"。设计文档定义输出为 float32，但 CO1 位置必须是 int32_t。
- **文档依据**：
  - MmadWithSparse.md："T 必须为 int32_t 类型，否则编译失败"
  - mmad_with_sparse.asc 示例：`cLocal` 定义为 `LocalTensor<int32_t>`，`cGM` 定义为 `GlobalTensor<int32_t>`
  - 示例代码 CopyOut 函数：使用 Fixpipe 进行 CO1→GM 搬运
- **DESIGN.md 变更**：
  - 更新数据类型映射：int8_t (输入) → int32_t (CO1) → float32 (输出)
  - 更新数据流描述，明确 CO1 是 int32_t
  - 更新 Buffer 规划中 L0C 的数据类型为 int32_t
  - 更新 CopyOut 伪代码，使用 Fixpipe 进行类型转换

### 问题 3：稀疏矩阵对象描述混乱

- **回应**：已修改
- **理由**：Developer 质疑完全正确。设计文档存在矛盾：§1.1 说 A 是稀疏矩阵，但示例代码显示 B 是稀疏矩阵。根据 API 文档和示例代码分析：
  - MmadWithSparse.md 说"左矩阵A为稀疏矩阵，右矩阵B为稠密矩阵"
  - 但示例代码：`bSize = k / 2 * n;`（B 矩阵大小减半，说明 B 是稀疏的）
  - 实际机制：B 是原始稀疏矩阵，通过 LoadDataWithSparse 预处理为稠密矩阵；A 在计算时根据索引进行稠密化
- **文档依据**：
  - mmad_with_sparse.asc 示例：
    ```cpp
    aSize = m * k;      // A 是稠密矩阵
    bSize = k / 2 * n;  // B 是稀疏矩阵，大小减半
    ```
  - LoadDataWithSparse.md："用于搬运存放在B1里的512B的稠密权重矩阵到B2里，同时读取128B的索引矩阵用于稠密矩阵的稀疏化"
- **DESIGN.md 变更**：
  - 更新数学公式：`C = A * B_sparse^T`
  - 更新输入定义：
    - A: shape=[batch, H_Q, N, D_H], dtype=int8_t（稠密矩阵）
    - B: shape=[batch, H_K, M, D_H/2], dtype=int8_t（稀疏矩阵，2:4 格式压缩）
    - Index: shape=[batch, H_K, M, D_H/8], dtype=uint8_t（索引矩阵）
  - 更新数据流描述，明确 B 是稀疏矩阵

### 问题 4：输入数据类型错误 — 应为 int8_t 而非 float16

- **回应**：已修改
- **理由**：Developer 质疑完全正确。LoadDataWithSparse 和 MmadWithSparse 都要求输入数据类型为 int8_t。
- **文档依据**：
  - LoadDataWithSparse.md："支持的数据类型为int8_t"
  - MmadWithSparse.md："U 必须为int8_t类型"
  - mmad_with_sparse.asc 示例：
    ```cpp
    AscendC::GlobalTensor<int8_t> aGM, bGM;  // 输入是 int8_t
    ```
- **DESIGN.md 变更**：
  - 更新基本信息表：支持数据类型改为 int8_t (输入)
  - 更新数学公式：A 和 B 的 dtype 改为 int8_t
  - 更新所有 Buffer 规划中的 sizeof(float16) 为 sizeof(int8_t)
  - 更新伪代码中的数据类型

### 问题 5：Buffer 数据类型错误

- **回应**：已修改
- **理由**：Developer 质疑完全正确。由于输入数据类型改为 int8_t，Buffer 大小计算中的 sizeof(float16) 需要改为 sizeof(int8_t)。
- **文档依据**：
  - LoadDataWithSparse.md："支持的数据类型为int8_t"
  - MmadWithSparse.md："U 必须为int8_t类型"
- **DESIGN.md 变更**：
  - 更新 Buffer 规划表：
    - L1_A: `depthA1 × baseM × baseK × sizeof(int8_t)`
    - L1_B: `depthB1 × baseN × baseK/2 × sizeof(int8_t)`
    - L0A: `baseM × baseK × sizeof(int8_t)`
    - L0B: `baseN × baseK/2 × sizeof(int8_t)`
    - L0C: `dbL0C × baseM × baseN × sizeof(int32_t)`
    - L1_Index: `baseN × baseK/8 × sizeof(uint8_t)`

### 问题 6：SetSparseIndex 使用约束未满足

- **回应**：已修改（随问题1一并解决）
- **理由**：选择基础 API 路线后，不再使用 SetSparseIndex。基础 API 中的索引通过 LoadDataWithSparse 的 idx 参数传入。
- **文档依据**：
  - mmad_with_sparse.asc 示例：索引通过 `LoadDataWithSparse(b2Local, b1Local, idxB1Local, loadDataParams)` 传入
  - SetSparseIndex.md：约束条件"仅支持在纯Cube模式（只有矩阵计算）且MDL模板的场景使用"
- **DESIGN.md 变更**：
  - 移除 SetSparseIndex API 引用
  - 更新 API 映射表，只保留基础 API
  - 更新伪代码，索引通过 LoadDataWithSparse 的 idx 参数传入

### 问题 7：尾块处理的稀疏格式约束

- **回应**：已修改
- **理由**：Developer 质疑正确。2:4 稀疏格式要求每4个元素中最多2个非零元素，K维度必须是4的倍数。尾块处理需要考虑这个约束。
- **文档依据**：
  - MmadWithSparse.md："原始稀疏矩阵B每4个元素中应保证最多2个非零元素"
  - mmad_with_sparse.asc 示例：`kBlocks = k / C0_SIZE;`（C0_SIZE=32，但稀疏格式要求K是4的倍数）
- **DESIGN.md 变更**：
  - 在 UB 切分策略中添加约束：baseK 必须是4的倍数
  - 在分支场景覆盖中添加：稀疏格式约束（K维度必须4对齐）
  - 在尾块分支中添加说明：如果K不是4的倍数，需要padding到4的倍数

### 问题 8：稀疏矩阵形状描述错误

- **回应**：已修改
- **理由**：Developer 质疑完全正确。设计文档中 B 的 shape 定义与稀疏格式说明不一致。
- **文档依据**：
  - mmad_with_sparse.asc 示例：
    ```cpp
    aSize = m * k;      // A: M*K
    bSize = k / 2 * n;  // B: K/2*N（稀疏压缩）
    ```
- **DESIGN.md 变更**：
  - 更新 B 矩阵的 shape：`[batch, H_K, M, D_H/2]`
  - 更新索引矩阵的 shape：`[batch, H_K, M, D_H/8]`
  - 更新数学公式中的尺寸说明

### 问题 9：Fixpipe 数据类型转换未说明

- **回应**：已修改
- **理由**：Developer 质疑正确。数据流中应该明确说明 Fixpipe 进行 int32_t → float32 的类型转换。
- **文档依据**：
  - mmad_with_sparse.asc 示例 CopyOut 函数：
    ```cpp
    AscendC::FixpipeParamsV220 fixpipeParams;
    fixpipeParams.nSize = n;
    fixpipeParams.mSize = m;
    // ...
    AscendC::Fixpipe(cGM, cLocal, fixpipeParams);
    ```
- **DESIGN.md 变更**：
  - 更新数据流描述，明确 CO1 (int32_t) → Fixpipe (int32_t→float32) → GM (float32)
  - 更新 CopyOut 伪代码，添加 Fixpipe 参数配置

### 问题 10：SetSparse 调用位置未明确

- **回应**：保留原设计
- **理由**：选择基础 API 路线后，SetSparse 不再使用。SetSparse 是高阶 API（MatmulApiTiling）的一部分，用于设置稀疏场景。基础 API 路线不需要在 Host 侧调用 SetSparse。
- **文档依据**：
  - SetSparse.md："设置Matmul的使用场景是否为Sparse Matmul场景" — 这是高阶 API 的接口
  - mmad_with_sparse.asc 示例：没有使用 SetSparse，直接使用基础 API
- **DESIGN.md 变更**：无变更（基础 API 路线不涉及 SetSparse）

---

### 回应统计

- **接受 9 项**：问题 1-9
- **保留原设计 1 项**：问题 10（基础 API 路线不涉及 SetSparse）
- **部分修改 0 项**

---

## 总结

### 设计修改要点

1. **API 路线统一**：选择基础 API 路线（LoadDataWithSparse + MmadWithSparse），移除所有高阶 API 引用
2. **数据类型修正**：int8_t (输入) → int32_t (CO1) → float32 (输出)
3. **稀疏矩阵定义明确**：
   - B 是稀疏矩阵（K/2*N），通过 LoadDataWithSparse 预处理
   - A 是稠密矩阵（M*K），在计算时根据索引进行稠密化
4. **Buffer 规划修正**：所有 sizeof(float16) 改为 sizeof(int8_t)
5. **稀疏格式约束**：K 维度必须是 4 的倍数，尾块需要 padding

### 参考资源

- 基础 API 示例：`asc-devkit/examples/01_simd_cpp_api/02_features/03_basic_api/01_matrix_compute/mmad_with_sparse/`
- API 文档：LoadDataWithSparse.md、MmadWithSparse.md、Fixpipe.md
