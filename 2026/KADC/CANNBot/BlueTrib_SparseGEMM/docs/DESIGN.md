# sparse_gemm 算子设计文档

## 0. 概述

### 0.0 需求类型判断

**判断标准**：
- **特定用例**：用户明确指定了具体的 shape 和 dtype（如："开发 sparse_gemm，shape=[batch=4, H_Q=64, H_K=4, N=4096, M=4096, D_H=128], dtype=float16"）

### 0.1 基本信息

| 项目 | 内容 |
|-----|------|
| 算子名称 | sparse_gemm |
| 算子类别 | MatMul（稀疏矩阵乘法） |
| 需求类型 | 特定用例（shape=[batch=2, H_Q=64, H_K=4, N=128, M=128, D_H=128], dtype=int8_t） |
| 支持数据类型 | int8_t (输入), int32_t (CO1中间结果), float32 (输出) |
| 支持芯片 | Atlas A2/A3 训练系列 |
| 特殊约束 | B矩阵采用2:4稀疏格式，每4个元素中最多2个非零元素 |
| API路线 | 基础API（LoadDataWithSparse + MmadWithSparse） |

---

## 1. 算子设计

### 1.1 数学公式

```
// 输入输出定义
输入: 
  A - 稠密矩阵，shape=[batch, H_Q, N, D_H], dtype=int8_t
  B - 稀疏矩阵（2:4格式），shape=[batch, H_K, M, D_H/2], dtype=int8_t
  Index - 索引矩阵，shape=[batch, H_K, M, D_H/8], dtype=uint8_t
输出: 
  C - 结果矩阵，shape=[batch, H_Q, N, M], dtype=float32

// 数学公式
C = A * B_sparse^T

其中：
- A 是稠密矩阵（M*K）
- B_sparse 是经过2:4稀疏压缩的矩阵（K/2*N）
- Index 记录稀疏矩阵B中非零元素的位置（每4个元素生成2个2位索引）
- 稀疏格式：每4个元素中最多2个非零元素
```

### 1.2 API 映射

**技术路线选择**：基础API路线（参考 `asc-devkit/examples/01_simd_cpp_api/02_features/03_basic_api/01_matrix_compute/mmad_with_sparse/` 示例）

| 数学操作 | 对应 API | 关键参数 | 数据布局 | 官方文档 |
|---------|---------|---------|---------|---------|
| 稀疏矩阵加载 | `LoadDataWithSparse` | dst(B2), src(B1), idx(B1), loadDataParams | NZ格式 | [链接](../../../asc-devkit/docs/api/SIMD-API/基础API/矩阵计算（ISASI）/数据搬运/LoadDataWithSparse.md) |
| 稀疏矩阵乘法 | `MmadWithSparse` | dst(CO1), fm(A2), filter(B2), mmadParams | ND格式 | [链接](../../../asc-devkit/docs/api/SIMD-API/基础API/矩阵计算（ISASI）/矩阵计算/MmadWithSparse.md) |
| 结果类型转换 | `Fixpipe` | dst(GM), src(CO1), fixpipeParams | ND格式 | [链接](../../../asc-devkit/docs/api/SIMD-API/基础API/数据搬运/Fixpipe.md) |

#### 1.2.1 API 语义验证

**每个 API 调用前，必须填写验证表**。

**验证表格**（所有 API 必须填写）：

| API | 数据布局 | 功能需求 | API选择 | 限制条件 | 匹配 | 文档 |
|-----|---------|---------|---------|---------|-----|------|
| LoadDataWithSparse | B1→B2，NZ格式，512B对齐 | 搬运稀疏权重矩阵到B2，同时加载索引矩阵 | `LoadDataWithSparse(b2Local, b1Local, idxB1Local, loadDataParams)` | 输入输出必须是int8_t；索引矩阵为uint8_t；不支持转置 | ✅ | [链接](../../../asc-devkit/docs/api/SIMD-API/基础API/矩阵计算（ISASI）/数据搬运/LoadDataWithSparse.md) |
| MmadWithSparse | A2+B2→CO1，ND格式 | 稀疏矩阵乘加操作 | `MmadWithSparse(cLocal, a2Local, b2Local, mmadParams)` | 左矩阵A为稠密矩阵，右矩阵B为稠密化后的矩阵；输出必须是int32_t；M/K/N不能为0 | ✅ | [链接](../../../asc-devkit/docs/api/SIMD-API/基础API/矩阵计算（ISASI）/矩阵计算/MmadWithSparse.md) |
| Fixpipe | CO1→GM，ND格式 | 结果输出+类型转换 | `Fixpipe(cGM, cLocal, fixpipeParams)` | 支持int32_t→float32转换 | ✅ | [链接](../../../asc-devkit/docs/api/SIMD-API/基础API/数据搬运/Fixpipe.md) |

**验证清单**（每个 API 必须完成）：
- [x] 1. 数据布局确认（内存排列、连续性、对齐）
- [x] 2. 功能需求明确（操作类型、维度、输出格式）
- [x] 3. 已查阅官方文档（提供链接）
- [x] 4. 匹配验证（数据布局与 API 能力匹配、限制条件满足）
- [x] 5. 已记录验证过程

### 1.3 数据流

```
输入 A (Global Tensor, 稠密矩阵, int8_t)
    ↓ DataCopy
输入 A (Local Tensor, A1位置, int8_t)
    ↓ LoadData (NZ转ZN)
输入 A (Local Tensor, A2位置, int8_t)
    ↓
输入 B (Global Tensor, 稀疏矩阵, int8_t, K/2*N)
    ↓ DataCopy
输入 B (Local Tensor, B1位置, int8_t)
    ↓ LoadDataWithSparse (加载稀疏矩阵+索引)
输入 B (Local Tensor, B2位置, int8_t, 稠密化后)
    ↓
索引矩阵 (Global Tensor, uint8_t, K/8*N)
    ↓ DataCopy
索引矩阵 (Local Tensor, B1位置, uint8_t)
    ↓
MmadWithSparse 计算 (A2 * B2 → CO1)
    ↓
输出 C (Local Tensor, CO1位置, int32_t)
    ↓ Fixpipe (int32_t → float32 转换)
输出 C (Global Tensor, float32)
```

### 1.4 核心计算步骤(复杂算子)

**重要说明**:
- **本节定位**: 核心计算步骤概览 + 关键设计要点(简化的)
- **核心伪代码**: 各分支Compute过程核心伪代码
- **避免重复**: 不要在本节写完整的 for 循环伪代码

**核心计算步骤**:
```
1. 数据准备 - 将稠密矩阵A、稀疏矩阵B、索引矩阵从GM搬运到L1/L0
2. 稀疏加载 - 使用LoadDataWithSparse加载稀疏矩阵B并生成稠密化矩阵到B2
3. 矩阵乘法 - 使用MmadWithSparse执行稀疏矩阵乘法（A2 * B2 → CO1）
4. 结果输出 - 使用Fixpipe将结果从CO1搬运到GM（同时完成int32_t→float32转换）
```

**分支差异对比**(如果有多个分支):

| 操作 | 主分支 | 尾块分支 |
|------|--------|----------|
| 数据加载 | 完整tile加载 | 处理边界tile |
| 矩阵乘法 | 标准MmadWithSparse | 使用SetTail设置尾块维度 |
| 结果输出 | 标准Fixpipe | 处理边界输出 |

**关键设计要点**:
1. **Buffer 使用**: 
   - A1/A2: 存储稠密矩阵A的中间表示
   - B1: 存储稀疏矩阵B和索引矩阵
   - B2: 存储稠密化后的B矩阵
   - CO1: 存储计算结果（int32_t）
2. **API 选择**: 
   - 使用LoadDataWithSparse处理稀疏格式（B1→B2）
   - 使用MmadWithSparse执行稀疏矩阵乘法（A2*B2→CO1）
   - 使用Fixpipe输出结果并转换类型（CO1→GM）
3. **参数含义**: 
   - 稀疏矩阵B的大小为K/2*N（因为2:4稀疏格式压缩了一半）
   - 索引矩阵大小为K/8*N（每4个元素生成2个2位索引）
   - 稠密矩阵A的大小为M*K
4. **优化技巧**: 
   - 使用流水线并行数据搬运和计算
   - 使用多核并行处理不同tile

**参数使用规则**:
| 参数位置 | 用有效长度 | 用对齐长度 |
|---------|-----------|-----------|
| DataCopy blockLen / 计算 API count | ✓ | ✗ |
| UB 数据偏移 / Buffer 大小 | ✗ | ✓ |

### 1.5 内存管理(Buffer 规划)

| Buffer 名称 | 用途 | 大小计算 | TPosition |
|------------|------|---------|-----------|
| L1_A | A矩阵滚动窗口 | depthA1 × baseM × baseK × sizeof(int8_t) | L1 |
| L1_B | B矩阵滚动窗口 | depthB1 × baseN × baseK/2 × sizeof(int8_t) | L1 |
| L0A | A分块计算 | baseM × baseK × sizeof(int8_t) | L0A |
| L0B | B分块计算 | baseN × baseK/2 × sizeof(int8_t) | L0B |
| L0C | 累加结果 | dbL0C × baseM × baseN × sizeof(int32_t) | L0C |
| L1_Index | 索引矩阵 | baseN × baseK/8 × sizeof(uint8_t) | L1 |

**总 UB 使用量**: 根据Tiling引擎自动计算，需满足 ≤ 192KB (A2/A3)

---

## 2. 架构设计

### 2.1 多核切分策略

| 项目 | 说明 |
|-----|------|
| 切分维度 | M × N 二维切分（K 轴在核内迭代，不参与核间切分） |
| 调度方式 | 蛇形调度（BlockScheduler），参考工程 `matmul_kernel_fused.h` |
| 单核任务量 | `singleCoreM × singleCoreN`（由 SWAT Tiling 引擎自动计算） |
| 核数计算 | `usedCoreNum = CeilDiv(M, singleCoreM) × CeilDiv(N, singleCoreN)` **强制动态计算**，禁止硬编码 |
| 负载均衡 | BlockScheduler 蛇形分配保证同 row/col 的核工作量一致 |

### 2.2 UB 切分策略

| 项目 | 说明 |
|-----|------|
| UB 容量 | 192KB (A2/A3) |
| 单次处理数据量 | 由Tiling引擎根据L1/UB容量优化得出 |
| 是否需要分 chunk | 根据M/N大小决定，大shape需要分tile |
| chunk 大小计算公式 | baseM/baseN/baseK 由Tiling引擎自动计算 |
| **稀疏格式约束** | baseK 必须是4的倍数（2:4稀疏格式要求） |

### 2.3 分支场景覆盖

| 分支条件 | 处理策略 |
|---------|---------|
| 数据类型 | int8_t输入，int32_t(CO1)，float32输出 |
| 大 shape | 多核M×N切分 + 核内K轴L1滚动 |
| 小 shape | 减少核数；baseM/baseN可适当缩小以匹配 |
| 对齐 | 32字节对齐要求 |
| 非对齐 | 使用DataCopyPad处理非对齐数据 |
| 边界情况 | 使用SetTail处理尾块 |
| **稀疏格式** | K维度必须4对齐，尾块需padding |

### 2.4 类别特有设计

**重要说明**：稀疏矩阵乘法使用Cube单元加速，支持2:4稀疏格式。采用基础API路线（LoadDataWithSparse + MmadWithSparse），参考官方示例 `mmad_with_sparse`。

#### 2.4.1 主分支（标准tile）

**适用场景**：M和N都是tile大小的整数倍，K是4的倍数

**Compute核心流程伪代码**：

```cpp
// 主分支的核心计算流程（基础API路线）
// 参考：asc-devkit/examples/01_simd_cpp_api/02_features/03_basic_api/01_matrix_compute/mmad_with_sparse/

// 1. CopyIn - 数据搬运到L1
DataCopy(a1Local, aGM, {1, aSize * sizeof(int8_t) / 32, 0, 0});
DataCopy(b1Local, bGM, {1, bSize * sizeof(int8_t) / 32, 0, 0});
DataCopy(idxB1Local, idxGM, {1, bSize / 4 * sizeof(int8_t) / 32, 0, 0});

// 2. SplitA - A矩阵NZ转ZN格式
LoadData2DParams loadDataParamsA;
loadDataParamsA.repeatTimes = kBlocks;
loadDataParamsA.srcStride = mBlocks;
loadDataParamsA.dstGap = 0;
loadDataParamsA.ifTranspose = false;
for (int i = 0; i < mBlocks; ++i) {
    LoadData(a2Local[i * dstOffset], a1Local[i * srcOffset], loadDataParamsA);
}

// 3. SplitB - 使用LoadDataWithSparse加载稀疏矩阵B
LoadData2DParams loadDataParamsB;
loadDataParamsB.repeatTimes = kBlocks * nBlocks / 2;
loadDataParamsB.srcStride = 0;
loadDataParamsB.ifTranspose = false;
LoadDataWithSparse(b2Local, b1Local, idxB1Local, loadDataParamsB);

// 4. Compute - 稀疏矩阵乘法
MmadWithSparse(cLocal, a2Local, b2Local, {m, n, k, false, 0, false, false, false});

// 5. CopyOut - 结果输出（int32_t → float32）
FixpipeParamsV220 fixpipeParams;
fixpipeParams.nSize = n;
fixpipeParams.mSize = m;
fixpipeParams.srcStride = m;
fixpipeParams.dstStride = n;
fixpipeParams.ndNum = 1;
fixpipeParams.srcNdStride = 0;
fixpipeParams.dstNdStride = 0;
Fixpipe(cGM, cLocal, fixpipeParams);
```

**Buffer 需求**：

| Buffer 名称 | 用途 | 大小计算 |
|------------|------|---------|
| L1_A | A矩阵滚动窗口 | depthA1 × baseM × baseK × sizeof(int8_t) |
| L1_B | B矩阵滚动窗口 | depthB1 × baseN × baseK/2 × sizeof(int8_t) |
| L0A | A分块计算 | baseM × baseK × sizeof(int8_t) |
| L0B | B分块计算 | baseN × baseK/2 × sizeof(int8_t) |
| L0C | 累加结果 | dbL0C × baseM × baseN × sizeof(int32_t) |
| L1_Index | 索引矩阵 | baseN × baseK/8 × sizeof(uint8_t) |

#### 2.4.2 尾块分支（边界处理）

**适用场景**：M或N不是tile大小的整数倍

**Compute核心流程伪代码**：

```cpp
// 尾块分支的核心计算流程
// 与主分支类似，但需要处理边界

// 1. 计算尾块大小
int tailM = M - mCoreIndex * singleCoreM;
tailM = tailM < singleCoreM ? tailM : singleCoreM;
int tailN = N - nCoreIndex * singleCoreN;
tailN = tailN < singleCoreN ? tailN : singleCoreN;

// 2. CopyIn - 数据搬运（边界处理）
// 使用DataCopyPad处理非对齐数据

// 3. SplitA/SplitB - 与主分支相同

// 4. Compute - 使用SetTail设置尾块维度
MmadWithSparse(cLocal, a2Local, b2Local, {tailM, tailN, k, false, 0, false, false, false});

// 5. CopyOut - 结果输出（处理边界）
```

**Buffer 需求**：与主分支相同，但实际使用量根据尾块大小调整。

**稀疏格式约束**：
- K维度必须是4的倍数（2:4稀疏格式要求）
- 如果K不是4的倍数，需要在Tiling阶段padding到4的倍数
- 索引矩阵大小为 K/8 * N（每4个元素生成2个2位索引）

---

## 3. 确认清单
- [x] 多核切分策略已确定
- [x] UB 切分策略已确定
- [x] Buffer 规划已完成
- [x] 分支场景已覆盖
- [x] 类别特有设计已完成
- [x] 所有API已通过文档验证
- [x] 稀疏格式约束已确认（2:4稀疏）
- [x] 数据类型映射已确认（int8_t输入，int32_t CO1，float32输出）
- [x] API路线确认（基础API：LoadDataWithSparse + MmadWithSparse）
- [x] 稀疏矩阵定义明确（B是稀疏矩阵K/2*N，A是稠密矩阵M*K）
