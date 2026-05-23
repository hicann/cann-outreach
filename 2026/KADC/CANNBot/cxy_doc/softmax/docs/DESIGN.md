# Softmax 算子设计与实施文档

---

## 0. 概述

### 0.0 需求类型判断

**判断结果**：通用

用户未明确指定具体 shape 和 dtype，要求支持 float16/float32、任意 shape 输入、沿最后一个维度或用户指定轴计算 softmax。

### 0.1 基本信息

| 项目 | 内容 |
|-----|------|
| 算子名称 | softmax |
| 算子类别 | Reduction / Norm（归约+广播变换） |
| 需求类型 | 通用 |
| 支持数据类型 | float16 (half), float32 (float) |
| 支持芯片 | Ascend 910B (DAV_2201) |
| 特殊约束 | 数值稳定性（max 减法防溢出）；SoftMax API 要求尾轴 32 字节对齐 |

### 0.2 方案决策

| 决策项 | 选择 | 理由 |
|--------|------|------|
| 技术路线 | 通用 SIMD/MemBase | 目标架构 DAV_2201，算子类型为 Reduction/vector，非 Matmul/Cube，不触发 RegBase/Blaze 路线 |
| 核心计算 API | SoftMax 高阶 API | 内部完成 ReduceMax→Broadcast→Sub→Exp→ReduceSum→Broadcast→Div 全流程，数值稳定（内置 max 减法），性能优化（支持 BasicBlock 模式），官方示例验证 |
| 归约模式 | AR 模式（尾轴） | Softmax 沿尾轴归约时，shape 合轴后为 (A1, R)，A0=1，属于 AR 模式 |
| 精度策略 | FP16 混合精度 / FP32 纯精度 | FP16 使用 SoftMax<half> 异质模板（内部 float 计算），FP32 使用 SoftMax<float> 同质模板 |

---

## 1. 算子设计

### 1.1 数学公式

```
输入: x — shape任意, dtype=float16或float32
输出: y — shape与x相同, dtype与x相同

// 数学公式（沿尾轴，即最后一维）
softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_i - max(x)))

// 逐行计算（每行 R 个元素）
对每行 row:
    max_val = max(row[0], row[1], ..., row[R-1])      // 数值稳定：减去最大值防 exp 溢出
    y[i]    = exp(row[i] - max_val) / sum(exp(row[j] - max_val), j=0..R-1)
```

### 1.2 API 映射

| 数学操作 | 对应 API | 关键参数 | 数据布局 | 官方文档 |
|---------|---------|---------|---------|---------|
| ReduceMax+Broadcast+Sub+Exp+ReduceSum+Broadcast+Div (组合) | `SoftMax<T>` | dst, sum, max, src, sharedTmpBuffer, tiling, shapeInfo | (M, alignedK) ND 格式，尾轴 32B 对齐 | [SoftMax.md](asc-devkit/docs/api/SIMD-API/高阶API/激活函数/SoftMax接口/SoftMax.md) |
| Tiling 参数计算 | `SoftMaxTilingFunc` | srcShape, dataTypeSize, localWorkSpaceSize | — | [SoftMax-SimpleSoftMax-Tiling.md](asc-devkit/docs/api/SIMD-API/高阶API/激活函数/SoftMax接口/SoftMax-SimpleSoftMax-Tiling.md) |
| 最小临时空间 | `GetSoftMaxMinTmpSize` | srcShape, dataTypeSize, isReuseSource | — | 同上 |
| BasicBlock 判断 | `IsBasicBlockInSoftMax` | tiling, dataTypeSize | — | [IsBasicBlockInSoftMax.md](asc-devkit/docs/api/SIMD-API/高阶API/激活函数/SoftMax接口/IsBasicBlockInSoftMax.md) |
| 极端值行后处理 | `AdjustSoftMaxRes<T1,T2>` | softMaxRes, maxTensor, from, to, shapeInfo | (M, K) ND 格式 | [AdjustSoftMaxRes.md](asc-devkit/docs/api/SIMD-API/高阶API/激活函数/SoftMax接口/AdjustSoftMaxRes.md) |
| GM→UB（对齐） | `DataCopy` | srcGm, dstLocal, count | 连续，32B 对齐 | 标准 API |
| GM→UB（非对齐） | `DataCopyPad` | srcGm, dstLocal, copyParams, padParams | 按 row 拷贝+填充 | 标准 API |
| UB→GM（对齐） | `DataCopy` | srcLocal, dstGm, count | 连续 | 标准 API |
| UB→GM（非对齐） | `DataCopy` (逐行) | srcLocal[rowOffset], dstGm[gmOffset], columnLength | 逐行拷贝有效数据 | 标准 API |

#### 1.2.1 API 语义验证

| API | 数据布局 | 功能需求 | API选择 | 限制条件 | 匹配 | 文档 |
|-----|---------|---------|---------|---------|-----|------|
| SoftMax (同质 float) | ND 格式，(M, alignedK)，尾轴 32B 对齐 | 沿尾轴 SoftMax 计算 | `SoftMax<float, false, isBasicBlock>(dst, sum, max, src, tmpBuf, tiling, shapeInfo)` | srcTensor 尾轴 32B 对齐；sum/max 尾轴固定 32B；dst 可复用 src | ✅ | [SoftMax.md](asc-devkit/docs/api/SIMD-API/高阶API/激活函数/SoftMax接口/SoftMax.md) |
| SoftMax (异质 half) | ND 格式，(M, alignedK)，尾轴 32B 对齐 | FP16 输入+FP32 内部计算 | `SoftMax<half, false, isBasicBlock>(dst<half>, sum<float>, max<float>, src<half>, tmpBuf, tiling, shapeInfo)` | dst=half, sum/max=float；内部 ReduceMax/ReduceSum 用 float 精度 | ✅ | 同上（异质模板签名） |
| SoftMaxTilingFunc | — | 计算 SoftMax 所需 Tiling 参数 | `SoftMaxTilingFunc(srcShape, dataTypeSize, localWorkSpaceSize, softmaxTiling)` | localWorkSpaceSize ≥ GetSoftMaxMinTmpSize 返回值 | ✅ | [SoftMax-SimpleSoftMax-Tiling.md](asc-devkit/docs/api/SIMD-API/高阶API/激活函数/SoftMax接口/SoftMax-SimpleSoftMax-Tiling.md) |
| GetSoftMaxMinTmpSize | — | 获取最小临时空间大小 | `GetSoftMaxMinTmpSize(srcShape, dataTypeSize, isReuseSource)` | dataTypeSize: half=2, float=4 | ✅ | 同上 |
| IsBasicBlockInSoftMax | — | 判断是否满足 BasicBlock 条件 | `IsBasicBlockInSoftMax(tiling, dataTypeSize)` | 返回 bool | ✅ | [IsBasicBlockInSoftMax.md](asc-devkit/docs/api/SIMD-API/高阶API/激活函数/SoftMax接口/IsBasicBlockInSoftMax.md) |
| AdjustSoftMaxRes | ND 格式，(M, alignedK) | max==指定值时将 softmax 结果置为自定义值 | `AdjustSoftMaxRes<T1, float>(softMaxRes, maxTensor, from, to, shapeInfo)` | from 为浮点数的十六进制表示；maxTensor 尾轴 32B 固定 | ✅ | [AdjustSoftMaxRes.md](asc-devkit/docs/api/SIMD-API/高阶API/激活函数/SoftMax接口/AdjustSoftMaxRes.md) |
| DataCopyPad | GM 连续 → UB 按 row 拷贝+填充 | 非对齐尾轴数据搬入 | `DataCopyPad(dstLocal, srcGm[offset], DataCopyExtParams{1, colBytes, 0,0,0}, DataCopyPadExtParams{false,0,0,0})` | blockLen 用有效长度（非对齐长度） | ✅ | 标准 API |

**验证清单**：

- [x] 1. SoftMax API 数据布局确认：ND 格式，尾轴 32B 对齐（通过 DataCopyPad 保证）
- [x] 2. SoftMax API 功能需求明确：沿尾轴 SoftMax，内部完成 max 减法+exp+sum+div
- [x] 3. 已查阅官方文档：SoftMax.md、SoftMax-SimpleSoftMax-Tiling.md、IsBasicBlockInSoftMax.md、AdjustSoftMaxRes.md
- [x] 4. 匹配验证：SoftMax 高阶 API 完整覆盖 softmax 数学定义，内置数值稳定处理
- [x] 5. 已记录验证过程：异质模板支持 FP16→FP32 内部计算，BasicBlock 模式提升性能

**SoftMax API 关键约束**：

| 约束 | 说明 | 设计应对 |
|------|------|---------|
| srcTensor 尾轴 32B 对齐 | API 硬性要求 | DataCopyPad 搬入时填充到对齐；SoftMaxShapeInfo 传入 oriSrcK |
| sum/max 尾轴固定 32B | 每个 datablock 存同一值 | 分配 tileRows × 32B 空间 |
| dst 可复用 src | 输入输出可同一 buffer | queueX 同时作为输入和输出 buffer |
| isBasicBlock 条件 | m%8==0, n<2048, n≥64(FP32)/128(FP16), n%64==0 | Tiling 中计算 isBasicBlock 标志，Kernel 侧条件选择 |
| oriSrcK vs srcK | oriSrcK=实际长度, srcK=对齐长度 | 非对齐场景：shapeInfo 传入两者；对齐场景：两者相同 |

### 1.3 数据流

```
输入 x (Global Tensor, shape=[A1, R])
    ↓ DataCopy / DataCopyPad
输入 x (Local Tensor, shape=[tileRows, alignedCols])
    ↓ SoftMax<T> 高阶 API
    │  内部流程：ReduceMax → Broadcast → Sub → Exp → ReduceSum → Broadcast → Div
    ↓ 得到 dst (Local Tensor), max (Local Tensor), sum (Local Tensor)
    ↓ AdjustSoftMaxRes (极端值行后处理)
输出 y (Local Tensor, shape=[tileRows, alignedCols])
    ↓ DataCopy (逐行或批量)
输出 y (Global Tensor, shape=[A1, R])
```

### 1.4 核心计算步骤

**核心计算步骤**：
```
1. CopyIn  — 从 GM 搬入 tileRows 行数据到 UB（对齐用 DataCopy，非对齐用 DataCopyPad）
2. Compute — 调用 SoftMax 高阶 API（内部完成 ReduceMax→Sub→Exp→ReduceSum→Div），
             然后调用 AdjustSoftMaxRes 处理极端值行（max==-FLT_MAX 时置结果为 0）
3. CopyOut — 从 UB 搬出 softmax 结果到 GM（对齐批量拷贝，非对齐逐行拷贝有效长度）
```

**分支差异对比**：

| 操作 | FP32 分支 | FP16 分支 |
|------|-----------|-----------|
| SoftMax 模板 | `SoftMax<float>` | `SoftMax<half>` (异质：sum/max 为 float) |
| 内部计算精度 | float | float（混合精度） |
| Input/Output dtype | float | half |
| sum/max dtype | float | float |
| dataTypeSize (Tiling) | 4 | 2（half 的 sizeof） |
| DataCopy 块大小 | 4 字节/元素 | 2 字节/元素 |
| 对齐最小列数 | 8 (32B/4B) | 16 (32B/2B) |

| 操作 | 对齐列 | 非对齐列 |
|------|--------|---------|
| CopyIn | `DataCopy` 批量搬入 | `DataCopyPad` 逐行搬入+填充 |
| CopyOut | `DataCopy` 批量搬出 | `DataCopy` 逐行搬出有效长度 |
| SoftMaxShapeInfo | srcK == oriSrcK | srcK = alignedCols, oriSrcK = columnLength |

| 操作 | BasicBlock | 非 BasicBlock |
|------|-----------|---------------|
| SoftMax 模板 | `SoftMax<T, false, true>` | `SoftMax<T, false>` |
| 性能 | 更优（内部使用基本块优化） | 标准 |

**关键设计要点**：
1. **Buffer 复用**：SoftMax API 支持 dst 复用 src（`xLocal` 同时作为输入和输出），减少 UB 占用
2. **数值稳定**：SoftMax API 内置 max 减法；AdjustSoftMaxRes 处理 max==-FLT_MAX 的极端行
3. **混合精度**：FP16 场景下 sum/max 使用 float 类型，保证 ReduceMax/ReduceSum 的精度
4. **对齐处理**：非对齐列长通过 DataCopyPad 填充，SoftMaxShapeInfo 区分 oriSrcK 和 srcK
5. **参数使用规则**：

| 参数位置 | 用有效长度 (columnLength) | 用对齐长度 (alignedCols) |
|---------|:---:|:---:|
| DataCopyPad blockLen | ✅ | ❌ |
| DataCopy count (对齐批量) | ✅ | ❌ |
| SoftMaxShapeInfo oriSrcK | ✅ | ❌ |
| SoftMaxShapeInfo srcK | ❌ | ✅ |
| UB Buffer 大小分配 | ❌ | ✅ |
| UB 内 rowOffset 计算 | ❌ | ✅ |

### 1.5 内存管理(Buffer 规划)

**FP32 场景（Kernel 侧）**：

| Buffer 名称 | 用途 | 大小计算 | TPosition |
|------------|------|---------|-----------|
| inQueueX | 输入+输出数据（dst 复用 src） | tileRows × alignedCols × sizeof(float) × BUFFER_NUM(2) | VECIN |
| queueMax | ReduceMax 中间结果 | tileRows × 32B × 1 | VECOUT |
| queueSum | ReduceSum 中间结果 | tileRows × 32B × 1 | VECOUT |
| sharedTmpBuffer | SoftMax 内部临时空间 | GetSoftMaxMinTmpSize | VECCALC |

**FP16 场景（Kernel 侧）**：

| Buffer 名称 | 用途 | 大小计算 | TPosition |
|------------|------|---------|-----------|
| inQueueX | 输入+输出数据 | tileRows × alignedCols × sizeof(half) × BUFFER_NUM(2) | VECIN |
| queueMax | ReduceMax 中间结果 (float) | tileRows × 32B × 1 | VECOUT |
| queueSum | ReduceSum 中间结果 (float) | tileRows × 32B × 1 | VECOUT |
| sharedTmpBuffer | SoftMax 内部临时空间 | GetSoftMaxMinTmpSize(half) | VECCALC |

**典型 UB 使用量估算（FP32, columnLength=1024, tileRows=8）**：
- inQueueX: 2 × 8 × 1024 × 4 = 64 KB
- queueMax: 8 × 32 = 256 B
- queueSum: 8 × 32 = 256 B
- sharedTmpBuffer: ≥ 4 KB
- **总 UB ≈ 68.5 KB**（远小于 192 KB 上限）

**典型 UB 使用量估算（FP32, columnLength=8192, tileRows=1）**：
- inQueueX: 2 × 1 × 8192 × 4 = 64 KB
- queueMax/queueSum: 32 B each
- sharedTmpBuffer: ≥ 4 KB
- **总 UB ≈ 68 KB**

---

## 2. 架构设计

### 2.1 多核切分策略

| 项目 | 说明 |
|-----|------|
| 切分维度 | 按 A1 维度（行方向）切分，每核处理若干连续行 |
| 单核任务量 | rowsPerCore = ceil(totalRows / blockDim) |
| 使用的核数 | **强制动态计算**：usedCoreNum = ceil(totalRows / rowsPerCore)；尾核可能少于 rowsPerCore 行 |
| 负载均衡方式 | 平均分配法：主核各处理 rowsPerCore 行，尾核处理 tailCoreRowNum 行 |

**分核计算公式**：
```
alignedRowNum = ceil(totalRows / coreNum) × coreNum   // 向上对齐到 coreNum 倍数
coreRowNum    = alignedRowNum / coreNum                 // 每核分配行数
tailCoreRowNum = totalRows % coreRowNum                 // 尾核行数（0 时所有核均匀）
usedCoreNum    = totalRows / coreRowNum                 // 实际使用核数
```

### 2.2 UB 切分策略

| 项目 | 说明 |
|-----|------|
| UB 容量 | 192 KB (DAV_2201 / 910B) |
| 单次处理数据量 | tileRows 行 × alignedCols 列，tileRows 由 SLICE_TABLE 决定 |
| 是否需要分 chunk | 是（UB 无法容纳所有行时，每核内再按 tileRows 分块循环处理） |
| chunk 大小计算 | SLICE_TABLE 查表：列数越大→每次处理行数越少 |

**SLICE_TABLE（列数→每次循环行数映射）**：

| 列数范围 ≥ | 每次循环行数 | 说明 |
|-----------|------------|------|
| 8192 | 1 | 大列，单行处理 |
| 4096 | 2 | |
| 2048 | 4 | |
| 1024 | 8 | |
| 512 | 16 | |
| 256 | 32 | |
| 0 | 64 | 小列，批量处理 |

**单核内循环分块参数**：
```
singleLoopCoreRowNum = SLICE_TABLE.lookup(columnLength)   // 每次循环行数
singleCoreLoopCount  = coreRowNum / singleLoopCoreRowNum  // 完整循环次数
singleCoreLoopTail   = coreRowNum % singleLoopCoreRowNum  // 尾块行数
```

### 2.3 分支场景覆盖

| 分支条件 | 处理策略 |
|---------|---------|
| 数据类型 FP32 | `SoftMax<float>` 同质模板，所有 buffer float 类型 |
| 数据类型 FP16 | `SoftMax<half>` 异质模板，sum/max 为 float，输入输出 half |
| 大 shape (A1 大) | 多核均分行数，每核内分 tile 循环 |
| 小 shape (A1 ≤ coreNum) | 仅使用 A1 个核，每核 1 行 |
| 尾轴对齐 (columnLength % 32B对齐) | DataCopy 批量搬入/搬出，SoftMaxShapeInfo srcK==oriSrcK |
| 尾轴非对齐 | DataCopyPad 逐行搬入+填充，SoftMaxShapeInfo 区分 oriSrcK/srcK |
| BasicBlock 条件满足 | isBasicBlock=true，使用高性能路径 |
| BasicBlock 条件不满足 | isBasicBlock=false，标准路径 |
| 极端值行 (max==-FLT_MAX) | AdjustSoftMaxRes 将对应行结果置为 0 |
| 尾核行数不同 | 尾核使用独立的 singleLoopCoreRowNum/loopCount/loopTail |

### 2.4 类别特有设计

#### 2.4.1 尾轴 Softmax（Primary Branch — AR 模式）

**适用场景**：axis = -1 或 axis = last_dim（最常见场景）

**Compute 核心流程伪代码**：

```cpp
// Kernel 核函数入口
__global__ __vector__ void softmax_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling) {
    TPipe pipe;
    VecTiling tilingData;
    CopyTiling(&tilingData, tiling);   // 从 GM 拷贝 tiling 数据
    
    KernelSoftmax op;
    op.Init(x, y, tilingData, &pipe);
    op.Process();
}

// Process: 分 tile 循环处理
void Process() {
    if (GetBlockIdx() > usedCoreNum) return;  // 超出实际核数直接返回
    
    // 尾核使用独立分块参数
    uint32_t loopRows = singleLoopCoreRowNum;
    uint32_t loopCount = singleCoreLoopCount;
    uint32_t loopTail = singleCoreLoopTail;
    if (GetBlockIdx() == usedCoreNum) {
        loopRows = tailCoreSingleLoopCoreRowNum;
        loopCount = tailCoreSingleCoreLoopCount;
        loopTail = tailCoreSingleCoreLoopTail;
    }
    
    for (uint32_t i = 0; i < loopCount; i++) {
        CopyIn(i, loopRows);
        Compute(i, loopRows);
        CopyOut(i, loopRows);
    }
    if (loopTail > 0) {
        CopyIn(loopCount, loopTail);
        Compute(loopCount, loopTail);
        CopyOut(loopCount, loopTail);
    }
}

// CopyIn: GM → UB
void CopyIn(uint32_t progress, uint32_t rowNum) {
    LocalTensor<T> xLocal = inQueueX.AllocTensor<T>();
    if (isAligned) {
        // 对齐场景：批量拷贝
        DataCopy(xLocal, xGm[progress * tileLength], rowNum * alignedCols);
    } else {
        // 非对齐场景：逐行 DataCopyPad
        for (uint32_t row = 0; row < rowNum; row++) {
            DataCopyExtParams copyParams{1, columnLength * sizeof(T), 0, 0, 0};
            DataCopyPadExtParams<T> padParams{false, 0, 0, 0};
            DataCopyPad(xLocal[row * alignedCols], xGm[gmOffset + row * columnLength], copyParams, padParams);
        }
    }
    inQueueX.EnQue(xLocal);
}

// Compute: SoftMax + AdjustSoftMaxRes
void Compute(uint32_t progress, uint32_t rowNum) {
    LocalTensor<T> xLocal = inQueueX.DeQue<T>();
    LocalTensor<float> maxLocal = queueMax.AllocTensor<float>();
    LocalTensor<float> sumLocal = queueSum.AllocTensor<float>();
    LocalTensor<uint8_t> tmpBuffer = sharedTmpBuffer.Get<uint8_t>();
    
    SoftMaxShapeInfo srcShape = {rowNum, alignedCols, rowNum, columnLength};
    
    // 条件选择 BasicBlock 模式
    if (isBasicBlock) {
        SoftMax<T, false, true>(xLocal, sumLocal, maxLocal, xLocal, tmpBuffer, softmaxTiling, srcShape);
    } else {
        SoftMax<T, false>(xLocal, sumLocal, maxLocal, xLocal, tmpBuffer, softmaxTiling, srcShape);
    }
    
    // 极端值行后处理：max == 0xFF7FFFFF (-FLT_MAX) 时，将 softmax 结果置为 0
    AdjustSoftMaxRes<T, float>(xLocal, maxLocal, ADJUST_FROM, ADJUST_TO, srcShape);
    
    inQueueX.EnQue<T>(xLocal);
    queueMax.EnQue<float>(maxLocal);
    queueSum.EnQue<float>(sumLocal);
}

// CopyOut: UB → GM
void CopyOut(uint32_t progress, uint32_t rowNum) {
    LocalTensor<T> yLocal = inQueueX.DeQue<T>();
    LocalTensor<float> maxLocal = queueMax.DeQue<float>();
    LocalTensor<float> sumLocal = queueSum.DeQue<float>();
    
    if (isAligned) {
        // 对齐场景：批量拷贝（只拷贝有效列数，alignedCols == columnLength）
        DataCopy(yGm[progress * tileLength], yLocal, rowNum * columnLength);
    } else {
        // 非对齐场景：逐行拷贝有效长度
        for (uint32_t row = 0; row < rowNum; row++) {
            DataCopy(yGm[gmOffset + row * columnLength], yLocal[row * alignedCols], columnLength);
        }
    }
    
    inQueueX.FreeTensor(yLocal);
    queueMax.FreeTensor(maxLocal);
    queueSum.FreeTensor(sumLocal);
}
```

**Buffer 需求**：

| Buffer 名称 | 用途 | 大小计算 |
|------------|------|---------|
| inQueueX | 输入+输出（dst 复用 src） | tileRows × alignedCols × sizeof(T) × 2 (Double Buffer) |
| queueMax | ReduceMax 中间结果 | tileRows × 32B |
| queueSum | ReduceSum 中间结果 | tileRows × 32B |
| sharedTmpBuffer | SoftMax 内部临时计算空间 | GetSoftMaxMinTmpSize(tileShape, sizeof(T), false) |

---

## 3. 确认清单

- [x] 多核切分策略已确定（按 A1 行切分，平均分配法）
- [x] UB 切分策略已确定（SLICE_TABLE 查表决定 tileRows）
- [x] Buffer 规划已完成（inQueueX + queueMax + queueSum + sharedTmpBuffer）
- [x] 分支场景已覆盖（FP32/FP16、对齐/非对齐、BasicBlock/非BasicBlock、极端值行）
- [x] 类别特有设计已完成（AR 模式尾轴 Softmax，含完整伪代码）
- [x] API 映射已验证（SoftMax、SoftMaxTilingFunc、AdjustSoftMaxRes 等均查阅官方文档）