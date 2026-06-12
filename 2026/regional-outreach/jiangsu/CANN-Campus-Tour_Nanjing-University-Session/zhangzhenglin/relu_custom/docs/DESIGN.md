# relu_custom 算子设计与实施文档

---

## 0. 概述

### 0.0 需求类型判断

**特定用例**：用户明确指定了 4D shape、float16 dtype、ND 格式。

### 0.1 基本信息

| 项目 | 内容 |
|-----|------|
| 算子名称 | relu_custom |
| 算子类别 | Elementwise（一元逐元素激活函数） |
| 需求类型 | 特定用例（4D shape=[N4,N3,N2,N1], dtype=float16, Format=ND） |
| 支持数据类型 | float16 |
| 支持数据格式 | ND |
| 支持芯片 | 910B3 |

### 0.2 用户原始需求

| # | 需求内容 |
|---|---------|
| 1 | 实现 ReLU 算子：y = max(0, x)，逐元素计算 |
| 2 | 支持 float16 数据类型 |
| 3 | 支持 4D shape，如 [N4, N3, N2, N1] |
| 4 | 数据格式为 ND（连续内存布局） |
| 5 | 精度标准：与 PyTorch `torch.relu` 对比，atol=1e-3, rtol=1e-3 |
| 6 | 提供可执行文件通路和 PyTorch 通路 |

---

## 1. 算子设计

### 1.1 数学公式

```
输入: x - shape [N4, N3, N2, N1], dtype=float16
输出: y - shape [N4, N3, N2, N1], dtype=float16

数学公式:
y[i] = ReLU(x[i]) = max(0, x[i]) = x[i] > 0 ? x[i] : 0
```

### 1.2 API 映射

| 数学操作 | 对应 API | 关键参数 | 数据布局 | 备注 |
|---------|---------|---------|---------|------|
| 数据搬运 GM→UB | `DataCopyPad` | `(dst, src, {blkLen, rLen, rLenAlign, 0, 0})` | 连续 1D | 对齐访问 |
| ReLU 计算 | `Relu` | `(yLocal, xLocal, count)` | 连续 1D | y=max(0,x) |
| 数据搬运 UB→GM | `DataCopyPad` | `(dst, src, {blkLen, rLen, rLenAlign, 0, 0})` | 连续 1D | 写回结果 |

#### 1.2.1 API 语义验证

| API | 数据布局 | 功能需求 | 限制条件 | 匹配 |
|-----|---------|---------|---------|------|
| `DataCopyPad` | GM/UB 连续 1D | 搬运数据 | blockLen 以 32B 为单位 | ✅ |
| `Relu` | UB 连续 1D 向量 | 激活函数：y=max(0,x) | 无特殊限制 | ✅ |

### 1.3 数据流

```
输入 x (GlobalTensor, GM float16)
    ↓ DataCopyPad
输入 x (LocalTensor, UB float16)
    ↓ Relu
输出 y (LocalTensor, UB float16)
    ↓ DataCopyPad
输出 y (GlobalTensor, GM float16)
```

### 1.4 核心计算步骤

```
对于每个 tile:
1. DataCopyPad: GM→UB 搬运 x
2. EnQue / DeQue 同步
3. Relu: y = max(0, x)
4. DataCopyPad: UB→GM 写回 y
```

### 1.5 内存管理 (Buffer 规划)

| Buffer 名称 | 用途 | 大小计算 | TPosition |
|------------|------|---------|-----------|
| inQueueX | 输入 x 数据 | BUFFER_NUM * tileLen * sizeof(half) | VECIN |
| outQueueY | 输出 y 数据 | BUFFER_NUM * tileLen * sizeof(half) | VECOUT |

**总 UB 使用量**: 2 * BUFFER_NUM * tileLen * sizeof(half) = 4 * tileLen * 2 = 8 * tileLen bytes
- 相比 atanh，ReLU 只需要一对 I/O buffer，不需要额外的 VECCALC 暂存 buffer
- tileLen 可更大，单次处理更多元素

---

## 2. 架构设计

### 2.1 多核切分策略

| 项目 | 说明 |
|-----|------|
| 切分维度 | 按 totalElements 均分（4D ND 格式展平为 1D 连续） |
| 单核任务量 | totalElements / usedCoreNum（末核处理余数） |
| 使用的核数 | `min(totalElements, ACL_DEV_ATTR_VECTOR_CORE_NUM)` 动态获取 |
| 负载均衡方式 | 首核多处理 tail 元素（如有余数） |

### 2.2 UB 切分策略

| 项目 | 说明 |
|-----|------|
| UB 容量 | 192KB (910B3) |
| 单次处理数据量 | 双缓冲下每 buffer 约 1/4 UB 容量 |
| 是否需要分 chunk | 大 shape 需要 |
| tileLen 计算公式 | tileLen = min(剩余元素数, UB可用 / (2*sizeof(half)*BUFFER_NUM)) |

### 2.3 分支场景覆盖

| 分支条件 | 处理策略 |
|---------|---------|
| 小 shape（1 tile 内） | 单 tile：搬运 → Relu → 写回 |
| 大 shape（多 tile） | 循环分 tile |
| 非对齐 | DataCopyPad 自动处理 |
| 尾 tile | curLen = 实际剩余元素数 |

### 2.4 类别特有设计

#### 2.4.1 relu_custom

**适用场景**：所有 4D shape, float16, ND

**Compute 核心流程伪代码**：

```cpp
for (int32_t i = 0; i < loopCount; i++) {
    int32_t offset = i * tileLen;
    int32_t curLen = min(remaining, tileLen);
    remaining -= curLen;

    // 搬运 x
    DataCopyPad(xLocal, xGm[offset], {curLen, curLen, alignLen, 0, 0});
    EnQue(inQueueX, xLocal);
    DeQue(inQueueX, xLocal);

    // ReLU: y = max(0, x)
    Relu(yLocal, xLocal, curLen);

    // 写回 y
    EnQue(outQueueY, yLocal);
    DeQue(outQueueY, yLocal);
    DataCopyPad(yGm[offset], yLocal, {curLen, curLen, alignLen, 0, 0});
}
```

**Buffer 需求**：

| Buffer | 用途 | 大小 |
|--------|------|------|
| xLocal | 输入缓存 | tileLen * sizeof(half) |
| yLocal | 输出缓存 | tileLen * sizeof(half) |

---

## 3. 确认清单
- [x] 多核切分策略已确定
- [x] UB 切分策略已确定
- [x] Buffer 规划已完成
- [x] 分支场景已覆盖
- [x] 类别特有设计已完成
