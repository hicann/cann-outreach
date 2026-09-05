# 东南大学

## 团队信息

- 提交者: 程康博
- 身份: 学生
- 单位: 东南大学

## 成员

- 程康博 (ArgoNorthrop): 提交者

## 算子: op_04_square

# Feature Request: Square 算子实现参考文档 — AscendC 开源仓算子开发指南

**Template**: Open Source Operator Repository Template (Problem #1740, Square)
**CANN Version**: 9.0.0
**Target Hardware**: Ascend 910B
**Status**: ✅ Pass（3/3 测试点，Wrong Answer 0.00%）
**关联赛题**: https://github.com/Ascend/ascend-cann

---

## 1. 背景与目标

### 1.1 算子定义
Square（平方）算子：`y = x × x`，对输入张量每个元素独立计算平方。输出 shape 与 dtype 同输入。

### 1.2 输入约束（来自赛题）
| 维度 | 约束 |
|------|------|
| Shape | 任意多维 `(..., N)`，N ∈ [1, 10240] |
| 非对齐 | N 可能非 32 字节对齐 |
| DataType | 仅 float16 / float32 |

### 1.3 模板结构
开源仓规范化模板，需要填的只有 2 个文件：
- `op_host/square_tiling.cpp` — Tiling 计算函数
- `op_kernel/square.h` — Kernel 五段式实现（Init / CopyIn / Compute / CopyOut / Process）

---

## 2. 核心踩坑点与解决方案

### 坑点 1：Host 侧 TilingFunc 如何正确拿到 totalNum

**❌ 错误尝试**：
```cpp
// 误以为 TilingContext::GetInputShape 返回 Shape*
const gert::Shape* input_shape = context-&gt;GetInputShape(0);  // 类型不匹配！
input_shape-&gt;GetShapeSize();  // CompileTimeTensorDesc 根本没有这个方法！
```
`TilingContext::GetInputShape` 返回的是 `const gert::StorageShape*`，**不是** `const gert::Shape*`。两者 API 完全不同。

**✅ 正确写法（Relu / Square / Mul 三题验证通过）**：
```cpp
const gert::StorageShape* input_shape = context-&gt;GetInputShape(0);
OP_CHECK_NULL_WITH_CONTEXT(context, input_shape);
// EnsureNotScalar 处理 rank=0 标量输入
gert::Shape safe_shape = EnsureNotScalar(input_shape-&gt;GetStorageShape());
uint64_t totalNumRaw = 1;
for (size_t d = 0; d &lt; safe_shape.GetDimNum(); ++d) {
    totalNumRaw *= safe_shape.GetDim(d);
}
tiling-&gt;totalNum = static_cast&lt;int64_t&gt;(totalNumRaw);
```

**关键认知**：`TilingContext` 和 `InferShapeContext` 是两个不同的 context 类型，`GetInputShape` 返回值类型也不同。别猜 API，看编译报错再修。

---

### 坑点 2：Kernel 侧非对齐长度 → 0.31% Wrong Answer

**现象**：Square 测试点 2 报 Wrong Answer 0.31%，其他 2 个 Pass。

**根因**：AscendC 的向量化 API（`Mul`、`Relu`、`Add` 等）底层是按 8 lane（fp32）或 16 lane（fp16）读数据的。如果传给 API 的长度 `currentNum` 不是 `alignElements` 的倍数，**最后一块向量化会越界读** UB 缓冲区尾部残留的脏数据，直接参与计算，输出就是错的。

Square 模板的题目明确写了 "N 可能非 32 对齐"。当 N = 10001 时，最后一块余 1，如果传 1 给 `AscendC::Mul(y, x, x, 1)`，底层按 8 lane 读就会多读 7 个脏值。

**✅ 修复（Process 里做双重 CeilAlign）**：
```cpp
// alignElements：fp32=8, fp16=16（32B / sizeof(T)）
constexpr int64_t ALIGN_FP32 = 8;
constexpr int64_t ALIGN_FP16 = 16;
constexpr int64_t alignElements = (sizeof(T) == 4) ? ALIGN_FP32 : ALIGN_FP16;

// 每核真实长度 → CeilAlign
int64_t coreLenRaw = min(blockLength_, totalNum - coreOffset);
int64_t coreLen = (coreLenRaw + alignElements - 1) / alignElements * alignElements;

// 每个 UB 块的 currentNum → CeilAlign
int64_t currentNum = (currentNumReal + alignElements - 1) / alignElements * alignElements;
if (currentNum &gt; ubLength_) currentNum = ubLength_;  // 防越界
```

**为什么 Relu 没出这个问题？** Relu 测试点 shape 是固定的 `8×2048=16384`，刚好整除 8 和 16。Square 题目 shape 是任意 N∈[1,10240]，才触发了非对齐边界。

**结论**：所有传给 AscendC 向量化 API 的长度参数，**必须保证是 alignElements 的倍数**。host 侧 blockFactor / ubFactor / kernel 侧 currentNum 都要做 CeilAlign。

---

### 坑点 3：`op_host/include tiling_key → kernel_operator.h 找不到`

**现象**：host 侧写 `#include "../op_kernel/square_tiling_key.h"` 后，编译报 GCC 找不到 `kernel_operator.h`。

**根因**：`square_tiling_key.h` 里用了 `GET_TPL_TILING_KEY` 宏，该宏展开间接 include 了 AscendC kernel 的头文件 `kernel_operator.h`，而 host 编译环境里没有这个 include path。

**✅ 解法**：host 侧**不要** include `*_tiling_key.h`。模板本身的 CMakeLists.txt 已经自动处理 tiling key 的链接，host 侧只需要 include 自己需要的 tiling_data.h。

---

## 3. Square 算子技术实现总结

### 3.1 AscendC API 选择
AscendC **没有独立的 `AscendC::Square` API**，用 `AscendC::Mul(y, x, x, n)` 实现。这点和 Relu（有独立 API）不同。

### 3.2 Host 侧 TilingFunc 逻辑（6 步）
```
1. GetPlatformInfo → ubSize, coreNum
2. GetWorkspaceSize → 默认 0
3. totalNum = ∏ input_shape.dims（StorageShape + 手动乘）
4. alignElements = 32 / dtypeSize（fp32=8, fp16=16）
5. blockFactor = CeilAlign(CeilDiv(totalNum, coreNum), alignElements)
   blockDim   = max(1, CeilDiv(totalNum, blockFactor))
6. ubFactor   = CeilAlign(2048, alignElements)
   tilingKey  = input dtype 决定（schMode=0→half, 1→float）
```

### 3.3 Kernel 侧 Process 双重对齐
```
每核 coreLenRaw → CeilAlign(alignElements) → coreLen
每块 currentNumReal → CeilAlign(alignElements) → currentNum（截断到 ubLength_）
```

### 3.4 Kernel 类成员变量（模板内必需）
```cpp
private:
    TPipe pipe;
    TQue&lt;QuePosition::VECIN,  BUFFER_NUM&gt; inputQueueX;
    TQue&lt;QuePosition::VECOUT, BUFFER_NUM&gt; outputQueueY;
    GlobalTensor&lt;T&gt; inputGMX;
    GlobalTensor&lt;T&gt; outputGMY;
    const SquareTilingData* tilingDataPtr_ = nullptr;  // Process 里读 totalNum
    int64_t blockLength_ = 0;
    int64_t ubLength_ = 0;
```

---

## 4. 4 道 AscendC 赛题统一模式对比

| 题目 | Kernel 类型 | Host Tiling 复杂度 | 对齐策略 | Pass 状态 |
|------|-------------|--------------------|----------|----------|
| Sub（Problem 467） | 单文件 kernel.asc | 无 host，Python 脚本 | 写死 blockFactor | ✅ |
| Mul（Problem 470） | op_kernel/mul.cpp | blockFactor + ubFactor | CeilAlign 32 | ✅ |
| Relu（Problem 473） | op_kernel/relu.h | totalNum + blockFactor + ubFactor | CeilAlign 32 | ✅ |
| **Square（Problem 1740）** | op_kernel/square.h | totalNum + blockFactor + ubFactor + **Kernel 侧二次对齐** | **CeilAlign 32 + Process 里对 currentNum 再次对齐** | ✅ |

**演进规律**：从写死 tiling → host 侧动态计算 → host + kernel 两侧都要对齐，题目难度逐步提升。最后一题 Square 之所以要 kernel 侧二次对齐，就是因为题目明确说了"N 可能非 32 对齐"。

---

## 5. 建议 / 可改进点

### 5.1 对 AscendC 框架侧
1. **建议在 AscendC 文档里明确标注**：向量化 API（Mul / Relu / Add 等）的长度参数**必须是 32B / sizeof(dtype) 的倍数**，否则行为未定义
2. **建议 AscendC 增加 Square API**：目前没有独立 Square，必须用 Mul(x, x, n) 绕一下
3. **建议为 Kernel 侧长度参数做 runtime assert**：检测到非对齐直接报错，而不是 silently 越界读

### 5.2 对模板侧
1. **建议模板自带 alignElements 计算工具**：host 侧和 kernel 侧重复写 `32 / dtypeSize` 很容易写错（host 侧有 dtype 信息，kernel 侧靠模板参数 T）
2. **建议 TilingContext::GetInputShape 返回类型文档化**：当前 StorageShape* vs Shape* 的区别只在编译报错时才能发现，没有前置说明
3. **建议模板注释里直接写清楚**："所有传给 AscendC 向量化 API 的长度必须 CeilAlign(32/dtypeSize)"，避免用户踩坑

### 5.3 对赛题侧
1. **题目描述里已经明确提了 "N 可能非 32 对齐"**，这点做得很好，和最终的难点一致
2. **建议再增加 1 个 N=31 / N=33 这样的边界测试点**，专门检测非对齐 case

---

## 6. 完整改动清单

| 文件 | 改动 | 备注 |
|------|------|------|
| `op_kernel/square.h` | 完整填充 Init / CopyIn / Compute / CopyOut / Process 五段式 | Compute 用 Mul(y,x,x,n) 实现 Square；Process 做双重 CeilAlign |
| `op_host/square_tiling.cpp` | 填充 SquareTilingFunc 的 totalNum / blockFactor / ubFactor / SetBlockDim | 用 StorageShape* + 手动乘拿 totalNum；复用 inputDesc 避免重定义 |

**未改动文件**：square.cpp、square_tiling_data.h、square_tiling_key.h、square_infershape.cpp、square_def.cpp（模板已完整）

---

*本 Issue 基于 AscendC 开源仓算子开发模板，总结 Square 算子（Problem #1740）开发过程中的踩坑点与解决方案，希望能帮助后续开发者少走弯路。*
