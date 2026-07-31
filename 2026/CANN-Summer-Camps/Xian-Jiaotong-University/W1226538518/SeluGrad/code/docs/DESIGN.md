# SeluGrad 算子设计文档

## 1. 概述

### 1.1 基本信息

| 项目 | 内容 |
|-----|------|
| 算子名称 | SeluGrad |
| 算子类别 | Elementwise |
| 支持数据类型 | BF16 / FP16 / FP32 / INT32 / INT8 / UINT8 |
| **目标芯片** | Ascend910B / Ascend910_93 |
| **目标架构** | arch22 (DAV_2201) |

### 1.2 算子功能

SeluGrad 是 SELU（Scaled Exponential Linear Unit）激活函数的梯度算子，用于神经网络反向传播。给定上游梯度 `gradients` 和 SELU 前向输出 `outputs`，计算当前层输入的梯度。

### 1.3 数学公式

SELU 函数定义：

$$
\text{SELU}(x) = \lambda \cdot
\begin{cases}
x & \text{if } x \geq 0 \\
\alpha \cdot (e^x - 1) & \text{if } x < 0
\end{cases}
$$

其中 $\lambda \approx 1.050700987$，$\alpha \approx 1.673263242$。

SeluGrad 计算输出的梯度（记 $y = \text{SELU}(x)$）：

$$
\frac{\partial \mathcal{L}}{\partial x} =
\frac{\partial \mathcal{L}}{\partial y} \cdot
\begin{cases}
\lambda & \text{if } y \geq 0 \\
y + \lambda \cdot \alpha & \text{if } y < 0
\end{cases}
$$

推导：当 $y \geq 0$ 时 $x \geq 0$，$\frac{dy}{dx} = \lambda$；当 $y < 0$ 时 $y = \lambda \cdot \alpha \cdot (e^x - 1)$，$\frac{dy}{dx} = \lambda \cdot \alpha \cdot e^x = \frac{y}{\alpha \cdot (e^x - 1)} \cdot \alpha \cdot e^x = \cdots = y + \lambda \cdot \alpha$。

**关键常数**：

| 常量 | 值 | 说明 |
|-----|-----|------|
| `SCALE` ($\lambda$) | 1.0507009873554805f | SELU scale 因子 |
| `SCALE_ALPHA_PRODUCT` ($\lambda \cdot \alpha$) | 1.7580993408473769f | 梯度计算中间常数 |

---

## 2. 架构设计

### 2.1 逻辑视图

**模块职责**：

| 模块 | 职责 | 核心文件 |
|------|------|---------|
| **op_host** | Host 侧逻辑：算子定义注册、Shape 推导、Tiling 切分计算 | `selu_grad_def.cpp`, `selu_grad_tiling.cpp`, `selu_grad_infershape.cpp` |
| **op_kernel** | Kernel 侧实现：NPU 上的数据搬运与向量化计算 | `selu_grad.cpp`, `selu_grad.h`, `selu_grad_tiling_key.h`, `selu_grad_tiling_data.h` |

**模块依赖**：

```
op_host (Tiling)
  └── GetTilingData() ──▶ op_kernel (TilingData 下发给 Device)
```

### 2.2 开发视图

```
selu_grad/
├── build.sh                      # 构建脚本
├── CMakeLists.txt                # 构建配置
├── README.md                     # 算子说明
├── docs/
│   └── DESIGN.md                 # 详细设计文档
├── examples/
│   ├── CMakeLists.txt
│   ├── run.sh
│   └── test_aclnn_selu_grad.cpp  # ACLNN 调用示例
├── op_host/                      # Host 侧实现
│   ├── CMakeLists.txt
│   ├── selu_grad_def.cpp         # 算子原型注册
│   ├── selu_grad_infershape.cpp  # Shape 推导
│   └── selu_grad_tiling.cpp      # Tiling 实现
├── op_kernel/                    # Kernel 侧实现
│   ├── CMakeLists.txt
│   ├── selu_grad.cpp             # Kernel 入口（6 调度模式）
│   ├── selu_grad.h               # Kernel 类模板实现
│   ├── selu_grad_tiling_data.h   # TilingData 结构体
│   └── selu_grad_tiling_key.h    # 模板参数声明
└── tests/                        # 测试代码
    ├── ut/                       # 单元测试
    └── ...
```

### 2.3 运行视图

**数据流**：

```
GM (Global Memory)
  │
  │ DataCopyPad (GM → UB, 含对齐填充)
  ▼
UB (Unified Buffer)
  │
  │ [可选] Cast (RawT → ComputeT)
  │ CompareScalar (output < 0) → mask
  │ Adds (tmp = output + SCALE_ALPHA_PRODUCT)
  │ Select (mask ? tmp : SCALE)
  │ Mul (y = grad × factor)
  │ [可选] Cast (ComputeT → RawT)
  ▼
UB (Unified Buffer)
  │
  │ DataCopyPad (UB → GM)
  ▼
GM (Global Memory)
```

**执行流程**：

```
Host 侧:
  GetTilingData()
    ├── GetPlatformInfo → ubSize, coreNum
    ├── 根据 dtype 计算 rawTypeSize / computeTypeSize / ubBytesPerElement
    ├── 确定核数 & blockFactor & ubFactor
    └── 选择调度模式 → SetTilingKey()

Device 侧:
  Init()
    ├── 解析 TilingData → blockOffset, blockLength, ubLength
    ├── SetGlobalBuffer 建立 GM Tensor
    └── InitBuffer 分配 UB 缓冲区 + TBuf

  Process()
    └── for (progress = 0; progress < blockLength; progress += currentNum)
          ├── CopyIn():  DataCopyPad GM → UB (双缓冲)
          ├── Compute(): 计算逻辑 (向量化 + 标量尾块)
          └── CopyOut(): DataCopyPad UB → GM
```

---

## 3. 实现方案

### 3.1 模板划分总览

采用 6 个独立调度模式，每个数据类型对应一个独立的模板实例化。不使用复合模板参数，每个模式原子化。

**模板参数定义**：

| 参数 | 类型 | 取值范围 | 说明 |
|-----|------|---------|------|
| schMode | uint32_t | {0,1,2,3,4,5} | 调度模式，对应 6 种数据类型 |

**模板划分表**：

| 模板 | 触发条件 | RawT | ComputeT | NeedCast | 适用场景 |
|-----|---------|------|---------|:--------:|---------|
| FP16 | dtype == DT_FLOAT16 | half | half | false | FP16 直算 |
| FP32 | dtype == DT_FLOAT | float | float | false | FP32 直算 |
| BF16 | dtype == DT_BF16 | bfloat16_t | float | true | BF16 → float 升精度计算 → 回写 |
| INT32 | dtype == DT_INT32 | int32_t | float | true | INT32 → float 升精度计算 → 回写 |
| INT8 | dtype == DT_INT8 | int8_t | half | true | INT8 → half 升精度计算 → 回写 |
| UINT8 | dtype == DT_UINT8 | uint8_t | half | true | UINT8 → half 升精度计算 → 回写 |

### 3.2 TilingData 结构体

**文件位置**：`op_kernel/selu_grad_tiling_data.h`

```cpp
struct SeluGradTilingData {
    uint64_t totalNum = 0;     // 总元素数量
    uint32_t blockFactor = 1;  // 每个核处理的元素数量
    uint32_t ubFactor = 0;     // 每次 UB 循环处理的元素数量
};
```

### 3.3 Tiling 计算逻辑

**文件**：`op_host/selu_grad_tiling.cpp`

#### 3.3.1 输入校验

- 校验两个输入 `gradients` 和 `outputs` 的 dtype 一致
- 校验两组 Shape 的 dim 数一致、各维 size 一致
- 空 Tensor（totalNum == 0）：设 blockDim=1，直接返回

#### 3.3.2 数据类型映射

| Host DataType | rawTypeSize | computeTypeSize | ubBytesPerElement | 调度模式 |
|-------------|:----------:|:--------------:|:-----------------:|:-------:|
| DT_FLOAT16 | 2 | 2 | 6×2 + 1×2 + 1 = 15 | FP16 |
| DT_FLOAT | 4 | 4 | 6×4 + 1×4 + 1 = 29 | FP32 |
| DT_BF16 | 2 | 4 | 6×2 + 4×4 + 1 = 29 | BF16 |
| DT_INT32 | 4 | 4 | 6×4 + 4×4 + 1 = 41 | INT32 |
| DT_INT8 | 1 | 2 | 6×1 + 4×2 + 1 = 15 | INT8 |
| DT_UINT8 | 1 | 2 | 6×1 + 4×2 + 1 = 15 | UINT8 |

`ubBytesPerElement` 计算逻辑：`6 × rawTypeSize + computeBufferNum × computeTypeSize + 1`
- `computeBufferNum = 1`（NeedCast=false，直算路径）
- `computeBufferNum = 4`（NeedCast=true，需额外 Compute 缓冲区）
- `+1` 为 mask 缓冲区的 uint8_t

#### 3.3.3 核数与 blockFactor

采用自适应核数策略：

```
总字节数 = totalNum × rawTypeSize
if 总字节数 ≤ SMALL_BYTES_PER_CORE × SMALL_CORE_LIMIT (64KB):
    展开到最多 8 个核，每核 SMALL_BYTES_PER_CORE (8KB)
else:
    每核 LARGE_BYTES_PER_CORE (32KB)
核数 = min(计算核数, 物理核数)
blockFactor = align_up(ceil_div(totalNum, 核数), 512 / rawTypeSize)
```

512 字节对齐保证 GM 搬运对齐到 32 元素的 cache line 边界。

#### 3.3.4 ubFactor

```
可用 UB = ubSize - RESERVED_UB_BYTES (8KB)
maxUbFactor = floor(可用UB / ubBytesPerElement)
ubFactor = max(alignNum, min(blockFactor, maxUbFactor))
```

---

### 3.4 Kernel 实现

#### 3.4.1 类模板

```cpp
template <typename RawT, typename ComputeT, bool NeedCast>
class SeluGrad { ... };
```

| 参数 | 说明 |
|-----|------|
| RawT | GM 中原始数据类型（如 int8_t, half, float） |
| ComputeT | UB 中计算类型（half 或 float） |
| NeedCast | 是否需要显式 Cast 转换 |

#### 3.4.2 Init 阶段

```
1. blockOffset = blockIdx × blockFactor（元素偏移）
2. blockLength = min(剩余元素数, blockFactor)（边界保护）
3. ubLength = ubFactor
4. SetGlobalBuffer 建立 GM Tensor（梯度/输出/结果）
5. bufferNum = (blockLength > ubLength) ? 2 : 1（双缓冲/单缓冲）
6. InitBuffer 分配队列缓冲区 × bufferNum
7. 若 NeedCast，InitBuffer 分配 Compute 缓冲区
8. InitBuffer 分配 tmpBuf 和 maskBuf（单次分配，复用）
```

#### 3.4.3 CopyIn 阶段

使用 `DataCopyPad` 处理非对齐搬运：

```
copyBytes = currentNum × sizeof(RawT)
alignedBytes = align_up(copyBytes, 32)
rightPadding = (alignedBytes - copyBytes) / sizeof(RawT)
DataCopyPad(gradLocal, gradGm[progress], {copyBytes, ...}, {padding, ...})
DataCopyPad(outLocal, outGm[progress], {copyBytes, ...}, {padding, ...})
```

#### 3.4.4 Compute 阶段

**向量化路径（处理对齐部分）**：

```
vectorAlign = 256 / sizeof(ComputeT)    // 256 字节向量宽度
vectorNum = (currentNum / vectorAlign) × vectorAlign

if vectorNum > 0:
    1. CompareScalar(mask, outputs, 0, CMPMODE::LT, vectorNum)
       → mask 位图标记 outputs < 0 的位置
    2. Adds(tmp, outputs, SCALE_ALPHA_PRODUCT, vectorNum)
       → tmp = output + λ·α（负值分支的梯度系数）
    3. Select(outputs, mask, tmp, SCALE, VSEL_TENSOR_SCALAR_MODE, vectorNum)
       → outputs = mask ? tmp : SCALE（选择梯度系数）
    4. Mul(yCompute, gradCompute, outputs, vectorNum)
       → 最终梯度 = grad × factor
```

**标量尾块（处理剩余元素）**：

```
for i = vectorNum to currentNum:
    if output[i] < 0:
        factor = output[i] + λ·α    // 负值分支
    else:
        factor = λ                    // 正值分支
    y[i] = grad[i] × factor
```

标量路径中，对 half 计算类型做额外精度保护：`SCALE_ALPHA_PRODUCT` 先截断为 half 再加，以保证输出与 TBE 原算子 bit-exact。

**类型转换（NeedCast=true 时）**：

```
向量化路径执行前:
  Cast(gradCompute, gradRaw, CAST_NONE, vectorNum)      // RawT → ComputeT
  Cast(outCompute, outRaw, CAST_NONE, vectorNum)

向量化路径执行后（NeedCast=true）:
  Cast(yRaw, yCompute, CAST_RINT, vectorNum)              // ComputeT → RawT（四舍五入）
```

#### 3.4.5 CopyOut 阶段

```
DataCopyPad(yGm[progress], yLocal, {copyBytes, ...})
```

---

### 3.5 API 映射

| 计算步骤 | Ascend C API | 参数签名 | 平台验证 | 约束说明 |
|---------|-------------|---------|---------|---------|
| 数据搬入 | DataCopyPad\<RawT\> | (dst, src, DataCopyExtParams{1,copyBytes,0,0,0}, DataCopyPadExtParams\<RawT\>{hasPad,0,padVal,0}) | ✅ DAV_2201 | copyBytes 需 ≥ 32 字节时对齐 |
| 类型转换 | Cast\<ComputeT, RawT\> | (dst, src, RoundMode::CAST_NONE, currentNum) | ✅ DAV_2201 | — |
| 比较 | CompareScalar\<ComputeT\> | (mask, src, scalar, CMPMODE::LT, vectorNum) | ✅ DAV_2201 | vectorNum 需为 256/sizeof(ComputeT) 的倍数 |
| 加法 | Adds\<ComputeT\> | (dst, src, scalar, vectorNum) | ✅ DAV_2201 | — |
| 选择 | Select\<ComputeT\> | (dst, mask, src1, scalar, SELMODE::VSEL_TENSOR_SCALAR_MODE, vectorNum) | ✅ DAV_2201 | mask 为 bit-packed uint8_t |
| 乘法 | Mul\<ComputeT\> | (dst, src1, src2, vectorNum) | ✅ DAV_2201 | — |
| 类型转换（回写） | Cast\<RawT, ComputeT\> | (dst, src, RoundMode::CAST_RINT, currentNum) | ✅ DAV_2201 | CAST_RINT 四舍五入 |
| 数据搬出 | DataCopyPad\<RawT\> | (dst, src, DataCopyExtParams{1,copyBytes,0,0,0}) | ✅ DAV_2201 | — |

### 3.6 内存管理

| 内存区域 | 大小计算 | 说明 |
|---------|---------|------|
| **gradients 队列** | `ubLen × sizeof(RawT)` × `bufferNum` | 双缓冲时 bufferNum=2，单缓冲=1 |
| **outputs 队列** | `ubLen × sizeof(RawT)` × `bufferNum` | 同上 |
| **y 队列** | `ubLen × sizeof(RawT)` × `bufferNum` | 同上 |
| **gradientsComputeBuf** | `ubLen × sizeof(ComputeT)` | NeedCast=true 时分配 |
| **outputsComputeBuf** | `ubLen × sizeof(ComputeT)` | NeedCast=true 时分配 |
| **yComputeBuf** | `ubLen × sizeof(ComputeT)` | NeedCast=true 时分配 |
| **tmpBuf** | `ubLen × sizeof(ComputeT)` | 梯度系数临时区 |
| **maskBuf** | `ubLen × sizeof(uint8_t)` | Compare 结果位图 |

### 3.7 UB 容量验证

| 平台 | 总 UB | 可使用 UB | 系统保留 |
|------|-------|----------|---------|
| DAV_2201 (Ascend910B/Ascend910_93) | 192 KB | ≤ 184 KB | 8 KB |

**Buffer 总和验证**（以 FP32 为例，双缓冲）：

```
gradientsQueue:   ubLen × 4 × 2 = ubLen × 8
outputsQueue:     ubLen × 4 × 2 = ubLen × 8
yQueue:           ubLen × 4 × 2 = ubLen × 8
tmpBuf:           ubLen × 4     = ubLen × 4
maskBuf:          ubLen × 1     = ubLen × 1
                 ────────────────────────
                      总计      = ubLen × 29 bytes
```

Tiling 中 `ubBytesPerElement = 29`（FP32），`ubFactor = floor(可用UB / 29)`，保证 `ubFactor × 29 ≤ 可用UB`。✅

---

## 4. 性能优化

### 4.1 并行策略

- **多核并行**：Host 侧 Tiling 按数据量自适应分配到最多物理核数
- **向量化计算**：主计算路径使用 256 字节宽度的向量 API（CompareScalar + Adds + Select + Mul），单指令多数据
- **标量尾块**：末尾不足向量宽度的元素逐个计算，保证任意尺寸输入的完整覆盖

### 4.2 流水线设计

- **双缓冲**：当 `blockLength > ubLength` 时，`InitBuffer(..., 2, ...)` 开启双缓冲，MTE 数据搬运与 Vector 计算并行
- **单缓冲收敛**：小 Tensor 时 `bufferNum=1`，简化流水线，减少初始化开销

---

## 5. 风险评估

### 5.1 API 风险

| 风险 | 影响 | 应对措施 |
|-----|------|---------|
| `CompareScalar` + `Select` 组合不支持某些数据类型 | 向量化路径不可用 | 标量尾块内的 `for` 循环可作为降级路径；`GetValue/SetValue` 支持所有类型 |
| `DataCopyPad` 在极小 Tensor 时可能有对齐问题 | 搬运数据错误 | 32 字节对齐填充 + rightPadding 清除未使用空间 |

### 5.2 精度风险

| 风险 | 影响 | 应对措施 |
|-----|------|---------|
| half 精度计算中 `SCALE_ALPHA_PRODUCT` 截断误差 | 与 TBE 结果不一致 | 标量路径中对 half 做双截断对齐：`static_cast<half>(output + float(half(λ·α)))` |
| `Cast(..., CAST_RINT, ...)` 四舍五入行为 | 整数类型回写偏差 | CAST_RINT 是 AscendC 标准四舍五入模式，与 TBE 一致 |

### 5.3 应对措施

- 标量尾块作为向量化路径的安全降级
- half 精度标量路径的截断对齐保证与 TBE bit-exact
- `NeedCast=false` 时零拷贝直接计算，消除类型转换精度损失

---

## 6. 交付件清单

**必需**：`op_host/selu_grad_def.cpp`, `op_host/selu_grad_tiling.cpp`, `op_host/selu_grad_infershape.cpp`, `op_kernel/selu_grad.cpp`, `op_kernel/selu_grad.h`, `op_kernel/selu_grad_tiling_data.h`, `op_kernel/selu_grad_tiling_key.h`, `tests/`, `CMakeLists.txt`

---

## 7. 迭代规划

| 迭代 | 目标 | 代码开发 | UT 开发 | ST 用例 |
|------|------|---------|--------|--------|
| 迭代一 | 骨架搭建 | 单 TilingKey + 预埋骨架 + FP16 dtype | 核心路径用例 | L0 标准用例（基础 shape + FP16） |
| 迭代二 | 策略完善 | Tiling 自适应核数 + ubFactor 动态计算 | Tiling 分支覆盖 | 多 shape 用例（多 dtype） |
| 迭代三 | 规格完整 | 全 6 dtype + 标量尾块 + NeedCast 转换路径 | 全覆盖用例 | 全 dtype + 边界 + 泛化测试 |
