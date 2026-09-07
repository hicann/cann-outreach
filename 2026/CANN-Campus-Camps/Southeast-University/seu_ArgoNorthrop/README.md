# 东南大学

## 团队信息

- 提交者: 程康博61525I07
- 身份: 学生
- 单位: 东南大学

## 成员

- 程康博61525I07 (ArgoNorthrop): 提交者

## 算子: op_10_gelu

# GELU 算子实现简介（Ascend C 开源仓模板）

&gt; 作者：程康博 · 东南大学 · AscendC 算子赛题 Problem #1747

---

## 一、算子核心功能

GELU（高斯误差线性单元）激活算子，对输入张量每个元素计算：

$$ \text{GELU}(x) = x \cdot 0.5 \cdot (1 + \text{erf}(x/\sqrt{2})) = x \cdot \Phi(x) $$

支持 `float32`（精确计算，误差 ≤ 1e-7）、`float16`（近似计算，误差 ≤ 3e-4，满足题目 1e-3 精度要求）。输入输出形状、数据类型一致，兼容任意多维 + 非 32B 对齐场景。

**参考实现**：`torch.nn.functional.gelu`（PyTorch）

---

## 二、整体实现架构

严格遵循 Ascend C 开源仓规范，分为 **Host 侧（算子描述 / tiling / 形参推导）** + **Kernel 侧（核函数 / 计算流水线）** 两层，共 5 个核心文件：

| 层级 | 文件 | 职责 |
|------|------|------|
| Host | `op_host/gelu.cpp` | 算子注册 + TilingFunc（资源分配 / 对齐 / 核数计算）+ InferShape + InferDataType |
| Host | `op_host/gelu_tiling.cpp` | Tiling 实现（对齐 / 软除法消除）|
| Kernel | `op_kernel/gelu.cpp` | Kernel 类实现（双队列流水线 + dtype 特化计算）|
| Kernel | `op_kernel/gelu_tiling.h` | Tiling 结构体定义（`length` / `perCore` 字段）|
| Kernel | `op_kernel/tiling_key_gelu.h` | Tiling 密钥定义（fp16 / fp32 模板分发）|

---

## 三、Host 侧实现（零软除法设计）

### 1. 总元素数计算

从 `TilingContext::GetInputShape(0)` 取 `StorageShape`，遍历所有维度相乘得到 `totalNum`，空维度（标量）特殊处理为 1。

### 2. 32B 对齐系数推导

根据 dtype 推导每 32B 对齐的元素数：

```cpp
int64_t dtypeSize = ge::GetSizeByDataType(dtype);  // fp32 = 4, fp16 = 2
int64_t alignElements = 32 / dtypeSize;            // fp32 = 8, fp16 = 16（均为 2 的幂）
```

### 3. 核级 tiling 预计算（核心优化）

**Host 侧提前完成所有对齐和除法，Kernel 侧全程零软除法**（AscendC AIV 核无硬件除法器，软除法开销 100+ cycle/次）：

```cpp
int64_t perCoreRaw = CeilDiv(totalNum, coreNum);
int64_t perCore = CeilAlign(perCoreRaw, alignElements);  // 对齐到 2 的幂
tiling-&gt;perCore = perCore;
context-&gt;SetBlockDim(CeilDiv(totalNum, perCore));  // 核数动态计算
```

### 4. UB 块大小

固定 `ubFactor = 2048`（对齐到 `alignElements`），Kernel 侧 tile 长度用移位消除软除法。

---

## 四、Kernel 侧实现（双队列流水线 + dtype 特化）

### 1. 双队列流水线架构

采用 AscendC 官方推荐的 `VECIN → VECOUT` 双队列流水（适配 MTE2 / MTE3 / V 三段硬件流水线的重叠）：

```cpp
static constexpr uint32_t BUF_DEPTH = 2;  // 2 深流水（实验验证 3/4 深负收益）
AscendC::TQue&lt;TPosition::VECIN,  BUF_DEPTH&gt; queueX_;
AscendC::TQue&lt;TPosition::VECOUT, BUF_DEPTH&gt; queueY_;
```

### 2. 计算流水线四步走

- **Init**：初始化 Pipe / Buf，读取 Host 侧预计算的 `perCore`，计算 `coreLen`（对齐到 `alignElements`，Kernel 侧无软除法）
- **CopyIn**：MTE2 从 GM 搬入 `queueX_`（tile 长度 4096，对齐后天然安全）
- **Compute**：从 `queueX_` 取出输入，**dtype 特化计算**后写入 `queueY_`
- **CopyOut**：MTE3 从 `queueY_` 搬出到 GM

### 3. dtype 特化计算路径（核心精度 / 性能平衡）

| dtype | 计算路径 | 步数 | 精度 | 选择理由 |
|-------|----------|------|------|----------|
| fp32 | **精确 Erf 版** | 5 | ≤ 1e-7 | 满足题目 1e-4 双万分之一要求；Tanh 近似误差 3e-4 过不了 |
| fp16 | **Tanh 近似版** | 8 | ≤ 3e-4 | 满足题目 1e-3 双千分之一要求；比 cast-fp32 版快 40%，比 half-Erf 直算快 |

#### fp32 精确版公式（AscendC 实现）

```cpp
AscendC::Muls(t, xLocal, 0.70710678f, n);  // t = x / √2
AscendC::Erf(e, t, n);                     // e = erf(t)
AscendC::Adds(e, e, 1.0f, n);              // e = 1 + erf(t)
AscendC::Muls(e, e, 0.5f, n);              // e = 0.5 * (1 + erf(t))
AscendC::Mul(yLocal, xLocal, e, n);        // y = x * e
```

#### fp16 Tanh 近似版公式（数学等价变换）

利用 GELU 的平滑近似（Hendrycks 2016）：

$$ \text{GELU}(x) \approx 0.5x \left(1 + \tanh\left(0.79788x + 0.035677x^3\right)\right) $$

最大绝对误差 3e-4，满足 fp16 精度要求：

```cpp
AscendC::Mul(t, xLocal, xLocal, n);        // t = x²
AscendC::Muls(t, t, 0.035677f, n);         // t = 0.035677x²
AscendC::Adds(t, t, 0.79788f, n);          // t = 0.79788 + 0.035677x²
AscendC::Mul(t, t, xLocal, n);             // t = 0.79788x + 0.035677x³
AscendC::Tanh(e, t, n);                    // e = tanh(t)
AscendC::Adds(e, e, 1.0f, n);              // e = 1 + tanh
AscendC::Mul(yLocal, xLocal, e, n);        // y = x * (1 + tanh)
AscendC::Muls(yLocal, yLocal, 0.5f, n);    // y = 0.5 * x * (1 + tanh)
```

### 4. UB 布局优化

- **tmp buffer 共用**：计算过程中 `t` 和 `e` 共用一个 `TBuf` 的前后半（`bufTmp_.Get&lt;DT&gt;()[0:TILE]` 和 `[TILE:2*TILE]`），省一次 `InitBuffer` 固定开销
- **UB 占用控制**：fp32 路径 80KB（输入输出各 32KB + tmp 16KB），fp16 路径 40KB（输入输出各 16KB + tmp 8KB），远小于 192KB UB 上限

---

## 五、关键优化与实验验证

### 已验证有效（核心贡献）

| 优化点 | 效果 | 证据 |
|--------|------|------|
| Host 侧对齐 + Kernel 零软除法 | 消灭 3 次软除法 / 核 | Square 题 Kernel 侧对齐踩坑后，优化 T5 快 10μs |
| fp16 Tanh 近似 | 比 cast-fp32 版快 40% | T5 从 24μs 降到 16.8μs |
| TILE_LENGTH = 4096 | DMA 粒度最优 | 比 2048 快 2.9μs，比 8192 快 1μs |
| BUF_DEPTH = 2 深流水 | 最优重叠 | 3/4 深流水实测负收益 |

### 已证伪（避免重复踩坑）

| 尝试 | 结果 | 原因 |
|------|------|------|
| BN3 / BN4 深流水 | T5 慢 2.9μs | 深队列同步链变长，重叠收益抵消 |
| fp16 cast-fp32 版 | T5 慢 7μs | Cast 开销 + fp32 计算比 half Tanh 还慢 |
| Sigmoid 近似版 | 比 Tanh 慢 2 步 | 数学等价但 AscendC 指令序列更长 |
| dtype 特化 TILE（fp16 = 2048） | 比 4096 慢 | tile 数翻倍 → API 调度开销翻倍 |
| 单队列直写（砍 VECOUT） | 竞态 WA + 编译不过 | 手写 SetFlag / WaitFlag 同步在 CANN 9.0 编译不支持 |
| 类内依赖 constexpr 作模板实参 | Compile Error | `static constexpr uint32_t BUF_DEPTH = (half ? 4 : 2)` 作 `TQue` 模板实参 ccec 不认 |

---

## 六、最终配置与测试结果

### 最终稳定配置

| 参数 | fp32 | fp16 |
|------|------|------|
| TILE_LENGTH | 4096 | 4096 |
| BUF_DEPTH | 2 | 2 |
| 计算路径 | Erf 精确版 | Tanh 近似版 |
| 对齐方式 | Host 侧 32B 对齐，Kernel 零软除法 | 同左 |

### 测试结果（最好成绩 67.94 分，全 Pass）

| 测试点 | 用时 | 输出错误占比 | dtype |
|--------|------|--------------|-------|
| 1 | 4.03μs | 0.00% | fp32 |
| 2 | 4.77μs | 0.00% | fp16 |
| 3 | 6.47μs | 0.00% | fp32 |
| 4 | 4.86μs | 0.00% | fp16 |
| 5 | 16.80μs | 0.00% | fp32 + fp16 混合 |

&gt; **注意**：分数在 49–68 区间波动来自评测分母（全场历史最优用时）在快窗口被刷新，同一份代码的绝对用时稳定在上述范围。

---

## 七、踩坑记录（给后续开发者）

1. **ccec 编译器限制**：类内依赖模板参数的 `constexpr`（如 `static constexpr uint32_t BUF_DEPTH = (half ? 4 : 2)`）作 `TQue` 模板实参时编译不过，必须用全局非依赖常量
2. **TQue 天然同步**：VECOUT 队列的 `EnQue / DeQue` 自动处理 V → MTE3 同步，手写 `SetFlag / WaitFlag` 在部分 CANN 版本可能不支持
3. **软除法陷阱**：AscendC AIV 核无硬件除法器，所有 `CeilDiv / CeilAlign` 必须提前在 Host 侧完成，Kernel 侧只能用移位 / 位运算（`&gt;&gt; n`、`&amp; ~((1&lt;&lt;k)-1)`）
4. **half 数学 API 精度**：half Erf/Tanh 等数学函数是逐元素模拟（非向量化多项式），性能远低于 fp32 版——fp16 路径直接用 half 数学 API 反而比 cast-fp32 更慢，这与直觉相反，务必实测
5. **对齐必须两层**：host 侧 perCore 对齐 + kernel 侧 coreLen 再对齐（位运算），缺一不可——Square 题漏了 kernel 侧对齐导致 Wrong Answer 0.31%

---

## 八、参考资料

- Ascend C 官方文档：https://www.hiascend.com/document
- PyTorch GELU：https://pytorch.org/docs/stable/generated/torch.nn.functional.gelu.html
- Hendrycks &amp; Gimpel (2016)：*Bridging Nonlinearities and Stochastic Computers with Gated Gaussian Error Linear Units*
- 同作者其他赛题实现：Sub / Mul / Relu / Square（同模板结构）
