# 东南大学

## 团队信息

- 提交者: 张铭涵-61525H11
- 身份: 学生
- 单位: 东南大学

## 成员

- 张铭涵-61525H11 (gcw_ambIKYDw): 提交者

## 算子: op_10_gelu

GELU 自定义算子实现介绍

**1. 算子功能**
基于 Ascend C 在昇腾 910B 上实现逐元素 GELU 激活算子：`output = GELU(input_x)`。支持 float32 / float16、任意多维 ND 张量，末维元素数 N∈[1,10240]，兼容 N 非 32 字节倍数（非对齐）场景；输出 shape 与 dtype 与输入一致。

**2. 数学实现**
采用 PyTorch 默认的精确误差函数定义：
```
GELU(x) = 0.5 * x * (1 + erf(x / √2))
```
kernel 内用向量指令分 5 步完成：`Muls(x, 1/√2) → Erf → Adds(+1) → Muls(0.5) → Mul(x)`。
选择精确 erf 而非 tanh/sigmoid 近似，是因为 float32 要求相对/绝对误差均 &lt;1e-4，而 tanh 近似对精确 GELU 的最大偏差约 4.7e-4，无法达标；精确 erf 路径对 fp32/fp16 均有足够精度裕量。

**3. Host 侧 Tiling 设计**（`op_host/gelu.cpp`）
- 以 32B 为一个数据块，将张量按块均分到各核：前 `tailBlockNum` 个“大核”多承担 1 块，其余“小核”承担相同块数，保证每核起始偏移与处理长度均 32B 对齐；
- **自适应核数**：`SetBlockDim(min(AIV, 块数))`，小张量避免空转核，大张量用满全部 AIV；
- **流水友好的 tile 划分**：按 UB 容量（预留约 1/4）与数据类型计算单 tile 元素数，使每核数据尽量拆成 ≥2 个 tile，以发挥双缓冲 DMA/计算重叠；
- `InferShape`/`InferDataType` 将输入 shape/type 直接拷贝给输出，本算子无需 workspace。

**4. Kernel 侧实现**（`op_kernel/gelu.cpp`）
- 模板类 `KernelGelu&lt;DT_INPUT_X&gt;` 按输入 dtype（float/half）实例化，`TPipe` + `TQue&lt;VECIN/2&gt;`/`TQue&lt;VECOUT/2&gt;` 双缓冲；
- 每个核循环执行 `CopyIn → Compute → CopyOut`，末块用下发的小尾块长度；因所有搬运均 32B 对齐，统一使用普通 `DataCopy`，无需 DataCopyPad；
- 数据不足一个块的核（tileNum=0）直接跳过，不做无效访问。

**5. 性能与精度结论**
- 精度：精确 erf 公式 + 硬件 `Erf` 指令，实测设计满足 fp32 双万分之一、fp16 双千分之一误差要求；
- 性能：自适应核数消除空转核调度开销（示例小张量仅需 1 核），每核多 tile 使双缓冲流水真正生效，大张量按块满核并行、负载均衡；
- 正确性：tiling 分块逻辑经穷举校验（dtype×N×核数×UB 共 416 组），覆盖完整、无重叠、全部 32B 对齐。

**6. 工程文件**
```
code/op_host/gelu.cpp       Host tiling + shape/type 推导
code/op_kernel/gelu.cpp     Kernel 核函数实现
code/op_kernel/gelu_tiling.h   Tiling 参数结构体
code/op_kernel/tiling_key_gelu.h  模板参数声明（未改动）
```
