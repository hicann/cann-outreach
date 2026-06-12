# gcd_ascend — Ascend C GCD Operator (float16, 4D ND, Broadcast)

## 概述

一个基于 **Ascend C** 的最大公约数（GCD）算子，支持 **float16** 数据类型，
**4D shape**（如 `[N4, N3, N2, N1]`）、**ND Format** 以及 **Broadcast 语义**，
运行在 **Ascend 910B3** NPU 上。

### 算子接口

| 参数 | 类型 | 说明 |
|------|------|------|
| `self`  (输入) | float16, Tensor | 第一个输入，4D ND |
| `other` (输入) | float16, Tensor | 第二个输入，4D ND |
| `out`   (输出) | float16, Tensor | 输出：`gcd(self, other)`，shape 为 self 和 other 的 broadcast 结果 |

### Broadcast 语义

两个输入的 4D shape 需满足 NumPy broadcast 规则：
- 对应维度相等，或其中一个为 1（该维度会被 broadcast）
- 输出 shape 取各维度的最大值

#### 支持的 Broadcast 场景

| 场景 | self shape | other shape | 输出 shape |
|------|-----------|-------------|------------|
| 同型 | `[1,4,8,16]` | `[1,4,8,16]` | `[1,4,8,16]` |
| broadcast self | `[1,1,1,16]` | `[1,4,8,16]` | `[1,4,8,16]` |
| broadcast other | `[1,4,8,16]` | `[1,1,1,16]` | `[1,4,8,16]` |
| 混合 broadcast | `[1,1,8,1]` | `[1,4,1,16]` | `[1,4,8,16]` |
| 全 broadcast | `[1,1,1,1]` | `[1,4,8,16]` | `[1,4,8,16]` |

## 项目结构

```
gcd_ascend/
├── CMakeLists.txt              # 顶层构建
├── build.sh                    # 构建脚本
├── op_kernel/
│   ├── CMakeLists.txt
│   └── gcd_custom.cpp          # Ascend C kernel (欧几里得算法)
├── op_host/
│   ├── CMakeLists.txt
│   └── gcd_custom.cpp          # 算子注册 + tiling + broadcast stride 计算
├── scripts/
│   ├── test_gcd.py             # Python 单元测试（含多种 broadcast 场景）
│   └── run_test.sh             # 构建 + 测试一键脚本
└── README.md
```

## 构建要求

- **硬件**: Ascend 910B3 (或其它 7 系列 NPU)
- **软件**: CANN >= 7.0 (Ascend C Toolkit)
- 已配置 Ascend 环境变量

## 构建

```bash
# 方法一：直接构建
bash build.sh Release

# 方法二：构建 + 测试
bash scripts/run_test.sh Release
```

构建产物在 `build/output/` 目录。

## 运行单元测试

```bash
# 需要先 source Ascend 环境
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 构建后运行测试
python3 scripts/test_gcd.py --op-dir build/output --device 0
```

测试内容包括：
- 6 种 broadcast 场景的正确性验证
- 与 NumPy 参考实现（Euclidean 算法）对比
- GCD 数学性质验证（结果应整除两个输入）
- Max diff 精度报告

## 算子说明

`out = gcd(self, other)` 逐元素最大公约数计算。

### GCD 算法（Euclidean）

```
gcd(a, b):
    a = |a|, b = |b|
    if a == 0: return b
    if b == 0: return a
    while |b| > eps (1e-4):
        q = floor(a / b)
        r = a - q * b
        if |r| <= eps: break
        a, b = b, r
    return |a|
```

算法在 float16 精度下最多 50 次迭代保证收敛。

### Broadcast 策略

- **Tiling**: 基于输出（broadcast 后）的总元素数进行数据划分
- **Stride 映射**: 每个输出元素通过预计算的 stride 数组映射到 self/other 的对应位置
  - 若某维度 size=1（broadcast 维度），则该维度的 stride=0（始终访问元素 0）
  - 若某维度 size>1，stride=后续维度的乘积
- **访存模式**: 由于 broadcast 导致非连续访存，每个元素使用标量加载；输出为连续写回

### 多核并行

Block 数由 Tiling 函数决定（最大 8 核），输出元素均匀分布到各核。
