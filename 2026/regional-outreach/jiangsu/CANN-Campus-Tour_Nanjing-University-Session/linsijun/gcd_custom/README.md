# Gcd Float16 算子 — Kernel 直调实现

Ascend C `gcd` 算子 Kernel 直调实现，支持 **float16** 数据类型与 **4D Broadcast**。

## 功能

对两个 float16 输入逐元素求最大公约数 (GCD)：

```
for each i: out[i] = gcd(|self[i]|, |other[i]|)
```

支持 NumPy 风格的 **Broadcast 语义**：`self` 和 `other` 的 4D shape 维度不同时，自动广播扩展至兼容形状。

## 算法

向量化固定迭代欧几里得算法 (GCD_MAX_ITER=64)：

```
gcd(a, b):
  1. a = |a|, b = |b|
  2. 确保 a ≥ b (逐元素)
  3. 重复 64 次:
     remainder = a - b × floor(a / b)
     若 b=0 则保持 a 不变
     a, b = b, remainder, 确保 a ≥ b
  4. 输出 a
```

利用 Ascend C Vector API 全向量化：`Abs → Max/Min → Div → Floor → Mul → Sub → Compare → Sel`。

## 支持的 Broadcast 测试用例

| Case | self shape | other shape | out shape | 说明 |
|------|-----------|-------------|-----------|------|
| A | `[1,1,1,128]` | `[1,1,1,128]` | `[1,1,1,128]` | 相同 shape |
| B | `[1,1,1,128]` | `[1,1,4,1]` | `[1,1,4,128]` | N2 维度广播 (4←1) |
| C | `[2,1,1,128]` | `[1,4,1,1]` | `[2,4,1,128]` | N1+N3 维度广播 |

## 文件结构

```
├── op_kernel/
│   ├── gcd_float16_tiling.h     Tiling 常量 + 结构体
│   └── gcd_float16_kernel.asc   纯 kernel 代码 (KernelGcdFloat16 类 + 核函数入口)
├── op_host/
│   ├── gcd_float16.asc          Host + main 入口 (含 Broadcast 扩展逻辑)
│   └── data_utils.h             数据读写工具
├── op_extension/
│   ├── gcd_float16_torch.cpp    PyTorch host 实现 (含 torch broadcast)
│   ├── register.cpp             TORCH_LIBRARY 注册
│   └── ops.h                    函数声明
├── CMakeLists.txt               双 target: 可执行文件 + libgcd_float16_ops.so
├── scripts/
│   ├── gen_data.py              测试数据生成 (多组 broadcast shape)
│   ├── golden.py                参考计算: 欧几里得 gcd
│   ├── verify_result.py         精度验证
│   └── test_torch.py            PyTorch 通路测试 (含 broadcast 场景)
├── run.sh                       一键运行脚本
└── README.md                    本文件
```

## 快速开始

### 前置条件

- CANN 开发套件已安装，`ASCEND_HOME_PATH` 已设置
- `torch` + `torch_npu` (可选，仅 PyTorch 通路需要)

### 方式一：直调验证 (可执行文件)

```bash
source ${ASCEND_HOME_PATH}/set_env.sh
cd gcd_float16

# 完整流程 (编译 + 生成数据 + 运行 + 验证)
bash run.sh

# 跳过编译 (代码审查阶段)
bash run.sh --skip-build
```

### 方式二：PyTorch 调用

```bash
bash run.sh --torch
```

Python:

```python
import torch
import torch_npu

torch.ops.load_library("build/libgcd_float16_ops.so")

# 相同 shape
x = torch.tensor([[48.0, 36.0, 17.0]], dtype=torch.float16).npu()
y = torch.tensor([[12.0, 24.0, 1.0]], dtype=torch.float16).npu()
z = torch.ops.npu.gcd_float16(x, y)  # gcd(48,12)=12, gcd(36,24)=12, gcd(17,1)=1

# Broadcast: self=[2,4], other=[4] → out=[2,4] 
a = torch.tensor([[48, 36, 17, 0]], dtype=torch.float16).npu()
b = torch.tensor([12, 24, 1, 5], dtype=torch.float16).npu()
c = torch.ops.npu.gcd_float16(a, b)
```

## Tiling 参数

| 参数 | 值 | 说明 |
|------|-----|------|
| TILE_LENGTH | 8192 | 单次搬运元素数 (float16) |
| DOUBLE_BUFFER | 2 | 双缓冲 |
| GCD_MAX_ITER | 64 | 欧几里得算法最大迭代 |

## 精度容差

| 通道 | rtol | atol |
|------|------|------|
| 可执行文件 | 1e-3 | 1e-3 |
| PyTorch | 1e-3 | 1e-3 |
