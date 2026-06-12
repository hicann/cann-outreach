# Gcd Custom Kernel 直调工程（FP16, 4D Broadcast）

Gcd（最大公约数）算子的 Ascend C Kernel 直调实现，支持 FP16 数据类型和 4D broadcast。

## 算子说明

| 属性 | 值 |
|------|-----|
| **算子类型** | Elementwise 逐元素二元算子 |
| **计算公式** | `out[i] = gcd(self[i], other[i])` — Euclidean 算法 |
| **数据类型** | FP16 (`half`)，寄存整数（1~65504） |
| **输入** | 2 个 Tensor: `self`, `other` (FP16 ND) |
| **输出** | 1 个 Tensor: `out` (FP16 ND) |
| **Broadcast** | 标准 NumPy 4D broadcast 规则 |
| **Format** | ND |

## 支持的 4D Broadcast 场景

| 用例 | Self Shape | Other Shape | Out Shape | 说明 |
|------|-----------|-------------|-----------|------|
| A | `[2,4,8,16]` | `[2,4,8,16]` | `[2,4,8,16]` | 同 shape |
| B | `[1,4,8,16]` | `[2,4,8,16]` | `[2,4,8,16]` | self 第0维 broadcast |
| C | `[2,1,8,1]` | `[2,4,8,16]` | `[2,4,8,16]` | 双 broadcast |
| D | `[1,1,1,1]` | `[2,4,8,16]` | `[2,4,8,16]` | self 全 broadcast |

## 算法说明

Euclidean 算法逐元素计算 GCD：

```
function gcd(a, b):
    a = |a|, b = |b|        // 取绝对值，GCD 定义为正数
    while b > ε:
        temp = b
        q = floor(a / b)      // 正值时 int 截断 = floor
        b = a - q * b          // fmod 展开: a mod b
        a = temp
    return a
```

中间计算使用 float32 以避免 FP16 精度损失。

## 文件结构

```
├── op_kernel/                   NPU 计算层
│   ├── gcd_custom_tiling.h      Tiling 常量
│   └── gcd_custom_kernel.asc    Kernel 实现（KernelGcd + GcdScalar）
├── op_host/                     Host 层
│   ├── gcd_custom.asc           Host + main（含 BroadcastExpand 逻辑）
│   └── data_utils.h             文件读写工具
├── op_extension/                PyTorch 层
│   ├── gcd_custom_torch.cpp     PyTorch 对接（含 expand_as broadcast）
│   ├── register.cpp             TORCH_LIBRARY 注册
│   └── ops.h                    函数声明
├── scripts/
│   ├── gen_data.py              数据生成（4 种预设 shape 用例）
│   ├── golden.py                Golden 计算（np.gcd + broadcast）
│   ├── verify_result.py         精度验证
│   └── test_torch.py            PyTorch 通路测试（8 个用例）
├── CMakeLists.txt               双 target 构建
├── run.sh                       一键运行
└── README.md
```

## 快速开始

```bash
export ASCEND_HOME_PATH=/path/to/Ascend/latest

# 同 shape 测试
bash run.sh

# Broadcast 测试
bash run.sh B     # self broadcast
bash run.sh C     # 双 broadcast
bash run.sh D     # 全 broadcast

# 自定义 4D shape
bash run.sh 1 2 3 4  5 6 7 8

# 分步执行
mkdir -p build && cd build && cmake .. && make -j && cd ..
python3 scripts/gen_data.py A
./build/gcd_custom  2 4 8 16  2 4 8 16
python3 scripts/verify_result.py output/output.bin output/golden.bin
```

## PyTorch 调用

```python
import torch
import torch_npu

torch.ops.load_library("build/libgcd_custom_ops.so")

self  = torch.randint(1, 256, (2, 4, 8, 16), dtype=torch.float16).npu()
other = torch.randint(1, 256, (2, 4, 8, 16), dtype=torch.float16).npu()
y = torch.ops.npu.gcd_custom(self, other)
```

## Tiling 设计

| 参数 | 值 | 说明 |
|------|-----|------|
| `TILE_LENGTH` | 4096 | 每 tile 元素数（FP16 下 = 8KB/每 buffer） |
| `DOUBLE_BUFFER` | 2 | 双缓冲 |
| UB 占用 | 48KB | 输入x 8KB + 输入y 8KB + 输出z 8KB，双缓冲 48KB |
| 对齐因子 | 512 元素 | 多核切分对齐 |

## 测试覆盖

| 场景 | 值 | 期望 GCD |
|------|-----|---------|
| 常规正数 | 48, 18 | 6 |
| 零值 | 0, 15 | 15 |
| 相等 | 12, 12 | 12 |
| 倍数 | 7, 21 | 7 |
| 互质 | 13, 17 | 1 |
| 正负 | -24, 18 | 6 |
| 负负 | -30, -45 | 15 |
| 含 1 | 997, 1 | 1 |
