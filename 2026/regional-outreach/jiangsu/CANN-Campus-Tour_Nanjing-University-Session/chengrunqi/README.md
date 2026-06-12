# ReLU 算子 - AscendC 直调工程

## 概述

AscendC 实现的 ReLU（线性整流单元）算子，支持 **float16** 数据类型、**ND** 格式。

**公式**: `ReLU(x) = max(0, x)`

## 支持的 Shape

| Shape | 元素数 | 说明 |
|-------|--------|------|
| `[1, 1, 1, 128]` | 128 | 小 4D shape |
| `[1, 4, 32, 64]` | 8192 | 中 4D shape |
| `[8, 16, 32, 32]` | 131072 | 大 4D shape（多核并行） |

## 项目结构

```
├── op_kernel/                 NPU 计算层
│   ├── relu_custom_tiling.h    Tiling 常量 + 结构体
│   └── relu_custom_kernel.asc  Kernel 代码（KernelRelu 类 + 核函数入口）
├── op_host/                   Host 直调层
│   ├── relu_custom.asc         Host + main 入口（批量运行 3 个 4D Shape）
│   └── data_utils.h            文件读写工具
├── op_extension/              PyTorch 接入层
│   ├── relu_custom_torch.cpp   PyTorch 实现
│   ├── register.cpp             TORCH_LIBRARY 注册
│   └── ops.h                    函数声明
├── scripts/                   测试脚本
│   ├── gen_data.py             生成 FP16 输入数据（3 个 4D Shape）
│   ├── golden.py               Golden 计算（np.maximum）
│   ├── verify_result.py        精度验证
│   └── test_torch.py           PyTorch 通路测试（5 种场景 × 3 Shape）
├── CMakeLists.txt             构建配置
├── run.sh                     一键运行
└── README.md
```

## 使用方法

```bash
# 完整流程（编译 + 运行 + 验证）
bash run.sh

# 仅运行，复用编译产物
bash run.sh --skip-build

# 仅运行 PyTorch 通路
bash run.sh --torch

# PyTorch 单次调用
python3 -c "
import torch; import torch_npu
torch.ops.load_library('build/librelu_custom_ops.so')
x = torch.tensor([-1.0, 0.0, 2.0], dtype=torch.float16).npu()
y = torch.ops.npu.relu_custom(x)
print(y.cpu())
"
```

## 精度标准

- `rtol = 1e-3`, `atol = 1e-4`（FP16 合理精度范围）

## 技术要点

- **计算逻辑**: `AscendC::Relu(zLocal, xLocal, count)` — AscendC 内置 API
- **TILE_LENGTH**: 8192（FP16 优化值）
- **Double Buffer**: 输入输出各 2 份，流水线并行
- **多核切分**: 按元素数在核间均匀分配
- **4D 支持**: 展平为一维处理，任意 4D Shape 均可
- **PyTorch 调用**: `torch.ops.npu.relu_custom(x)`
