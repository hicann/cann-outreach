# ReLU 算子 — Ascend C 直调实现（float16, 4D ND）

`relu(x) = max(0, x)` 的 Ascend C 实现。

## 规格

| 项目 | 值 |
|------|-----|
| 算子 | ReLU（线性整流单元） |
| 输入 | 1 个 Tensor, float16, ND Format |
| 输出 | 1 个 Tensor, float16 |
| Shape | 任意 4D ND [N4,N3,N2,N1] |
| 定义域 | 全体实数 |

## 文件结构

```
├── op_kernel/
│   ├── relu_custom_tiling.h       Tiling 常量 + 结构体
│   └── relu_custom_kernel.asc     纯 Kernel 代码（AscendC::Relu）
├── op_host/
│   ├── relu_custom.asc            Host + main（支持命令行传 4D shape）
│   └── data_utils.h
├── op_extension/
│   ├── relu_custom_torch.cpp      PyTorch host 实现
│   ├── register.cpp               TORCH_LIBRARY 注册
│   └── ops.h
├── scripts/
│   ├── gen_data.py                测试数据生成（正负混合）
│   ├── golden.py                  golden = np.maximum(0, x)
│   ├── verify_result.py           精度验证（rtol=1e-4, atol=1e-6）
│   └── test_torch.py              9 个测试用例覆盖多种 4D shape + 边界
├── CMakeLists.txt
├── run.sh
└── README.md
```

## 快速开始

```bash
source ${ASCEND_HOME_PATH}/set_env.sh
cd relu_custom

# 默认 shape [8,16,128,256] 完整编译运行
bash run.sh

# 指定 4D shape
SHAPE_N1=16 SHAPE_N2=32 SHAPE_N3=64 SHAPE_N4=128 bash run.sh

# PyTorch 通路
bash run.sh --torch
```

PyTorch 调用：

```python
import torch
import torch_npu

torch.ops.load_library("build/librelu_custom_ops.so")
x = torch.randn(4, 16, 64, 128, dtype=torch.float16).npu()
y = torch.ops.npu.relu_custom(x)
```

## 算子设计

- **Kernel**: `AscendC::Relu` 内置向量运算，half 数据类型直算（无需精度提升）
- **UB**: 2 个双 Buffer 队列（输入 + 输出），单 Buffer 16KB
- **Tiling**: 根据设备核数自动分核
