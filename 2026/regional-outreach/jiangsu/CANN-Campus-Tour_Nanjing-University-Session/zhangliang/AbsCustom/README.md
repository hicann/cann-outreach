# abs_ascend — Ascend C Abs Operator (float16)

## 概述

一个基于 **Ascend C** 的绝对值（abs）算子，支持 **float16** 数据类型，运行在 **Ascend 910B3** NPU 上。

### 支持的 Shape

| Shape | 元素数 | 典型场景 |
|-------|--------|----------|
| `[1, 128]` | 128 | MLP 中间激活 |
| `[4, 2048]` | 8,192 | Batch 推理 |
| `[32, 4096]` | 131,072 | 大 batch 训练 |

## 项目结构

```
abs_ascend/
├── CMakeLists.txt              # 顶层构建
├── build.sh                    # 构建脚本
├── op_kernel/
│   ├── CMakeLists.txt
│   └── abs_custom.cpp          # Ascend C kernel (y = |x|)
├── op_host/
│   ├── CMakeLists.txt
│   └── abs_custom.cpp          # 算子注册 + tiling 策略
├── scripts/
│   ├── test_abs.py             # Python 单元测试
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
python3 scripts/test_abs.py --op-dir build/output --device 0
```

测试内容包括：
- 三种 shape 的正确性验证
- 与 NumPy `np.abs()` 参考实现对比
- 输出非负性检查
- Max diff 精度报告

## 算子说明

`y = abs(x)` 逐元素取绝对值，输入输出均为 `float16`。

### Tiling 策略

- 数据按 Block 数均匀切分
- 每个 Block 内部以 Tile（2048 元素）为单位循环处理
- 使用 Double Buffer 提升带宽利用率
- Tile 大小对齐到 32B (16 half)，适配向量指令

### 多核并行

Block 数由 Tiling 函数决定（最大 8 核），数据均匀分布到各核。
