# softmax_ascend — Ascend C Softmax Operator (float32)

## 概述

一个基于 **Ascend C** 的 softmax 算子，支持 **float32** 数据类型，
Format=ND，沿最内层轴（axis=-1）进行 softmax 计算。

### 数学公式

```
softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
```

其中 `max(x)` 和 `sum(exp(...))` 沿 axis=-1 计算。

### 支持的 Shape

| Shape | 行数 (rows) | 每行元素 | 总元素 |
|-------|-------------|---------|--------|
| `[128,64,32]` | 8,192 | 32 | 262,144 |
| `[4,8,32]` | 32 | 32 | 1,024 |
| `[16,32,32]` | 512 | 32 | 16,384 |

## 项目结构

```
softmax_ascend/
├── CMakeLists.txt                  # 顶层构建
├── build.sh                        # 构建脚本
├── op_kernel/
│   ├── CMakeLists.txt
│   └── softmax_custom.cpp          # Ascend C kernel
├── op_host/
│   ├── CMakeLists.txt
│   └── softmax_custom.cpp          # 算子注册 + tiling
├── scripts/
│   ├── test_softmax.py             # Python 单元测试
│   └── run_test.sh                 # 一键构建+测试
└── README.md
```

## 构建与测试

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
bash build.sh Release
python3 scripts/test_softmax.py --op-dir build/output --device 0
```

## 算子说明

### 计算流程

对于每行（32 个 float32 元素）：
1. `max_val = row.max()` — 找最大值
2. `row = row - max_val` — 减去最大值（数值稳定）
3. `row = exp(row)` — 逐元素指数
4. `sum_val = row.sum()` — 求和
5. `row = row / sum_val` — 归一化

### 多核并行

- 数据按行均匀分配到各 AICore（最多 8 核）
- 每 Tile 处理 8 行（256 个 float32 = 1KB）
- Double Buffer 隐藏 GM↔UB 传输延迟
