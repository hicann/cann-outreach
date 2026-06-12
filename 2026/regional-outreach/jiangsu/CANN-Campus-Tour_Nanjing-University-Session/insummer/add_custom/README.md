# add_ascend — Ascend C Add Operator (float16)

## 概述

一个基于 **Ascend C** 的逐元素加法（add）算子，支持 **float16** 数据类型，运行在 **Ascend 910B3** NPU 上。

- **算子名称**: `add_custom`
- **数学公式**: `z = x + y`（逐元素）
- **输入**: `x (float16, ND)`, `y (float16, ND)`
- **输出**: `z (float16, ND)`
- **数据格式**: ND

## 支持的 Shape

| Shape | 元素数 | 说明 |
|-------|--------|------|
| `[1, 128]` | 128 | 单样本小向量 |
| `[4, 2048]` | 8,192 | 小 batch |
| `[32, 4096]` | 131,072 | 标准训练 batch |

## 项目结构

```
add_ascend/
├── CMakeLists.txt              # 顶层构建
├── build.sh                    # 构建脚本
├── op_kernel/
│   ├── CMakeLists.txt
│   └── add_custom.cpp          # Ascend C kernel (z = x + y)
├── op_host/
│   ├── CMakeLists.txt
│   └── add_custom.cpp          # 算子注册 + tiling 策略
├── scripts/
│   ├── test_add.py             # Python 单元测试
│   └── run_test.sh             # 构建 + 测试一键脚本
└── README.md
```

## 构建要求

- **硬件**: Ascend 910B3（或其它 7 系列 NPU）
- **软件**: CANN >= 7.0（Ascend C Toolkit）

## 构建 & 测试

```bash
# 1. Source Ascend 环境
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 2. 构建
bash build.sh Release

# 3. 运行单元测试
python3 scripts/test_add.py --op-dir build/output --device 0
```

## 算子实现细节

### Kernel 核心

```cpp
// 从 GM 加载两个输入
DataCopy(xLocal, xGm[i * tileLen], curTile);
DataCopy(yLocal, yGm[i * tileLen], curTile);

// 逐元素加法：xLocal = xLocal + yLocal
Add(xLocal, xLocal, yLocal, curTile);

// 结果写回 GM
DataCopy(zGm[i * tileLen], zLocal, curTile);
```

- 使用 `AscendC::Add` 向量指令进行逐元素加法
- 一次加载两个输入，原地修改 xLocal 作为结果，节省 UB 空间

### Tiling 策略

| Shape | 元素数 | Tile | Block 数 |
|-------|--------|------|----------|
| `[1,128]` | 128 | 128 | 1 |
| `[4,2048]` | 8,192 | 2,048 | 4 |
| `[32,4096]` | 131,072 | 2,048 | 8 |

### 内存模型

```
GM (HBM)          ┌──────┐ ┌──────┐ ┌──────┐
                  │  x   │ │  y   │ │  z   │
                  └──┬───┘ └──┬───┘ └──▲───┘
                     │        │         │
UB (Local Memory)   ▼        ▼         │
                  ┌──────────────┐      │
                  │  xLocal  │   │      │
                  │  yLocal  │   │      │
                  │  Add()   ├───┘      │
                  │  zLocal ─┼──────────┘
                  └──────────────┘
```

### 多核并行

数据均匀分布到多个 AICore（最多 8 核），每个核处理 `totalLength / blockNum` 个元素。
