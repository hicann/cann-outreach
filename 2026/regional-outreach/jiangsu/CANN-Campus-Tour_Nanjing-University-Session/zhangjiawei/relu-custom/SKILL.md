---
name: ascendc-relu-operator
description: Ascend C ReLU 算子完整实现（float16）。包含算子定义、Tiling、Kernel、ACLNN 接口、构建脚本和测试用例。ReLU(x) = max(0, x)，支持 4 维任意 shape，Format=ND。
---

# Ascend C ReLU 算子（float16）

线性整流（Rectified Linear Unit）一元逐元素算子。

## 规格说明

| 项目 | 内容 |
|------|------|
| **算子类型** | Elementwise（一元） |
| **输入** | x (float16, ND格式) |
| **输出** | y (float16, 与输入 shape 相同) |
| **Shape** | 4 维 [N4,N3,N2,N1]（任意维度均可） |
| **计算公式** | ReLU(x) = max(0, x) |
| **芯片架构** | Ascend910B (arch22) / Ascend950 (arch35) |
| **精度策略** | FP16 直算，无需升精度 |

## 设计要点

### 1. 纯 FP16 计算

ReLU 仅是取最大值运算（比较符号位），**无精度损失风险**，直接在 FP16 域完成。

### 2. Kernel 实现

```
Duplicate(zeros, 0, count)   // 填充零 Buffer
Max(y, x, zeros, count)      // y[i] = max(x[i], 0)
```

### 3. Tiling

与 abs 算子结构相同（纯 FP16、一元、单输入队列），但额外需要 1 个 TBuf 存放 zeros 常量。

| 项目 | Abs | ReLU |
|------|-----|------|
| InQueue | 1 | 1 |
| OutQueue | 1 | 1 |
| 额外 TBuf | 0 | 1 (zeros) |
| bufferDivisor（单缓冲） | 2×2 = 4 | **3×2 = 6** |
| bufferDivisor（双缓冲） | 4×2 = 8 | **5×2 = 10** |

### 4. 4 维 Shape 适配

| Shape | 元素数 | 核数 | 缓冲模式 |
|-------|--------|------|---------|
| [1,16,16,16] | 4096 | 1-2 | 单缓冲 |
| [4,32,32,32] | 131072 | 多核 | 双缓冲 |
| [8,64,64,64] | 2097152 | 满核 | 双缓冲 |

## 文件结构

```
references/relu_example/
├── build.sh / CMakeLists.txt
├── op_host/
│   ├── relu_def.cpp                    # 算子定义 (float16)
│   ├── relu_infershape.cpp
│   └── arch22/relu_tiling.cpp
├── op_kernel/
│   ├── relu_arch22.cpp                 # Kernel 入口 (arch22)
│   ├── relu_arch35.cpp                 # Kernel 入口 (arch35)
│   └── arch22/
│       ├── relu.h                      # Max + Duplicate 实现
│       ├── relu_tiling_data.h
│       └── relu_tiling_key.h
├── op_api/ (L0 + L2 API)
├── op_graph/relu_proto.h
└── examples/
    ├── test_aclnn_relu.cpp             # 测试（含负值、零值、正值边界）
    └── run.sh
```

## 使用方式

```bash
cd references/relu_example/
bash build.sh
bash examples/run.sh
```

## 相关技能

- `ascendc-registry-invoke-template` — 自定义算子工程模板
- `ascendc-tiling-design` — Tiling 设计指南
- `ascendc-api-best-practices` — API 使用最佳实践
