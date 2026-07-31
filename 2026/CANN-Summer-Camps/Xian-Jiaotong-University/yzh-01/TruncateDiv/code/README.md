# TruncateDiv 算子

## 概述

TruncateDiv 算子实现逐元素除法后向零取整的功能：

$$y = \text{trunc}\left(\frac{x1}{x2}\right)$$

输入 `x1` 和 `x2` 支持广播（Broadcast），即两个张量形状可以不同，按照 NumPy 广播规则自动对齐。

## 支持规格

| 项目 | 支持 |
|------|------|
| **数据类型** | float16, float32, bfloat16 |
| **数据格式** | FORMAT_ND |
| **芯片平台** | ascend910b, ascend950 |
| **最大维度** | 8 维 |
| **广播** | 支持（维度右对齐广播） |

## 文件结构

```
code/
├── op_host/
│   ├── truncate_div_def.cpp          # 算子定义（OpDef 注册）
│   ├── truncate_div_infershape.cpp   # 输出形状推导（广播规则）
│   └── truncate_div_tiling.cpp       # Tiling 实现（分核 / UB分配 / TilingKey）
├── op_kernel/
│   ├── truncate_div.h                # Kernel 类（TruncateDiv<T> 模板实现）
│   ├── truncate_div.cpp              # Kernel 入口（TilingKey 分派）
│   ├── truncate_div_tiling_key.h     # TilingKey 枚举定义
│   └── truncate_div_tiling_data.h    # TilingData 结构体
└── tests/
    └── ut/
        ├── run.sh                    # 单元测试执行脚本
        ├── op_host/                  # Host 侧 UT（Tiling 验证）
        └── op_kernel/                # Kernel 侧 UT（精度验证）
```

## 设计要点

### Tiling 策略

| 参数 | 值 | 说明 |
|------|-----|------|
| MIN_SPLIT_THRESHOLD | 1024 | 总元素 ≤ 1024 时单核处理 |
| MIN_ELEMS_PER_CORE | 2048 | 多核模式下每核最少处理元素数 |
| 对齐方式 | CeilAlign | 向上对齐，保证覆盖全部元素 |
| 对齐粒度 | 32 字节 | fp16/bf16 对 16 元素对齐，fp32 对 8 元素对齐 |

### TilingKey 分派

| Mode | 数据类型 | 计算路径 |
|------|---------|---------|
| Mode 0 | float16 | Cast→fp32→Div→Trunc→Cast→fp16 |
| Mode 1 | float32 | Div→Trunc（直接路径） |
| Mode 2 | bfloat16 | Cast→fp32→Div→Trunc→Cast→bf16 |

> **说明**：float16 和 bfloat16 均使用 fp32 中间计算路径。原因是 Ascend 910b Vector/Memory 层 `Div` API 仅支持 `half` 和 `float` 类型，不支持 `bfloat16_t`；同时低精度除法结果在整数边界附近可能因舍入产生误差，使用 fp32 中间计算可避免此问题。

### UB 内存分配

- `UB_RESERVE_SIZE = 4096` 字节（框架开销预留）
- TQue 双缓冲：x1(2 slot) + x2(2 slot) + y(2 slot)，实现 CopyIn 与 Compute/CopyOut 重叠流水
- Cast 路径额外 fp32 TBuf：3 个（x1/x2/y 中间结果各一）
- Trunc 临时缓冲区：`ubFactor × sizeof(float) + 256` 字节

### 搬运策略

- 全程使用 `DataCopyPad`：兼容 32 字节对齐和非对齐场景，避免 `DataCopy` 的对齐约束风险
- 同 shape 快速路径：TQue 双缓冲流水线
- 广播路径：分块 gather（MapIndex 地址映射 → 逐元素 DataCopyPad）→ 向量化 compute → 逐元素 scatter

## 构建与测试

### 编译

```bash
cd code
bash build.sh -j8    # 编译算子包
```

### 单元测试

```bash
cd code/tests/ut
bash run.sh           # 运行 Host UT + Kernel UT + 精度比对
```

### 精度标准

| 数据类型 | MERE 阈值 | MARE 阈值 |
|---------|----------|----------|
| float16 | 2^-10 ≈ 0.000977 | 10 × MERE |
| float32 | 2^-13 ≈ 0.000122 | 10 × MERE |
| bfloat16 | 2^-7 ≈ 0.00781 | 10 × MERE |

## 环境要求

- CANN Toolkit 9.0.0+
- NPU 设备：ascend910b 或 ascend950
- 编译依赖：CMake ≥ 3.16, GCC ≥ 9.0
- Python 测试依赖：numpy, ml_dtypes
