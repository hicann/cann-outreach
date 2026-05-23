# sparse_gemm 算子

## 概述

sparse_gemm 是一个稀疏矩阵乘法算子，支持 2:4 稀疏格式。该算子基于 Ascend C 实现，使用基础 API 路线（LoadDataWithSparse + MmadWithSparse）。

## 功能特性

- 支持 2:4 稀疏格式（每 4 个元素中最多 2 个非零元素）
- 支持 batch 维度
- 支持多核并行（M×N 二维切分）
- 支持 int8_t 输入，float32 输出

## 文件结构

```
sparse_gemm/
├── kernel/
│   ├── sparse_gemm_kernel.asc      # Kernel 计算逻辑
│   └── sparse_gemm_tiling.h        # Tiling 结构体定义
├── host/
│   ├── sparse_gemm.asc             # Host + main 入口
│   └── data_utils.h                # 数据读写工具
├── scripts/
│   ├── gen_data.py                 # 生成测试数据
│   └── verify_result.py            # 验证结果
├── CMakeLists.txt                  # 编译配置
├── run.sh                          # 运行脚本
└── README.md                       # 本文档
```

## 编译

```bash
# 创建构建目录
mkdir -p build
cd build

# 运行 CMake
cmake ..

# 编译
make
```

## 运行测试

### 生成测试数据

```bash
python3 scripts/gen_data.py --batch 2 --m 128 --n 128 --k 128 --output_dir ./input
```

### 运行算子

```bash
./build/sparse_gemm
```

### 验证结果

```bash
python3 scripts/verify_result.py --output ./output/output.bin --golden ./input/golden.bin --input_dir ./input
```

## 使用一键脚本

```bash
# 运行所有步骤
bash run.sh all

# 仅编译
bash run.sh build

# 仅运行（需要先编译）
bash run.sh run

# 仅验证（需要先运行）
bash run.sh verify

# 跳过编译（代码审查阶段使用）
bash run.sh --skip-build
```

## 输入输出格式

### 输入

- **A 矩阵**：稠密矩阵，shape=[batch, m, k]，dtype=int8_t
- **B 矩阵**：稀疏矩阵（2:4 格式），shape=[batch, n, k/2]，dtype=int8_t
- **Index 矩阵**：索引矩阵，shape=[batch, n, k/8]，dtype=uint8_t

### 输出

- **C 矩阵**：结果矩阵，shape=[batch, m, n]，dtype=float32

## 数学公式

```
C = A * B_sparse^T
```

其中：
- A 是稠密矩阵（m×k）
- B_sparse 是经过 2:4 稀疏压缩的矩阵（n×k/2）
- Index 记录稀疏矩阵 B 中非零元素的位置

## 参考示例

本算子参考以下示例：
- `asc-devkit/examples/01_simd_cpp_api/02_features/03_basic_api/01_matrix_compute/mmad_with_sparse/`

## 注意事项

1. K 维度必须是 4 的倍数（2:4 稀疏格式要求）
2. 索引矩阵采用 NZ 格式，需要 512 字节对齐
3. 当前实现使用硬编码参数，实际应该从 Tiling 数据获取
4. 当前实现假设 H_Q = H_K = 1，实际需要根据具体形状计算

## 已知问题和修复记录

### 2026-05-22：精度问题修复

**问题描述**：精度验证失败，Max diff: 2.36e+03

**问题原因**：golden 计算逻辑中，int32 结果直接保存为 int32 类型，但验证脚本期望 float32 类型，导致数据被错误解释。

**解决方案**：修改 gen_data.py 中的 gen_sparse_golden 函数，将 int32 结果转换为 float32 类型保存，与硬件输出一致。

**验证结果**：精度验证通过，Max diff: 0.000000e+00
