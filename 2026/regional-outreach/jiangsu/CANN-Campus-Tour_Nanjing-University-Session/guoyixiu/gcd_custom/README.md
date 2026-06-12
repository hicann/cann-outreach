# Gcd 算子 — Ascend C 二元最大公约数算子

## 概述

Gcd 是一个基于 Ascend C 的自定义二元最大公约数算子，支持 **float16** 数据类型和 **ND format**，适用于 **ascend910b/ascend910_93/ascend950** 芯片架构。

输入 `self` 和 `other` 满足 **broadcast 关系**，输出 shape = broadcast(self_shape, other_shape)。

## 项目结构

```
gcd_op/
├── op_host/                    Host 侧代码
│   ├── gcd_def.cpp             算子定义（self, other → out，float16）
│   ├── gcd_infershape.cpp      Shape 推导（broadcast 规则）
│   ├── arch22/gcd_tiling.cpp   Tiling 实现（ascend910b）
│   ├── arch35/gcd_tiling.cpp   Tiling 实现（ascend950）
│   └── CMakeLists.txt
├── op_kernel/                  Kernel 侧代码
│   ├── gcd_arch22.cpp          Kernel 入口（arch22）
│   ├── gcd_arch35.cpp          Kernel 入口（arch35）
│   ├── arch22/
│   │   ├── gcd.h               Kernel 类（含 GcdScalar + Broadcast 索引映射）
│   │   ├── gcd_tiling_data.h   Tiling 数据结构（含 shape/strides）
│   │   └── gcd_tiling_key.h    Tiling 模板参数（float16, BUFFER_MODE）
│   ├── arch35/                 （同上）
│   └── CMakeLists.txt
├── op_api/                     ACLNN 接口
│   ├── gcd.h / gcd.cpp         L0 API（broadcast InferShape + Kernel 调度）
│   └── aclnn_gcd.h/cpp         L2 API（参数检查 + Contiguous/ViewCopy）
├── op_graph/
│   └── gcd_proto.h             图模式 Proto
├── examples/
│   ├── test_aclnn_gcd.cpp      调用示例（含 broadcast 测试用例）
│   ├── CMakeLists.txt
│   └── run.sh
├── CMakeLists.txt              顶层 CMake 配置
└── build.sh                    一键编译脚本
```

## 支持的特性

| 特性 | 说明 |
|------|------|
| 算子类型 | 二元（self, other → out） |
| 数据类型 | float16 |
| Format | ND |
| Shape | 任意维（最大 8 维），支持 broadcast |
| Broadcast 规则 | 尾部对齐，逐维度取 max |
| 芯片架构 | ascend910b, ascend910_93, ascend950 |
| 调用方式 | ACLNN L2 API（两段式）、图模式 |

## 算法

采用**欧几里得辗转相减法**（Euclidean algorithm by subtraction）：

```
gcd(a, b):
  a = abs(a), b = abs(b)
  if a == 0: return b
  if b == 0: return a
  while a != b:
    if a > b: a = a - b
    else:     b = b - a
  return a
```

## Broadcast 索引映射

Kernel 内部逐元素处理，通过 Tiling 传递的 shape/strides 信息实现 broadcast 索引映射：

```
对输出位置 outIdx[d]：
  selfIdx[d]  = (selfShape[d] == 1)  ? 0 : outIdx[d]
  otherIdx[d] = (otherShape[d] == 1) ? 0 : outIdx[d]
```

## 编译方法

```bash
cd gcd_op

# 编译 ascend910b
bash build.sh --soc=ascend910b -j8

# 编译并运行示例（需先安装算子包）
bash build.sh --soc=ascend910b -e
```

## 调用的 ACLNN 接口

```c
#include "aclnn/aclnn_gcd.h"

// 两段式调用
uint64_t workspaceSize = 0;
aclOpExecutor* executor = nullptr;
aclnnGcdGetWorkspaceSize(tensorSelf, tensorOther, tensorOut,
                         &workspaceSize, &executor);
aclnnGcd(workspaceAddr, workspaceSize, executor, stream);
```

## 测试用例（example 中包含）

| 用例 | self shape | other shape | out shape | 说明 |
|------|-----------|-------------|-----------|------|
| 1 | [2,4,8,16] | [2,4,8,16] | [2,4,8,16] | 同 shape 4 维 |
| 2 | [1,4,1,1] | [2,1,8,1] | [2,4,8,1] | Broadcast 4 维 |
| 3 | [4,1] | [1,8] | [4,8] | Broadcast 2 维 |
| 4 | [1,128] | [4,1] | [4,128] | Broadcast 2 维 |
| 5 | [3] | [4,3] | [4,3] | 不同维度数 broadcast |
