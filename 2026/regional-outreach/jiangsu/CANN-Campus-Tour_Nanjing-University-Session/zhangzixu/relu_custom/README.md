# ReluExample

## 产品支持情况

| 产品 | 是否支持 |
| ---- | :----:|
| Atlas A3 训练系列产品/Atlas A3 推理系列产品 | √ |
| Atlas A2 训练系列产品/Atlas A2 推理系列产品 | √ |

## 功能说明

- **算子功能**：计算张量每个元素的线性整流值（Rectified Linear Unit）。

- **计算公式**：

$$
y = \text{Relu}(x) = \max(0, x)
$$

- **数据类型**：float16

- **数据格式**：ND

- **Shape**：支持任意维度，主要面向 4D shape `[N4, N3, N2, N1]`

## 目录结构

```
relu_example/
├── op_host/                        # Host 侧代码
│   ├── relu_example_def.cpp        # 算子定义
│   ├── relu_example_infershape.cpp # 形状推导
│   ├── arch22/                     # arch22 架构 Tiling
│   └── arch35/                     # arch35 架构 Tiling
├── op_kernel/                      # Kernel 侧代码
│   ├── arch22/                     # arch22 架构实现
│   └── arch35/                     # arch35 架构实现
├── op_api/                         # ACLNN 接口
│   ├── aclnn_relu_example.cpp      # L2 API 实现
│   ├── aclnn_relu_example.h        # L2 API 头文件
│   ├── relu_example.cpp            # L0 API 实现
│   └── relu_example.h              # L0 API 头文件
├── tests/                          # 测试代码
│   ├── ut/                         # 单元测试
│   │   └── op_host/                # Host 侧 UT
│   └── st/                         # 系统测试
├── CMakeLists.txt                  # 构建配置
└── build.sh                        # 构建脚本
```

## 快速验证

```bash
# 1. 设置环境变量
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 2. 编译算子
bash build.sh --soc=ascend910b -j8

# 3. 运行 UT 测试
bash build.sh -u --soc=ascend910b

# 4. 运行 ST 测试（Real 模式，需 NPU）
bash build.sh -s --soc=ascend910b
```

## 参数说明

| 参数名 | 输入/输出/属性 | 描述 | 数据类型 | 数据格式 |
|--------|--------------|------|---------|---------|
| x | 输入 | 输入张量 | FLOAT16 | ND |
| y | 输出 | Relu 计算结果（负数置零，正数保持） | FLOAT16 | ND |

## 约束说明

- 仅支持 float16 数据类型
- 输入输出 shape 必须一致

## 调用说明

| 调用方式 | 调用样例 | 说明 |
|---------|---------|------|
| aclnn调用 | ST 测试 (C++) | 系统测试通过 C++ 原生测试验证算子精度 |
