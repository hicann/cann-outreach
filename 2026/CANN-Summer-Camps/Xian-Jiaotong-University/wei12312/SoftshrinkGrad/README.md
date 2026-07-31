# SoftshrinkGrad 算子项目

**CANN SUMMER CAMPS 2026 · XJTU · 第16组**

本目录是项目设计题目 **SoftshrinkGrad 算子开发** 的项目交付件，包含 Ascend C
自定义算子工程、ACLNN 调用样例、CPU 参考实现测试、构建脚本和项目任务说明。
算子设计方案与自测报告由第16组的“算子设计文档”PR独立提交。

## 功能

给定上游梯度 `gradOutput`、Softshrink 前向输入 `self` 和阈值 `lambd`，逐元素计算：

```text
gradInput[i] = gradOutput[i],  self[i] > lambd 或 self[i] < -lambd
               0,             其他情况
```

等价写法为：

```text
gradInput = (abs(self) > lambd) ? gradOutput : 0
```

边界 `self == ±lambd` 的输出为 `0`。

## 支持范围

- 数据类型：FLOAT、FLOAT16、BFLOAT16。
- 数据格式：ND。
- Shape：0～8 维，支持动态 shape 和空 Tensor。
- 平台：Atlas A2/Ascend 910B、Atlas A3/Ascend 910_93。
- `lambd`：FLOAT 属性，默认 `0.5`，要求 `lambd >= 0`。
- 核心算子要求两个输入同 shape、同 dtype；ACLNN 接口层先完成连续化、dtype
  统一和 broadcast，再调用核心算子。

## 目录

```text
SoftshrinkGrad/
├── docs/
│   └── 项目设计题目.md
├── custom_op/
│   ├── op_host/                 # 原型、Shape推导、Tiling
│   ├── op_kernel/               # Ascend C Kernel
│   ├── test/                    # ACLNN端到端样例
│   ├── CMakeLists.txt
│   ├── CMakePresets.json
│   └── build.sh
├── test/
│   └── test_reference.py        # 无NPU也可执行的参考测试
└── run.sh
```

## 快速验证

### 1. CPU 参考测试

```bash
python3 test/test_reference.py
```

### 2. NPU 编译和端到端测试

```bash
export ASCEND_TOOLKIT_HOME=/usr/local/Ascend/ascend-toolkit/latest
bash run.sh
```

`run.sh` 会依次编译自定义算子、安装到当前用户目录、编译 ACLNN 样例并执行。

## 验证结果

已于 2026-07-30 在以下环境完成设备端验证：

- CPU架构：aarch64。
- NPU：Ascend 910B3。
- CANN Toolkit：9.0.0。
- CMake：3.22.1。
- GCC/G++：11.4.0。

验证结果：

- Python CPU参考测试：8/8通过。
- Host/Tiling及Ascend C Kernel编译：通过。
- 910B算子二进制生成、打包与安装：通过。
- ACLNN端到端运行及结果比对：通过。
- 最终输出：`SoftshrinkGrad ACLNN test: PASS`。

容器已经完成物理NPU到逻辑设备0的映射时，不应再次设置
`ASCEND_RT_VISIBLE_DEVICES`；测试程序默认使用逻辑设备0。

## 文档入口

- [项目设计题目](docs/项目设计题目.md)
