# 东南大学

## 团队信息

- 提交者: dzy0324seu
- 身份: 学生
- 单位: 东南大学

## 成员

- dzy0324seu (dzy0324seu): 提交者

## 算子: op_03_relu

本实验基于 Ascend C 实现 ReLU 激活算子，其计算公式为 y=\max(0,x)。输入和输出张量 Shape 均为 (8, 2048)，支持 float16 和 float32 数据类型。Host 侧通过 ReluTilingFunc 获取输入张量 Shape 以及 NPU 平台的 UB 和 Core 信息，根据输入规模计算总元素数量、单核处理数据量以及 UB 分块大小，并通过 ReLuTilingData 将 totalNum、blockFactor 和 ubFactor 等参数传递给 Kernel。Kernel 侧采用 Init、CopyIn、Compute、CopyOut 和 Process 的流水线结构，通过 DataCopy 将数据由 Global Memory 搬运至 Unified Buffer，在 Vector Core 上调用 Ascend C 的 Relu 向量接口完成逐元素激活运算，再将结果写回 Global Memory。同时使用 TPipe、TQue 和 Double Buffer 管理数据流水，提高数据搬运与计算的并行程度，从而充分利用昇腾 NPU 的向量计算能力。
