# 东南大学

## 团队信息

- 提交者: dzy0324seu
- 身份: 学生
- 单位: 东南大学

## 成员

- dzy0324seu (dzy0324seu): 提交者

## 算子: op_01_sub

本算子基于 Ascend C 实现二维张量的逐元素减法运算 z=x-y。算子采用 Host 与 Device 协同执行模式，Host 侧负责读取输入张量信息、生成 Tiling 参数并完成核函数调度，Device 侧负责实际的数据搬运与向量计算。Kernel 内部使用 TPipe 和 TQue 构建流水线，将计算过程划分为 CopyIn、Compute 和 CopyOut 三个阶段，通过 DataCopy 完成 Global Memory 与 Unified Buffer 之间的数据传输，并调用 AscendC::Sub 向量接口完成减法计算。同时，通过多核并行、Tiling 分块以及 Double Buffer 机制提高数据搬运与计算的重叠程度，从而充分利用昇腾 NPU Vector Core 的并行计算能力。算子支持 float16 和 float32 数据类型，输入输出张量形状保持一致。
