# 西安邮电大学

## 团队信息

- 提交者: 金佳蒙
- 身份: 学生
- 单位: 西安邮电大学

## 成员

- 金佳蒙 (AYB_): 提交者

## 算子: op_03_relu

该 ReLU 算子基于昇腾 Ascend C 编程模型实现，主要采用 Host 侧 Tiling、Device 侧多核并行以及 Vector 计算 的方式完成。算子的核心计算为 \(y=\max(0,x)\)，支持 FP16 和 FP32 数据类型。Host 侧通过 relu_tiling.cpp 获取输入张量的元素总数，并根据输入规模划分计算任务。本实现针对固定的 8×2048 输入，将数据划分到 8 个 AI Core，每个 Core 负责 2048 个元素，再将每个 Core 的数据进一步划分为多个 Tile，以降低单次数据处理量并提高数据搬运效率。Device 侧由 relu.cpp 作为 Kernel 入口，通过 GET_TILING_DATA 获取 Tiling 参数，并创建 KernelRelu 对象执行计算。在 relu.h 中，首先通过 Init 根据当前 Core 编号计算其对应的 Global Memory 数据区域，并初始化输入、输出队列和 Buffer。随后 CopyIn 使用 DataCopy 将 Global Memory 中的数据搬运到 Unified Buffer（UB），Compute 调用 Ascend C 提供的 Vector Relu 接口，对 Tile 内的数据进行逐元素计算，最后 CopyOut 将计算结果从 UB 搬回 Global Memory。整个过程采用 CopyIn → Compute → CopyOut 的分块处理模式，同时利用多 AI Core 并行处理不同数据区域，从而提高整体执行效率。通过合理设置 Core 数量和 Tile 大小，可以在保证计算正确性的同时减少数据搬运和调度开销。
