# 西安邮电大学

## 团队信息

- 提交者: 金佳蒙
- 身份: 学生
- 单位: 西安邮电大学

## 成员

- 金佳蒙 (AYB_): 提交者

## 算子: op_02_mul

在算子 Host 侧，首先通过 TilingFunc 根据输入 Tensor 的 Shape、数据类型以及总数据量计算 Kernel 执行所需的 tiling 参数，并将计算结果写入 MulTilingData 结构体，为后续核函数的数据切分和任务调度提供依据。同时，通过 InferShape 推导输出张量形状，使输出 z 与输入 x 保持一致；通过 InferDataType 保证输出数据类型与输入保持一致，从而完成算子的基本信息推导。

在 Kernel 侧，采用典型的 “CopyIn → Compute → CopyOut” 流水处理方式实现矢量乘法。Init 函数负责读取 Tiling 参数并初始化输入、输出 Global Memory 地址以及内部 Local Tensor；CopyIn 将 Global Memory 中的输入数据分块搬运到片上 Local Memory；Compute 使用 Ascend C 提供的矢量计算接口对对应数据块执行逐元素乘法；CopyOut 将计算结果从 Local Memory 搬回 Global Memory。Process 负责组织上述流程，并按照 Tiling 结果循环处理全部数据。

整体实现充分利用了昇腾 NPU 的片上存储和矢量计算能力，通过数据分块降低 Global Memory 与片上存储之间的数据搬运压力，同时利用流水化执行提高计算与数据搬运的重叠程度。对于本题 (8, 2048) 的输入规模，可以通过合理设置每次处理的数据块大小，使计算资源得到充分利用，在保证结果正确性的同时提高算子的执行效率。
