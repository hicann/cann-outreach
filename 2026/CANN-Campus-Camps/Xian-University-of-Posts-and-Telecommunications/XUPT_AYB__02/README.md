# 西安邮电大学

## 团队信息

- 提交者: AYB_
- 身份: 学生
- 单位: 西安邮电大学

## 成员

- AYB_ (AYB_): 提交者

## 算子: op_02_mul

本算子基于 Ascend C 实现向量乘法，完成 z = x * y 的计算。整体采用 Host 侧 Tiling + Kernel 侧向量计算 的方式实现。

Host 侧 Tiling
获取输入张量 x 的元素总数和数据类型。
将输入数据类型映射到对应的 Ascend C 模板参数，实现 float32 和 float16 两种数据类型的 Kernel 分发。
将输入元素总数写入 MulTilingData。
根据 NPU 的 AIV Core 数量确定 BlockDim，并保证每个 Core 负责的数据量能够按照 256 个元素一个 Tile 完整划分，避免部分数据未被计算。
本算子不需要额外的 Workspace。

Shape 与 DataType 推导
输出 z 的 Shape 与输入 x 保持一致。
输出 z 的数据类型与输入 x 保持一致。
因此对于题目中的输入 (8, 2048)，输出同样为 (8, 2048)。

Kernel 侧计算
每个 AI Vector Core 根据 GetBlockIdx() 获取自己的数据处理范围。
将全局内存中的 x、y 数据通过 DataCopy 搬运到 UB。
每次处理 256 个元素，通过双缓冲队列管理输入和输出数据。
在 UB 中调用 Ascend C 的 Mul 向量指令完成：
zLocal = xLocal * yLocal
将计算结果通过 DataCopy 写回全局内存。
