# 东南大学

## 团队信息

- 提交者: kk_axilter
- 身份: 学生
- 单位: 东南大学

## 成员

- kk_axilter (kk_axilter): 提交者

## 算子: op_02_mul

本算子基于 Ascend C 实现逐元素矢量乘法 z = x × y，支持 float16 和 float32 数据类型。Host 侧根据输入 Shape 计算 totalLength，设置输出 Shape 和数据类型与输入 x 一致，并启动 8 个 AIV 核并行计算。针对固定输入 Shape（8, 2048）进行专用优化，每核处理 2048 个连续元素，数据搬运满足 32 字节对齐要求。Kernel 侧采用单缓冲、单大块流水设计，依次完成 CopyIn、AscendC::Mul 和 CopyOut，减少循环、Tiling 数据读取及调度开销。
