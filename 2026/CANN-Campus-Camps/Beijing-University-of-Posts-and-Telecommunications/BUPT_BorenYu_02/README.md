# 北京邮电大学

## 团队信息

- 提交者: 俞博仁
- 身份: 学生
- 单位: 北京邮电大学

## 成员

- 俞博仁 (BorenYu2691): 提交者

## 算子: op_02_mul

题目内容
描述
一、赛题背景
本题要求基于 Ascend C 编程语言进行算子原生开发，在昇腾 NPU 硬件上实现一款矢量乘法算子。

二、算子功能描述
矢量乘法：z = x * y。

三、修改内容
在 op_host/mul.cpp 内补充：

TilingFunc：按输入 shape 计算 tiling 数据并写回 tiling 结构体
InferShape：输出 z 的 shape 与输入 x 一致
InferDataType：输出 z 的 dtype 与输入一致
在 op_kernel/mul.cpp 内补充：

class KernelMul 类实现：Init / Process / CopyIn / Compute / CopyOut
在 op_kernel/mul_tiling.h 内补充：

MulTilingData 结构体字段（需按 tiling 方案补充）
初始模板：https://gitcode.com/weixin_45085744/AscendC_learning/tree/main/Lesson_02/code_template

代码参考：https://gitcode.com/weixin_45085744/AscendC_learning/tree/main/Lesson_02/code_add
