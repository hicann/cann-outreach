# 东南大学

## 团队信息

- 提交者: 王博_71124114
- 身份: 学生
- 单位: 东南大学

## 成员

- 王博_71124114 (wblxr408): 提交者

## 算子: op_02_mul

在 op_host/mul.cpp 内补充：

TilingFunc：按输入 shape 计算 tiling 数据并写回 tiling 结构体
InferShape：输出 z 的 shape 与输入 x 一致
InferDataType：输出 z 的 dtype 与输入一致
在 op_kernel/mul.cpp 内补充：

class KernelMul 类实现：Init / Process / CopyIn / Compute / CopyOut
在 op_kernel/mul_tiling.h 内补充：

MulTilingData 结构体字段
