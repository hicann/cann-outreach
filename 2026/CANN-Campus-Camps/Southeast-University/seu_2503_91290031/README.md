# 东南大学

## 团队信息

- 提交者: 2503_91290031
- 身份: 学生
- 单位: 东南大学

## 成员

- 2503_91290031 (2503_91290031): 提交者

## 算子: op_02_mul

算子功能
矢量乘法算子 Mul 用于计算两个输入张量的逐元素乘积，即 z = x * y。该算子基于 Ascend C 编程语言开发，运行于昇腾 NPU 硬件平台。

输入输出
类型	参数名	维度形状	数据类型	说明
INPUT	x	(8, 2048)	float32 / float16	第一个乘法输入张量
INPUT	y	(8, 2048)	float32 / float16	第二个乘法输入张量
OUTPUT	z	(8, 2048)	float32 / float16	计算结果张量
计算逻辑
对输入张量 x 和 y 中的每个对应元素执行乘法运算

输出张量 z 的每个元素为：z[i] = x[i] * y[i]

输入与输出的维度形状和数据类型完全一致

实现架构
采用 Ascend C 标准开发框架，基于三阶段流水线设计：

CopyIn：从 Global Memory 拷贝数据到 Local Memory (Unified Buffer)

Compute：调用 AscendC::Mul 指令执行向量乘法

CopyOut：将计算结果从 Local Memory 拷贝回 Global Memory

优化机制
双缓冲：使用 BUFFER_NUM = 2 实现数据搬运与计算并行

Tiling 分块：根据 Unified Buffer 大小动态划分数据块

多核并行：自动适配昇腾 NPU 的多核架构

适配平台
Atlas 训练系列 (ascend910b)

支持 ND 数据格式

数据类型：float16 / float32
