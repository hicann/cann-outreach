# 北京邮电大学

## 团队信息

- 提交者: Stephen
- 身份: 学生
- 单位: 北京邮电大学

## 成员

- Stephen (transient_coder): 提交者

## 算子: op_02_mul

实现课堂 Mul 算子 z=x*y，补全 Host TilingFunc、InferShape、InferDataType、MulTilingData 及 KernelMul。根据 AIV 核数和 UB 容量进行多核分块，使用双缓冲队列组织搬运与 AscendC::Mul 计算。CANNJudge 提交 201427 已通过 2/2 测试点。验证范围为课堂题目规定的 (8, 2048)、float32/float16、ND 输入。
