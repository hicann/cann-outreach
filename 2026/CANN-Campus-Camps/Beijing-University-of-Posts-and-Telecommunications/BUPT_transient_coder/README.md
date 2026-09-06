# 北京邮电大学

## 团队信息

- 提交者: Stephen
- 身份: 学生
- 单位: 北京邮电大学

## 成员

- Stephen (transient_coder): 提交者

## 算子: op_03_relu

实现课堂 ReLU 算子 y=max(0,x)，补全 ReluTilingFunc 及 Init、CopyIn、Compute、CopyOut、Process。使用分核、UB 分块和双缓冲队列，调用 AscendC::Relu 完成计算。CANNJudge 提交 201480 已通过 2/2 测试点。验证范围为课堂题目规定的 (8, 2048)、float32/float16、ND 输入。
