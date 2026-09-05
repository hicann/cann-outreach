# 北京邮电大学

## 团队信息

- 提交者: 俞博仁
- 身份: 学生
- 单位: 北京邮电大学

## 成员

- 俞博仁 (BorenYu2691): 提交者

## 算子: op_03_relu

题目内容
描述
一、赛题背景
本题要求基于 Ascend C 编程语言进行开源仓规范化算子开发，在昇腾 NPU 硬件上实现 ReLU 激活算子。

二、算子功能描述
PyTorch 原生算子：torch.nn.functional.relu，即 y = max(0, x)。

参考文档：https://docs.pytorch.org/docs/2.13/generated/torch.nn.modules.activation.ReLU.html#relu

三、修改内容
在 op_host/relu_tiling.cpp 内补充：

ReluTilingFunc：实现 tiling 计算
在 op_kernel/relu.h 内补充：

模板中 Init / CopyIn / Compute / CopyOut / Process 共 5 处 TODO
在通过验证的基础上，尽量优化性能，如调整tilenum,加入DoubleBuffer机制

初始模板：https://gitcode.com/weixin_45085744/AscendC_learning/tree/main/Lesson_03/code_temple

代码参考：https://gitcode.com/weixin_45085744/AscendC_learning/tree/main/Lesson_03/code_add
