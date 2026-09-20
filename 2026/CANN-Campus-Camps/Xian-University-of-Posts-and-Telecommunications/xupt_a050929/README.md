# 西安邮电大学

## 团队信息

- 提交者: a050929
- 身份: 学生
- 单位: 西安邮电大学

## 成员

- a050929 (a050929): 提交者

## 算子: op_03_relu

本算子基于Ascend C语言开发，在昇腾NPU上实现ReLU激活函数运算，计算公式为$y=\max(0,x)$。算子分为Host侧与Device侧，Host侧完成Tiling分块参数计算；Device侧通过Init、CopyIn、Compute、CopyOut、Process接口，将全局内存的输入张量搬运至片上存储，调用矢量指令完成取最大值运算，再将结果写回全局内存，支持DoubleBuffer等性能优化，并在CANNTJudge平台完成功能验证。
