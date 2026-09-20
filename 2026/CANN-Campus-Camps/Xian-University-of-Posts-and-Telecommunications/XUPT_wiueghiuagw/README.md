# 西安邮电大学

## 团队信息

- 提交者: wiueghiuagw
- 身份: 学生
- 单位: 西安邮电大学

## 成员

- wiueghiuagw (wiueghiuagw): 提交者

## 算子: op_01_sub

本算子使用Ascend C语言开发，部署在昇腾NPU上，实现张量逐元素减法运算z=x−y。算子分为Host侧与Device侧，Host侧完成分块Tiling参数的构造并启动核函数；Device侧负责将全局内存中的输入张量搬运至片上存储，调用矢量减法指令完成计算，再将结果写回全局内存，最终在CANNTJudge平台完成功能验证。
