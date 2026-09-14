# 南京邮电大学

## 团队信息

- 提交者: kingpoem
- 身份: 学生
- 单位: 南京邮电大学

## 成员

- kingpoem (sun_king_poem): 提交者

## 算子: op_01_relu

基于 Ascend C 实现 ReLU：多核均分输入后，用TQueBind将同一块 UB 绑定为输入输出，经DataCopy搬入、原地AscendC::Relu计算、再写回 Global Memory；单缓冲一次处理完本核数据，支持 float32 / float16。
