# 东南大学

## 团队信息

- 提交者: kk_axilter
- 身份: 学生
- 单位: 东南大学

## 成员

- kk_axilter (kk_axilter): 提交者

## 算子: op_03_relu

采用多核并行与分块 Tiling 实现 ReLU 算子，计算公式为 y = max(0, x)。Host 侧根据输入元素总数和数据类型选择 FP16/FP32 内核及核数，并计算每核处理长度和 UB 分块大小。Kernel 侧完成 GM→UB 搬入、ReLU 矢量计算和 UB→GM 搬出；针对常用输入规模使用静态 Tensor 和编译期定长 Vector mask/repeat，减少 TPipe、队列管理及动态长度判断开销；较大输入使用分块流水处理，在保证计算正确性的基础上提升并行度和数据搬运效率。
