# 东南大学

## 团队信息

- 提交者: dzy0324seu
- 身份: 学生
- 单位: 东南大学

## 成员

- dzy0324seu (dzy0324seu): 提交者

## 算子: op_04_square

本 Square 算子基于 Ascend C 实现，对输入张量执行逐元素平方运算，即 `y = x × x`。算子支持 float16 和 float32 数据类型，并将任意维度的输入展平后进行多核并行处理。Host 侧根据张量元素总数、AIV 核数和 UB 容量动态生成 tiling 参数；Kernel 侧采用 Double Buffer 和流水线方式完成 GM→UB 搬入、Mul 计算及 UB→GM 搬出。针对无法被核数或 32 字节整除的场景，使用尾核、尾 tile 和 `DataCopyPad` 处理，确保输出形状、数据类型与输入一致且不会发生越界访问。
