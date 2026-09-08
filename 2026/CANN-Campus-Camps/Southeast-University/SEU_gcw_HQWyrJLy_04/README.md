# 东南大学

## 团队信息

- 提交者: gcw_HQWyrJLy
- 身份: 学生
- 单位: 东南大学

## 成员

- gcw_HQWyrJLy (gcw_HQWyrJLy): 提交者

## 算子: op_04_square

基于 Ascend C 实现 Square 算子 y = x × x，支持任意维度输入，通过动态 Tiling、UB 分块及 Double Buffer 实现高效计算，并使用非对齐数据搬运机制兼容非 32B 整倍数的尾块，支持 float16 和 float32。
