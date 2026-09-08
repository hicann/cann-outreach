# 东南大学

## 团队信息

- 提交者: gcw_HQWyrJLy
- 身份: 学生
- 单位: 东南大学

## 成员

- gcw_HQWyrJLy (gcw_HQWyrJLy): 提交者

## 算子: op_01_sub

基于 Ascend C 实现矢量减法算子 z = x - y，采用 CopyIn → Compute → CopyOut 流水处理，并通过 Tiling 对数据进行分块，实现多核并行计算，支持 float16 和 float32 数据类型。
