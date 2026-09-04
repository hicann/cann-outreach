# 东南大学

## 团队信息

- 提交者: gcw_I1wD4CpJ
- 身份: 学生
- 单位: 东南大学

## 成员

- gcw_I1wD4CpJ (gcw_I1wD4CpJ): 提交者

## 算子: op_01_sub

这是一个把 x、y 从 Global 搬进本地、用 Sub 算出 z=x−y 再搬回 Global 的矢量算子，靠"多核切分 + 单核 tiling + 双缓冲"跑出流水线并行，切块默认按 32B 对齐。
