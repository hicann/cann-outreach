# 东南大学

## 团队信息

- 提交者: kk_axilter
- 身份: 学生
- 单位: 东南大学

## 成员

- kk_axilter (kk_axilter): 提交者

## 算子: op_01_sub

使用多核并行实现，UB 切分满足 32B 对齐。
Tiling 策略：根据可用核数划分 totalLength，各核处理连续数据，并按 256B Vector Repeat 粒度进行计算。Kernel 使用 Static Tensor，减少队列及 Tensor 管理开销，提高矢量减法执行效率。
