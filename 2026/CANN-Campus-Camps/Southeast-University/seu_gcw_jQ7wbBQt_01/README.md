# 东南大学

## 团队信息

- 提交者: 谷泽楷
- 身份: 学生
- 单位: 东南大学

## 成员

- 谷泽楷 (gcw_jQ7wbBQt): 提交者

## 算子: op_01_sub

使用多核实现，UB 切分按 32B 对齐
Tiling 策略：totalLength 对齐到 BLOCK_SIZE 的整数倍；总数据均匀切分给每个 AI Core，单 Core 内部再拆分为多块 tile，配合 Double Buffer 流水线完成计算。
