# 东南大学

## 团队信息

- 提交者: gcw_HQWyrJLy
- 身份: 学生
- 单位: 东南大学

## 成员

- gcw_HQWyrJLy (gcw_HQWyrJLy): 提交者

## 算子: op_03_relu

基于 Ascend C 实现 ReLU 激活算子 y = max(0, x)，采用多核 Tiling、UB 分块和 Double Buffer 流水机制，在 Kernel 中完成数据搬入、ReLU 计算和结果搬出，支持 float16 和 float32。
