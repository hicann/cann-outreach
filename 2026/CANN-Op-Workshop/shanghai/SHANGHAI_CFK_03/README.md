# 上海实操工坊

## 团队信息

- 提交者: 孙涛
- 身份: 企业员工
- 单位: 上海实操工坊

## 成员

- 孙涛 (CFK): 提交者

## 算子: op_03_relu

Relu 实现 `y = max(0, x)`。Tiling 按输入类型、32 字节对齐和 UB 可用空间确定核间 `blockFactor` 与核内 `ubFactor`，并选择对应 tiling key。Kernel 使用输入、输出双缓冲；每块通过 `DataCopyPad` 对齐搬运，以 `Maxs` 与零比较后写回 GM。支持 float32、float16 的 ND `(8, 2048)` 输入，并处理尾块。
