# 智梯

## 团队信息

- 提交者: Fdivinity
- 身份: 企业员工
- 单位: 智梯

## 成员

- Fdivinity (Fdivinity): 提交者

## 算子: op_01_sub

每个核处理 totalLength / blockNum 个元素；
每核划分为 TILE_NUM × BUFFER_NUM 个数据块；
CopyIn() 将 x/y 从 GM 搬到 LocalTensor；
AscendC::Sub() 完成 z = x - y；
CopyOut() 将结果写回 GM；
dtype == 0 使用 float，dtype == 1 使用 half。
