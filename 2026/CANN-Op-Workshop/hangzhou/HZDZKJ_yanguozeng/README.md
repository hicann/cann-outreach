# 杭州电子科技大学

## 团队信息

- 提交者: wsdsg_23
- 身份: 学生
- 单位: 杭州电子科技大学

## 成员

- wsdsg_23 (wsdsg_23): 提交者

## 算子: op_01_relu

算子采用多核并行 + Double Buffer 方式实现 ReLU 计算。Host 侧根据输入总元素数和核数进行 Tiling，每个核负责一段连续数据，并进一步切分为多个 Tile。Kernel 侧通过 DataCopy 将数据从 GM 搬运到 UB，调用 AscendC::Relu 完成 y = max(0, x) 计算，再将结果搬回 GM。UB 缓冲区按 32B 对齐，以满足数据搬运和向量计算要求。
