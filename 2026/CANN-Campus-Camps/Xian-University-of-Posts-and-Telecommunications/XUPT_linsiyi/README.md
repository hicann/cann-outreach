# 西安邮电大学

## 团队信息

- 提交者: linsiyi
- 身份: 学生
- 单位: 西安邮电大学

## 成员

- linsiyi (fanmaiji): 提交者

## 算子: op_03_relu

Host侧ReluTilingFunc将张量展平，以32B对齐块均分任务到AI核，完成多核负载均衡。
Device侧Relu模板类使用双缓冲流水线，分CopyIn、Compute、CopyOut三步完成计算。
Tiling策略：总长度对齐至32B块大小整数倍，内核内部tile分片，控制UB内存占用。
