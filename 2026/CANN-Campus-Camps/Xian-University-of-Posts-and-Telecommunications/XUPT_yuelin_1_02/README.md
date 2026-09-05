# 西安邮电大学

## 团队信息

- 提交者: yuelin_1
- 身份: 学生
- 单位: 西安邮电大学

## 成员

- yuelin_1 (yuelin_1): 提交者

## 算子: op_02_mul

Host 侧算 tiling 并启动，Device 侧每颗 Vector Core 用 "拷贝 → 计算 → 拷出" 双缓冲流水处理一段数据。与初始版相比计算链 in-place 复用输出缓冲（3 队列→2 队列和tile 从 256 扩大到 2048，数值结果逐位不变。
