# 东南大学

## 团队信息

- 提交者: gcw_jQ7wbBQt
- 身份: 学生
- 单位: 东南大学

## 成员

- gcw_jQ7wbBQt (gcw_jQ7wbBQt): 提交者

## 算子: op_02_mul

UB切分按 32B 对齐，满足DataCopy硬件对齐约束
Tiling策略：totalLength 对齐到BLOCK_SIZE的整数倍，多AICore均分数据，核内做二次分块
