# 北京邮电大学

## 团队信息

- 提交者: 邵文天
- 身份: 学生
- 单位: 北京邮电大学

## 成员

- 邵文天 (s-2024): 提交者

## 算子: op_01_sub

双缓冲：BUFFER_NUM = 2，每个核的数据被切成 tileNum * BUFFER_NUM 次搬运，tileLength = blockLength / tileNum / BUFFER_NUM，与官方 Add 样例的切分方式一致
