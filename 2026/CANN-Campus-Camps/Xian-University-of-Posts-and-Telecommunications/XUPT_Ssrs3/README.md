# 西安邮电大学

## 团队信息

- 提交者: 陈宇楠
- 身份: 学生
- 单位: 西安邮电大学

## 成员

- 陈宇楠 (Ssrs3): 提交者

## 算子: op_01_sub

采用多核并行 + 核内 Double Buffer 流水实现。Tiling 策略：总数据量按 availableCoreNum 均分至各核，blockLength 对齐到 BLOCK_SIZE（32B）整数倍，核内按 tileLength 切块，每块数据经过 CopyIn → Compute → CopyOut 三级流水，通过 TQue 队列实现 MTE2（搬入）、Vector（计算）、MTE3（搬出）的并行流水，充分提升 Vector 核利用率。
