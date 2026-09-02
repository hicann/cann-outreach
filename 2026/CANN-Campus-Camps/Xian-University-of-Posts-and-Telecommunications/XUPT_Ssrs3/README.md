# 西安邮电大学

## 团队信息

- 提交者: 陈宇楠
- 身份: 学生
- 单位: 西安邮电大学

## 成员

- 陈宇楠 (Ssrs3): 提交者

## 算子: op_01_sub

本算子实现 z = x − y，采用 Vector 核的“多核切分 + 核内分块 + Double Buffer”方案：将 totalLength 平均切分给 8 个核并行处理，每个核再将数据划分为 tileNum 块，配合双缓冲流水执行 CopyIn→Compute→CopyOut；Tiling 策略上保证 totalLength 对齐到 BLOCK_SIZE（32B）的整数倍，UB 内每块切分长度 tileLength×4B 均按 32B 对齐，满足 DataCopy 的对齐要求；计算通过 AscendC 的 Sub 接口完成，host 侧 run_kernel 依据 TensorGroupInfo 推导 totalLength 并构造 tiling 结构体传给核函数 sub_custom 启动执行。
