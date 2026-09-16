# 西安邮电大学

## 团队信息

- 提交者: 薛雅洁
- 身份: 学生
- 单位: 西安邮电大学

## 成员

- 薛雅洁 (xinya_): 提交者

## 算子: op_01_sub

本算子基于 Ascend C 原生开发实现矢量减法 z = x - y，采用多核并行 + Double Buffer 流水线方案。
1. 多核切分：host 侧 run_kernel 根据 TensorGroupInfo 计算总元素数 totalLength，kernel 侧通过 AscendC::GetBlockNum() 获取核数，将数据均匀切分到各核，每核处理 blockLength = totalLength / blockNum 个元素，核间无数据依赖。
2. Tiling 策略：每核数据再按 tileNum × BUFFER_NUM 细分为若干小块，tileLength = blockLength / tileNum / BUFFER_NUM。本实现 tileNum = 8，BUFFER_NUM = 2，保证数据规模在 UB 容量范围内，同时满足 Double Buffer 流水需求。
3. UB 切分与对齐：输入 x、y 与输出 z 各分配 BUFFER_NUM 块 UB 空间，每块大小 tileLength × sizeof(T)。tileLength 为 BLOCK_SIZE（32B）的整数倍，保证 DataCopy 搬运按 32B 对齐，避免非对齐访问带来的性能损失。
4. 流水线设计：通过 TPipe + TQue 管理 UB 内存，CopyIn → Compute → CopyOut 三段式循环，利用 Double Buffer 让搬运与计算在时间上重叠，掩盖访存延迟。
5. dtype 分派：host 侧根据 TensorInfo::dtype 选择核函数实例，fp16 走 sub_custom&lt;half&gt;，fp32 及其他走 sub_custom&lt;float&gt;。
验证结果：验证集测试点 1 通过，输出错误占比 0.00%，功能正确。
