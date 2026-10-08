# op_01_sub

提交团队: 山东大学
提交者: 顾君


## 算子实现介绍

算子采用多核切分+Tile+Local Memory 计算的执行方式，将全部数据划分给 8 个 Vector Core，每个核处理一行数据，共 2048 个元素。每个核再将数据划分成多个 Tile，每次从 Global Memory 搬运一小块数据到 Local Memory。Local Memory 中使用向量减法完成 x - y。之后将计算结果写回 Global Memory。
使用双缓冲管理输入和输出数据，从而减少数据搬运对计算过程的影响。
再通过核编号计算每个核对应的数据起始位置，避免多个核处理相同数据。
然后根据输入数据类型选择 float32 或 float16 计算。
