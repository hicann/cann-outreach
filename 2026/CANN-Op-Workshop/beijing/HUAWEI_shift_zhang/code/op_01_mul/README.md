# op_01_mul

提交团队: 华为技术有限公司
提交者: m0_61162086


## 算子实现介绍

使用单核实现，UB切分按32B对齐
Tiling策略：totalLength对齐到BLOCK_SIZE的整数倍
