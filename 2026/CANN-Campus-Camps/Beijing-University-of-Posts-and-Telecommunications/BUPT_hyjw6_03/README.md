# 北京邮电大学

## 团队信息

- 提交者: hyjw6
- 身份: 学生
- 单位: 北京邮电大学

## 成员

- hyjw6 (hyjw6): 提交者

## 算子: op_03_relu

Tiling 策略：
- 根据输入元素总数和 AIV 核数动态进行多核切分
- 每个核的处理长度按 32B 对齐
- 根据数据类型、UB 容量及双缓冲占用动态计算单个 Tile 大小
- 对最后一个核和最后一个 Tile 进行尾块处理
- 根据输入数据类型选择对应的 TilingKey
Kernel 实现：
- 通过 CopyIn → Compute → CopyOut 流水完成计算
- 使用 AscendC::Relu 执行向量化 ReLU 运算
- 输入和输出队列均采用 DoubleBuffer，降低数据搬运等待开销
- 支持多核并行和 UB 分片处理
同时修复了 relu_tiling_data.h 中因模板乱码导致的结构体定义异常，并更新 Host Tiling 单元测试期望值。
