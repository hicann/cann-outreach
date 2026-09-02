# 西安邮电大学

## 团队信息

- 提交者: 金佳蒙
- 身份: 学生
- 单位: 西安邮电大学

## 成员

- 金佳蒙 (AYB_): 提交者

## 算子: op_01_sub

整体采用 Host + Kernel 的执行方式：
Host 侧
从 TensorGroupInfo 获取输入张量的 Shape 和数据类型。
根据张量总元素数和 availableCoreNum 计算核间数据划分。
将每个 Core 负责的数据进一步划分为多个 Tile。
根据数据类型选择对应的 Kernel，并通过 sub_custom&lt;&lt;&lt;...&gt;&gt;&gt; 启动计算。
Kernel 侧
每个 Core 从 Global Memory 中读取对应的 x、y 数据到 Local Memory。
使用 Ascend C 的 Sub API 执行逐元素减法：
z = x - y。
将计算结果从 Local Memory 写回 Global Memory。
流水化处理
使用 TPipe 和 TQue 管理数据搬运与计算过程，通过 Tile 分块实现数据的分批处理，提高计算和内存访问效率。
