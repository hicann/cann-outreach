# 西安邮电大学

## 团队信息

- 提交者: X1217000
- 身份: 学生
- 单位: 西安邮电大学

## 成员

- X1217000 (X1217000): 提交者

## 算子: op_01_sub

Host 侧：完成算子定义、形状推导（输出 shape 与输入一致）、数据类型推导（输出 dtype 与输入一致），并根据输入元素总数和核数进行 Tiling 切分，将 totalLength、tileNum、tileLength 写入 AddTilingData 结构体，设置多核并行策略。
Kernel 侧：每个核根据 Tiling 信息，通过双缓冲流水（CopyIn → Compute → CopyOut）完成计算，核心调用 Add() 实现 z = x + y 的矢量加法，充分利用多核并行与片上 UB 内存，保证高性能执行。
