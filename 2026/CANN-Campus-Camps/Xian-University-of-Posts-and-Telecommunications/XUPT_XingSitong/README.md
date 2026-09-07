# 西安邮电大学

## 团队信息

- 提交者: X1217000
- 身份: 学生
- 单位: 西安邮电大学

## 成员

- X1217000 (X1217000): 提交者

## 算子: op_03_relu

Host 侧（op_host/add.cpp）：完成算子定义（OpDef）、形状推导（InferShape，输出 shape 与输入一致）、数据类型推导（InferDataType，输出 dtype 与输入一致）以及 Tiling 切分（TilingFunc，按输入元素总数和核数计算多核/单核 tile 方案，并写入 AddTilingData结构体）。
Kernel 侧（op_kernel/add.cpp）：实现 KernelAdd类，通过双缓冲流水（BUFFER_NUM=2）完成 CopyIn → Compute → CopyOut循环，核心计算调用 Add()完成 z = x + y的矢量加法，支持多核并行处理。
