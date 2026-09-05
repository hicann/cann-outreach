# 西安邮电大学

## 团队信息

- 提交者: fx12345
- 身份: 学生
- 单位: 西安邮电大学

## 成员

- fx12345 (fx12345): 提交者

## 算子: op_02_mul

矢量乘法算子 `z = x * y` 采用 Host-Kernel 协同架构：Host 侧 `TilingFunc` 从输入 shape 累乘得到总元素数 `totalLength=16384`，配合常量 `tileNum=8` 写入 `MulTilingData` 并固定启动 8 核，`InferShape` 和 `InferDataType` 直接将输出 z 的形状与数据类型对齐为输入 x 的值；Kernel 侧核函数 `mul` 解析 tiling 数据后实例化 `KernelMul&lt;DT_X&gt;`（DT_X 由 tiling key 自动选为 float 或 half），`Init` 阶段按 `blockLength=totalLength/8` 完成多核地址切分并按 `tileLength=blockLength/8/2=128` 为三个双缓冲队列分配 UB，`Process` 阶段以 `tileNum×BUFFER_NUM=16` 次循环调度 CopyIn（GM→UB 搬入 x、y）、Compute（`AscendC::Mul` 逐元素相乘）、CopyOut（UB→GM 写出 z），每核 16×128=2048 元素恰好覆盖，8 核无重叠无遗漏地完成全量矢量乘法计算
