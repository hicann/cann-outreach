# 西安邮电大学

## 团队信息

- 提交者: linsiyi
- 身份: 学生
- 单位: 西安邮电大学

## 成员

- linsiyi (fanmaiji): 提交者

## 算子: op_10_add

采用Host侧Tiling+AICore Kernel的异构开发方案，多核并行计算，UB缓冲区切分要求32B对齐，Tiling策略将总元素长度totalLength向上对齐至BLOCK_SIZE整数倍。
Host 侧:
算子注册：定义 Add 算子输入`x`、`y`与输出`z`，支持 FP32/FP16 数据类型，ND 格式；绑定 InferShape、InferDataType、TilingFunc 函数。
   - Tiling 函数：读取输入张量 shape，计算元素总长度`totalLength`；配置切分块数`tileNum=8`，设置核数`BlockDim=8`；将`totalLength`对齐为`BLOCK_SIZE`整数倍，填充到自定义`AddTilingData`结构体，下发给 Kernel。
   - 形状推导：输出 shape 与输入 shape 保持完全一致；数据类型继承输入数据类型。
2. Tiling 模板
   - `AddTilingData`结构体承载切分参数：`totalLength`总元素数、`tileNum`分片数量。
   - 通过`ASCENDC_TPL`宏做模板参数声明与选择，适配`DT_FLOAT`、`DT_FLOAT16`两种数据类型，实现算子多类型泛化。
3. Kernel 侧
   - 采用双缓冲流水线架构，BUFFER_NUM=2，使用 TPipe/Queue 完成 GlobalMemory 与 UB 之间数据搬运，分三段：CopyIn、Compute、CopyOut。
   - `Init`：从 Tiling 数据获取`totalLength`与`tileNum`，计算单核内总长度、每 tile 长度，初始化队列与全局张量地址。
   - `Process`循环：循环读取分片，CopyIn 将 GM 的 x、y 分片搬入 UB 局部张量；Compute 调用 AICore 内置 Add 接口完成逐元素加法；CopyOut 将 UB 计算结果写回 GM 输出地址。
   - 全局入口函数：通过`GET_TILING_DATA_WITH_STRUCT`拿到 Host 下发的 tiling 参数，实例化 KernelAdd 并启动流水线执行。
