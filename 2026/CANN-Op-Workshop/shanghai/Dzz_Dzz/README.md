# DZZ

## 团队信息

- 提交者: dzzw123
- 身份: 企业员工
- 单位: DZZ

## 成员

- dzzw123 (dzzw123): 提交者

## 算子: op_01_sub

算子功能：z = x - y。

工程结构：Host侧（run_kernel）计算Tiling并启动核函数；Kernel侧（KernelSub类）执行实际计算。

Tiling参数：SubCustomTilingData 包含 totalLength, blockNum, coreDataLen。

多核并行：根据 availableCoreNum 切分数据，每个核处理 coreDataLen 个元素。

数据搬运：使用 TPipe, TQue, DataCopy 在 GM 和 UB 间搬运。

计算：使用 AscendC::Sub 指令。

流水线：使用双缓冲（BUFFER_NUM=2）或单缓冲，循环处理分块。

关键常量：BUFFER_SIZE 或 BUFFER_NUM, QUEUE_DEPTH 等。
