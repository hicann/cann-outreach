# 北京首芯科技文化有限公司

## 团队信息

- 提交者: zhangwei0402
- 身份: 企业员工
- 单位: 北京首芯科技文化有限公司

## 成员

- zhangwei0402 (zhangwei0402): 提交者

## 算子: op_01_mul

基于 Ascend C 实现逐元素乘法 z=x*y，支持 float32 和 float16。根据输入 shape 计算总元素数，每块处理 1024 个元素，启动核数取数据块数与可用核数的较小值；各核循环处理独立数据块。使用 TPipe/TQue 管理双缓冲，AscendC::Mul 执行向量乘法，32 字节对齐数据使用 DataCopy，非对齐尾部使用 DataCopyPad。CANNJudge 提交 272049：2/2 测试点 Pass，输出错误占比均为 0.00%。
