# 东南大学

## 团队信息

- 提交者: 2401_87688128
- 身份: 学生
- 单位: 东南大学

## 成员

- 2401_87688128 (2401_87688128): 提交者

## 算子: op_03_relu

基于 Ascend C 实现了逐元素 ReLU 算子 y = max(0, x)，支持 float16/float32 和 ND 格式输入输出。Host 侧 ReluTilingFunc根据输入元素数计算 totalNum、blockFactor和 ubFactor，按“每核约 2048 个元素”确定启动核数，并对 blockFactor 做 32 字节对齐，从而保证 DataCopy数据搬运正确；Kernel 侧采用 Init → Process → CopyIn → Compute → CopyOut 的经典结构，通过 TQue&lt;VECIN, 2&gt; 和 TQue&lt;VECOUT, 2&gt;管理片上缓冲区，使用 Maxs完成 max(0, x) 计算，最终把结果从 GM 搬回并输出。
