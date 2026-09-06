# 北京邮电大学

## 团队信息

- 提交者: Stephen
- 身份: 学生
- 单位: 北京邮电大学

## 成员

- Stephen (transient_coder): 提交者

## 算子: op_01_sub

实现课堂 Sub 算子 z=x-y，补全 SubCustomTilingData、KernelSub、核函数入口及 Host 侧启动函数。按 32 字节对齐块进行多核负载分配，通过单缓冲队列完成数据搬入、AscendC::Sub 计算和结果搬出。CANNJudge 提交 201401 已通过 1/1 测试点，适用范围为题面规定的 (8, 2048)、float32/float16、ND 输入。
