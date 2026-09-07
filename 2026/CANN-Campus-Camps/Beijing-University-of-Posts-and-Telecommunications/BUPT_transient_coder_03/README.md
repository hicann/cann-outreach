# 北京邮电大学

## 团队信息

- 提交者: Stephen
- 身份: 学生
- 单位: 北京邮电大学

## 成员

- Stephen (transient_coder): 提交者

## 算子: op_03_relu

第三题 ReLU 目录序号修正：原 PR #3313 已合并，个人目录 BUPT_transient_coder 缺少题号后缀，本次按题目提交说明使用 transient_coder_03。实现课堂 ReLU 算子 y=max(0,x)，补全 ReluTilingFunc 及 Init、CopyIn、Compute、CopyOut、Process，采用分核、UB 分块和双缓冲队列调用 AscendC::Relu。沿用 CANNJudge 提交 201480 的代码，原记录 Pass，2/2 测试通过，验证范围为 (8, 2048)、float32/float16、ND。关联 issue：https://gitcode.com/cann/cann-outreach/issues/360。
