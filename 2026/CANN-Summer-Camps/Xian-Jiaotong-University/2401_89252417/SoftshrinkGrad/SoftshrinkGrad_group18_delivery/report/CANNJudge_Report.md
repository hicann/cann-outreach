# SoftshrinkGrad CANNJudge 验收报告

## 基本信息

| 项目 | 内容 |
| --- | --- |
| 算子名称 | SoftshrinkGrad |
| CANNJudge 提交 ID | 104235 |
| 提交人账号 | 17354011239 |
| 提交时间 | 2026-07-22 14:49:06 |
| 提交状态 | Pass |
| 测试点总数 | 50 |
| 通过测试点 | 50 |
| 输出错误占比 | 全部为 0.00% |
| 算子代码提交 | `46139556` |

## 验收结果

CANNJudge 对 SoftshrinkGrad 工程完成了编译、功能精度和性能测试。
本次提交的 50 个测试点全部通过，通过率为 100%，所有测试点输出错误占比
均为 0.00%。

测试点运行用时范围为 3.74μs 至 614.00μs，平均运行用时为 28.51μs。
每个测试点的本次用时、最优用时及比值详见
`SoftshrinkGrad_CANNJudge_Report.xlsx`。

## 代码范围

材料包内 `code/` 目录对应 `ops-nn` PR #7824 的
`experimental/activation/soft_shrink_grad`，包含：

- 算子注册与 Shape 推导；
- Host 侧 Tiling 实现；
- Ascend C Kernel 实现；
- Tiling 数据结构和 Tiling Key；
- aclnn 调用示例；
- 算子 README 与 CMake 配置。

代码支持 FP16、FP32、BF16 和 ND 格式，采用多核分块、UB 分块和双缓冲
处理泛化 Shape。

## 结论

SoftshrinkGrad 算子通过 CANNJudge 平台全部 50 个测试点，输出错误占比均为
0.00%，满足本次任务的平台精度验收要求。通过截图位于
`../evidence/cannjudge_pass.png`。

## 来源

- CANNJudge：https://cannjudge.cn/cann_2026_xjd/xjd_2026_0613/softshrinkgrad/submission/6a6067e21336c465ba5ea816
- 算子代码 PR：https://gitcode.com/cann/ops-nn/pull/7824
- 需求 Issue：https://gitcode.com/cann/ops-nn/issues/4258

