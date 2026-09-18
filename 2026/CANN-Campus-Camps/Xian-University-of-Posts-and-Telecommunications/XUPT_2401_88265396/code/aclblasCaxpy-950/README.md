# aclblasCaxpy（Ascend 950）社区任务代码

本目录是课程仓的个人代码交付件，和同级的 `aclblasCaxpy_design.md` 设计文档分开提交。

实现内容：

- complex64：`y = alpha * x + y`；
- `n < 0`、空 handle、空指针、零步长的错误码校验；
- `n == 0` 的成功空操作；
- 连续、非连续和正负 stride；
- 连续大规模数据的 AIV 向量快速路径，以及非连续、负 stride 和尾块的 SIMT 回退路径；
- 异步 stream 语义。

本次实现的官方代码 PR：

https://gitcode.com/cann/ops-blas/merge_requests/436

对应提交：以该 PR 的最新提交为准。

官方 PR 还包含 CSV 驱动的 GTest、CPU golden 和 NPU wrapper。课程仓保存核心算子源码，以避免复制整个 `ops-blas` 仓库。

## 验证状态

已在 CANN 9.1.0、Ascend 950PR 环境实际构建通过；`caxpy_test` 的 15/15 项功能测试通过，覆盖参数校验、正负 stride、尾块和大规模连续数据。

已使用临时性能驱动完成 60 次预热、100 次 ACL event 计时。连续 complex64、`incx=incy=1` 条件下：`n=1048576` 为 9.391 us（门槛 14.53 us），`n=2097152` 为 14.329 us（门槛 28.36 us），`n=4194304` 为 26.048 us（门槛 89.76 us），三档均达标。该临时驱动未提交到官方 PR，以遵守仓库对非标准性能测试的提交规范。
