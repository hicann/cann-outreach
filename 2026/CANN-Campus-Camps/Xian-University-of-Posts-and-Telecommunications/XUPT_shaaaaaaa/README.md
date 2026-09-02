# 西安邮电大学

## 团队信息

- 提交者: shaaaaaaa
- 身份: 学生
- 单位: 西安邮电大学

## 成员

- shaaaaaaa (shaaaaaaa): 提交者

## 算子: op_01_sub

本算子基于 Ascend C 三级流水编程范式实现，整体分为 Host 侧 Tiling 调度 与 Device 侧核函数计算 两部分。

Host 侧（run_kernel）：张量形状固定为 (8, 2048)，共 16384 个 float32 元素。根据运行时查询到的矢量核数，将核数对齐为不超过它的最大 2 的幂（保证数据整除无尾块），随后均分数据得到单核处理长度 blockLength；核内再按 256 元素/块切分为 tileNum 个分块，连同总长度打包进 SubCustomTilingData 结构体下发给核函数。

Device 侧（KernelSub 类）：每个核通过 GetBlockIdx() 获取核号并偏移基址，只处理自己分到的数据段。采用**双缓冲队列（TQue，深度 2）**组织三级流水：

CopyIn：DataCopy 将 GM 上的 x、y 分块搬入 UB（矢量队列 VECIN）；
Compute：调用矢量指令 Sub(z, x, y, count) 逐元素计算 z = x − y（VECOUT）；
CopyOut：将结果从 UB 搬回 GM。
队列深度为 2 使搬运与计算在不同 stage 上重叠执行，流水并行掩盖访存延迟。分块粒度 256×4B = 1KB，满足 UB 数据 32 字节对齐约束。核函数以 __global__ __vector__ 声明，运行于矢量核，&lt;&lt;&lt;coreNum, stream&gt;&gt;&gt; 多核并行，各核独立处理各自数据段后由 Host 侧同步完成整个 (8, 2048) 张量的减法输出。
