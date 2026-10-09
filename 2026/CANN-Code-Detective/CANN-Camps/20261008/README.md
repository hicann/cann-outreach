# CANN训练营 · ReduceSumCustom —— SIMT 多线程块归约求和算子

题目目录（相对于 `cann-outreach` 仓库根目录）：`2026/CANN-Code-Detective/CANN-Camps/20261008/`。

## 题目背景

前几期题目都在 **Ascend C 的 SIMD/Vector 编程范式**下完成：`TPipe` + `TQue` 管理 UB，`CopyIn → Compute → CopyOut` 三段式流水，核内用 `GetBlockNum()` / `GetBlockIdx()` 做多核切分。本题切换到 **SIMT 编程范式**——核函数以 `__global__` 声明、用 `<<<gridDim, blockDim, dynUB, stream>>>` 启动，线程以 `threadIdx.x` / `blockIdx.x` 索引，`warpSize`（32）个线程组成一个 warp，同一 block 内的线程共享 UB 上的共享内存（`extern __ubuf__ float smem[]`）。

本题对 `1024 × 1024` 个 `float` 做**全局归约求和**。SIMD 范式下各核各算自己那份数据、互不通信；而 SIMT 的多线程块协作归约里，**多个 block 要读写同一块全局内存**（各 block 的部分和 `block_sums`），这就引入了**数据竞争（Data Race）**：NPU 上线程的执行顺序与内存访问顺序可能不一致，别的 block 可能读到尚未写好的部分和，而且这类错误是**偶发**的（参考实现中删掉内存屏障后，连续运行 60 次有 24 次失败）。怎么把这条跨 block 的链路写对，完全由你决定。

## 算子描述

```text
output[0] = Σ input[i]
```

| 项目 | 说明 |
| :--- | :--- |
| 算子名称 | `ReduceSumCustom` |
| 输入 | `input`：float32 张量，形状 `[1024 * 1024]`，ND |
| 输出 | `output`：float32 张量，形状 `[1]`，ND |
| 中间量 | `block_sums`：float32 张量，长度 = `gridDim.x`，存放各 block 的部分和 |
| 辅助量 | `counter`：`uint32` 全局计数器，host 侧已初始化为 `0`，怎么用由你决定 |
| 数据切分 | gridDim = `(1024, 1, 1)`，blockDim = `(1024, 1, 1)`，单线程处理 1 个元素 |
| 计算 | `input[i] = i % 10`，golden = `4718580` |
| 精度标准 | `abs_err < 1e-4` 且 `rel_err < 1e-4` |

## 答题准备

进入 [Hidevlab](https://hidevlab.huawei.com/online-develop-intro)，申请 **Ascend 950PR / Ascend 950DT** 算力（NPU 架构 `dav-3510`）；申请中注意备注「CANN训练营AscendC950作业算力申请」，审批通过后即可在在线 WebIDE 中开发。

进入环境后拉取题库仓库：

```bash
git clone https://gitcode.com/cann/cann-outreach.git
```

开始答题前，先从 `cann-outreach` 仓库根目录进入题目目录，并将模板复制到以本人 GitCode 用户名命名的个人目录。请将 `your_gitcode_name` 换成自己的用户名：

```bash
cd 2026/CANN-Code-Detective/CANN-Camps/20261008
cp -r Template/ReduceSumCustom your_gitcode_name/ReduceSumCustom
```

后续请在 `your_gitcode_name/ReduceSumCustom/reduce_sum_custom.asc` 中补全代码。

## 待补全任务

代码骨架位于 [Template/ReduceSumCustom/](./Template/ReduceSumCustom/) 目录，其中保留了 `block_reduce()`、`reduce_sum_custom()` 两个函数，**函数体需要你补全**：

- [reduce_sum_custom.asc](./Template/ReduceSumCustom/reduce_sum_custom.asc) —— 算子实现骨架，主机侧代码（数据准备、`<<<>>>` 启动配置、精度校验）已给出，无需修改；
- [CMakeLists.txt](./Template/ReduceSumCustom/CMakeLists.txt) —— 构建脚本，无需修改；
- [run.sh](./Template/ReduceSumCustom/run.sh) —— 编译运行脚本，一般无需修改。

本题分两步：

1. **block 内归约**：在 `block_reduce()` 中用**蝶式归约**求出各 warp 的和，再合并成本 block 的部分和；
2. **全局归约**：在 `reduce_sum_custom()` 中由**最后一个完成的 block** 读回全部部分和，汇总成最终结果。

跨 block 没有硬件同步原语，多个 block 读写 `block_sums` / `counter` 存在**数据竞争**，怎么保证这条链路正确由你决定。

## 环境准备

| 类别 | 需要的组件 |
| :--- | :--- |
| 硬件与系统 | Ascend 950PR / Ascend 950DT 系列 NPU（NPU 架构 `dav-3510`）、配套驱动和固件 |
| CANN | CANN 9.1.0 或更高版本 |
| 构建工具 | CANN 自带的 ASC 编译器（bisheng）、CMake ≥ 3.16 |
| 离线仿真（可选） | CANN 自带的 NPU Simulator（`cannsim` / `npusim`），无真实设备时可离线运行 |

本题涉及的 SIMT 接口仅在 **Ascend 950PR / Ascend 950DT**（NPU 架构 `dav-3510`）上支持，在其他芯片上编译或运行会失败。

先加载 CANN 环境；如果 CANN 不在示例路径，请改成实际的 `set_env.sh` 路径：

```bash
source "$ASCEND_TOOLKIT_HOME/set_env.sh"
```

## 运行验证

在复制出的参与者目录下运行：

```bash
# 注意：在线环境一般不用修改脚本，可根据实际情况修改脚本中激活 CANN 环境的路径
bash run.sh
```

`run.sh` 会自动执行 `cmake .. && make -j && ./reduce_sum_custom`。通过时输出：

```
Output: 4.71858e+06
Golden: 4.71858e+06
[Success] Case accuracy verification passed.
```

测试数据为 `input[i] = i % 10`（`i` 取 `0 ~ 1024*1024-1`），故 `golden = Σ (i % 10) = 4718580`。host 侧按与 kernel 一致的分块 + 两阶段归约顺序计算 golden，并用 `abs_err < 1e-4 && rel_err < 1e-4` 做容差比对（归约顺序不同会带来浮点累加差异，因此不使用严格相等比较）；本题数据均为整数且所有部分和不超过 `2²⁴`，float32 下可精确表示，任何归约顺序都应得到精确的 `4718580`。

**没有设备也能做题**：可用 CANN 自带的 NPU Simulator 做离线仿真运行（旧版本命令名为 `cannsim`，2026 年 7 月 30 日之后的版本更名为 `npusim`），它集成在 CANN toolkit 包中，**不依赖驱动和固件**：

```bash
# 1. 先编译出可执行文件（编译只需要 CANN，不需要真实设备）
source "$ASCEND_TOOLKIT_HOME/set_env.sh"
mkdir -p build && cd build && cmake .. && make -j

# 2. 用仿真器运行同一个可执行文件
cannsim record ./reduce_sum_custom -s Ascend950 -o ./output
# 新版 CANN 中命令为：npusim record ./reduce_sum_custom -s Ascend950 -o ./output
```

> 注意：直接 `bash run.sh` 会在编译完成后**立即执行**该可执行文件，在没有真实设备时会卡在最后一步并报错——编译产物本身已经生成。请按上面两步分开执行。仿真结果会打印在终端，同时写入 `./output/cannsim_*/cannsim.log`，看到 `[Success] Case accuracy verification passed.` 即通过。仿真器与板上运行**保持二进制兼容**，两种方式验证的是同一份代码，无需改动 `CMakeLists.txt` 或源码。
>
> 仿真约束：仅支持 **Ascend 950PR / Ascend 950DT** 芯片、仅支持**单卡**（代码中只能使用 **0 卡**）、仅支持 AI Core 计算类算子、**不支持 arm 环境**，建议运行环境为 16 核 CPU + 32GB 以上内存。本题（`gridDim = 1024 × blockDim = 1024`）在仿真器上约 **1 分钟**跑完。

## 提交说明

将本人 `2026/CANN-Code-Detective/CANN-Camps/20261008/<你的用户名>/ReduceSumCustom/` 下补全的代码提交为 PR：

- 提交文件只需要 `reduce_sum_custom.asc`、`CMakeLists.txt`、`run.sh` 三个文件，不要提交 `build/` 等编译产物；
- 保留 `Template/` 目录，**请勿直接修改**；
- 在 PR 描述中提供精度验证通过的截图，代码风格规范、注释清晰。

提交和 CLA 操作可参考[代码侦探活动说明](../../README.md)与 [GitCode 讨论 #287](https://gitcode.com/org/cann/discussions/287)。本题考点是 SIMT 归约本身，以及多线程块协作时怎么保证结果正确，建议先把自己的同步方案想清楚再提交；可以合理使用 AI 辅助编程，但请确保理解每一行代码的含义。

## 推荐学习资料

- [SIMT 编程简介](https://gitcode.com/cann/asc-devkit/blob/master/docs/zh/api/SIMT-API/SIMT_programming_intro/SIMT_programming_intro.md)：SIMT 线程层级（warp / block / grid）、`__global__` 与 `<<<>>>` 启动配置、`__ubuf__` 共享内存。
- [Warp 函数](https://gitcode.com/cann/asc-devkit/blob/master/docs/zh/api/SIMT-API/Warp_functions/Warp_functions_intro.md)：重点吃透 [`asc_shfl_down`](https://gitcode.com/cann/asc-devkit/blob/master/docs/zh/api/SIMT-API/Warp_functions/Warp_shfl_functions/asc_shfl_down.md) 的蝶式交换语义（越界返回自身值），以及 [`laneid`](https://gitcode.com/cann/asc-devkit/blob/master/docs/zh/api/SIMT-API/Warp_functions/lane_id_functions/laneid.md)。
- [同步与内存屏障](https://gitcode.com/cann/asc-devkit/blob/master/docs/zh/api/SIMT-API/sync_and_memory_fence/sync_and_memory_fence_intro.md)：掌握 [`asc_threadfence`](https://gitcode.com/cann/asc-devkit/blob/master/docs/zh/api/SIMT-API/sync_and_memory_fence/memory_fence/asc_threadfence.md)（核间、不阻塞）与 [`asc_syncthreads`](https://gitcode.com/cann/asc-devkit/blob/master/docs/zh/api/SIMT-API/sync_and_memory_fence/sync_interface/asc_syncthreads.md)（块内、阻塞）的区别与适用场景；原子操作见 [`asc_atomic_add`](https://gitcode.com/cann/asc-devkit/blob/master/docs/zh/api/SIMT-API/atomic_operations/asc_atomic_add.md)。

完成学习后，再回到本题目尝试独立实现。
