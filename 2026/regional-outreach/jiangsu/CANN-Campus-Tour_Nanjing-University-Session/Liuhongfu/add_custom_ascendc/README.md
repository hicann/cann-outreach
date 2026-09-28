# add_custom —— Ascend C 自定义算子

对两个输入张量做逐元素加法的昇腾自定义算子，目标平台 **Ascend 910B / Atlas 800T A2**（CANN 8.x）。

| 项目 | 内容 |
| --- | --- |
| 算子名称 | `AddCustom` |
| 数学公式 | `z = x + y` |
| 输入 | `x`、`y`：float16，ND，2 维 `[N2, N1]` |
| 输出 | `z`：float16，ND，2 维 `[N2, N1]` |
| 支持调用方式 | GE 图模式、aclnn 两段式单算子接口 |

## 目录结构

```
add_custom_ascendc/
├── add_custom.json                 算子原型定义（msopgen 输入）
├── op_host/
│   ├── add_custom.cpp              算子原型注册 + InferShape/InferDataType + Tiling
│   └── add_custom_tiling.h         TilingData 结构（host / kernel 共用）
├── op_kernel/
│   └── add_custom.cpp              Ascend C kernel 实现
├── op_api/
│   ├── aclnn_add_custom.h          对外两段式接口声明
│   ├── aclnn_add_custom.cpp        对外两段式接口实现
│   ├── add_custom.h                内部 level-0 接口声明
│   └── add_custom.cpp              内部 level-0 接口实现
├── test/
│   ├── utils.h                     fp16 转换 / 文件读写 / 结果比对（自实现，无外部依赖）
│   ├── test_aclnn_add_custom.cpp   aclnn 两段式调用 + 精度校验
│   └── test_ge_graph_add_custom.cpp GE 单算子图调用 + 精度校验
└── scripts/
    ├── gen_project.sh              用 msopgen 生成与本地 CANN 匹配的工程骨架并覆盖源码
    ├── build.sh                    生成工程 -> 编译安装算子包 -> 编译样例
    └── run.sh                      多组 shape 跑样例并校验
```

## 快速开始

```bash
# 0. 准备 CANN 环境
source /usr/local/Ascend/ascend-toolkit/set_env.sh

# 1. 一键构建（SOC_VERSION 按实际型号填，Atlas 800T A2 一般是 Ascend910B2）
./scripts/build.sh Ascend910B2

# 2. 跑测试（默认 device 0）
./scripts/run.sh 0
```

关于 `SOC_VERSION`：用 `npu-smi info` 确认芯片型号。Ascend 910B 系列常见取值有
`Ascend910B1` / `Ascend910B2` / `Ascend910B3` / `Ascend910B4`，选错会导致编译期
报找不到 `ascend910b` 配置或运行期 kernel 找不到。

## 实现要点

**Tiling 策略**。Host 侧把整张张量按"tile"而非"元素"切分。单 tile 元素数由 UB 容量反推：
UB 中同时驻留 `BUFFER_NUM(2) × 3` 块 tile（`x`、`y` 各双缓冲，`z` 双缓冲），
所以 `tileNum = ubSize / (2 × 3 × 2B)`，再向下对齐到 32B（fp16 即 16 个元素）。
`tileCount = ceil(totalLength / tileNum)`，然后按 `perCoreTile` 均分给各 AI Core。

按 tile 粒度切分的好处是**天然对齐**：每个 tile 的起始地址都是 32B 的整数倍，
唯一可能不足 32B 的是最后一个 tile 的长度，只需在 kernel 里用 `DataCopyPad` 统一搬运，
不用为头部/尾部写特判分支。

**核数选择**。本算子只用 Vector 单元，Host 侧用 `GetCoreNumAiv()` 取 AIV 核数，
kernel 侧用 `KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY)` 声明为纯 Vector 任务。
这两者必须配套：若 kernel 不声明，910B 会按 MIX（AIC+AIV）模式下发，
`GetBlockIdx()` 的取值域变成 `2 × blockDim`，与 Host 侧设置的 `blockDim` 不匹配，
结果会错位。

**流水**。`TQue` + `BUFFER_NUM = 2` 做双缓冲，`CopyIn/Compute/CopyOut` 三段式循环中
`AllocTensor` 在双缓冲都被占用时才会阻塞，因此 MTE2 搬入下一块、Vector 算当前块、
MTE3 搬出上一块三者可以重叠，不需要手写 event 同步。

**边界情况**。`totalLength == 0` 时 Host 侧把 tiling 全置 0，kernel 侧 `loopCount == 0`
直接返回；`tileCount` 不能被核数整除时，靠后的核 `loopCount == 0` 也会直接跳过，
不会越界访问 GM。

## 构建脚本为什么走 msopgen

`scripts/gen_project.sh` 先用 `msopgen gen` 生成官方工程骨架（`cmake/` 下的构建脚本
随 CANN 版本变化，手写很难跨版本兼容），再把本目录的 `op_host/`、`op_kernel/`、`op_api/`
源码覆盖进去。如果你的 CANN 版本生成的骨架里没有 `op_api` 目录，脚本会给出提示，
此时 aclnn 接口不会被编进算子包，需要按 CANN 文档的 aclnn 算子工程流程单独接入
（`op_api/` 下 4 个文件可直接复用）。

若你已经有现成的算子工程，也可以跳过脚本，直接把这 4 个目录里的文件按同名路径放进去。

## 依赖与版本说明

算子实现本身只用 `kernel_operator.h` 的标准接口（`TPipe` / `TQue` / `DataCopyPad` /
`Add`），CANN 7.0 以上均可编译。`op_api/` 下的 aclnn 实现依赖 CANN 8.x 的
`aclnn_kernels` 与 `opdev` 头文件；不同小版本里 `view_copy.h`、`CommonOpExecutorRun`
的声明位置可能略有差异，编译报错时按提示调整 include 即可。

`test/utils.h` 中的 fp16 转换是自实现的，不依赖 CANN 里版本不一的 `aclFloatToFloat16`，
因此样例程序在只有 ACL 运行时的环境下也能编译。
