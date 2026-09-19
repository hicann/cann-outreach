# Sub 算子设计文档

## 1. 概述

### 1.1 基本信息

| 项目 | 内容 |
|-----|------|
| 算子名称 | Sub（op_01_sub，算子名 `sub_custom`） |
| 算子类别 | Elementwise（逐元素二元运算） |
| 支持数据类型 | FP16 / FP32 / INT32 |
| 目标芯片 | Ascend910B |
| 目标架构 | dav-2201 |
| 开发形态 | Ascend C Kernel **直调**（Direct Invocation，无 op_host/op_kernel 分层、不打包成算子包） |
| 提交团队 | 西安邮电大学 |
| 提交者 | 刘昊（will13546） |

### 1.2 算子功能

Sub 计算两个等形状张量的逐元素差。输入 `x`、`y` 与输出 `z` 的 shape、dtype 完全一致，元素一一对应，无广播语义。

本算子实现还额外承担了「张量扁平化 + 多核切分」的职责：不论输入是几维张量，Host 侧都先按行主序展平成一维，再切分给各核，因此对任意维度、任意 shape 的输入都能工作。

### 1.3 数学公式

$$
z_i = x_i - y_i,\quad i = 0,1,\dots,N-1
$$

其中 $N$ 为展平后的元素总数。

---

## 2. 架构设计

### 2.1 逻辑视图

本算子采用**直调（Direct Invocation）**形态，不走标准的 `op_host` + `op_kernel` 双目录分层，而是把 Host 侧启动逻辑与 Device 侧核函数都放在同一个编译单元里：

| 模块 | 职责 | 核心文件 |
|------|------|---------|
| **Host 侧启动逻辑** | 展平 shape 求总长、按 32B 对齐块做多核切分、填 TilingData、启动核函数 | `kernel.asc` 中的 `run_kernel()` |
| **Device 侧核函数** | 核间按切片偏移定位、核内分 tile 双缓冲搬运与 `Sub` 计算 | `kernel.asc` 中的 `sub_custom()` 与 `KernelSub<T>` 类 |
| **验证宿主** | ACL 初始化、显存申请、bin 文件读写、流同步、结果落盘 | `main.asc` |
| **文件 I/O 工具** | `ReadFile` / `WriteFile` 二进制读写 | `data_utils.h` |

模块依赖：

```
main.asc
  ├── data_utils.h           (二进制文件读写)
  ├── acl/acl.h              (ACL 运行时)
  └── #include "kernel.asc"  (核函数与 run_kernel 一起编进同一可执行文件)
        ├── kernel_operator.h (Ascend C 编程接口)
        └── TensorGroupInfo / TensorInfo (由调用方预定义的结构体)
```

`kernel.asc` 被 `main.asc` 以 `#include` 方式引入，因此该文件**不能**包含 `main()`、`#pragma once` 或 include guard —— 这是直调形态的约定，源码注释中也做了说明。

### 2.2 开发视图

```
op_01_sub/
├── CMakeLists.txt          # find_package(ASC)，生成可执行文件 sub_custom
├── main.asc                # 本地测试宿主：造数 → 调 run_kernel → 落盘
├── kernel.asc              # 核函数 + run_kernel（Host 侧切分逻辑）
├── data_utils.h            # ReadFile / WriteFile
├── run.sh                  # 一键构建 + 生成数据 + 运行 + 精度校验
└── scripts/
    ├── sub.py              # 参考实现 impl(x, y) = x - y
    ├── gen_data.py         # 生成输入 bin 与 golden bin
    └── verify_result.py    # 输出 bin 与 golden bin 的数值比对
```

其中 `run.sh` 的流水线为：

```
[1/4] source ${ASCEND_HOME_PATH}/set_env.sh
[2/4] cmake + make  →  build/sub_custom
[3/4] python3 ../scripts/gen_data.py （在 build/ 下生成 input/、output/）
[4/4] 把 case0 的输入拷到 input/、golden 拷到 output/，运行 sub_custom，再跑 verify_result.py
```

### 2.3 运行视图

**数据流**（每个核内部）：

```
GM (Global Memory)
  │  DataCopy  (GM → UB，32B 对齐)
  ▼
UB (Unified Buffer)：inQueueX / inQueueY
  │
  │  Sub(zLocal, xLocal, yLocal, count)
  ▼
UB：outQueueZ
  │  DataCopy  (UB → GM)
  ▼
GM (Global Memory)
```

**执行流程**：

```
Host 侧 (run_kernel)：
  1. 由 info_x.tensors[0].shape 连乘得到 totalLength；任一维 ≤ 0 或总长 ≤ 0 直接返回
  2. elemSize = (dtype == 1) ? 2 : 4          // 仅区分 fp16 与「4 字节类型」
  3. chunkElems = 32 / elemSize               // 一个 32B 对齐块内的元素数
     chunks     = totalLength / chunkElems    // 只取完整的 32B 块
     chunks < 1 → 直接返回（数据太小，不启动核）
  4. blockNum = min(availableCoreNum, chunks)
     baseChunks = chunks / blockNum, remainChunks = chunks % blockNum
  5. 填写 SubCustomTilingData 并启动 sub_custom<<<blockNum, nullptr, stream>>>

Device 侧 (sub_custom → KernelSub<T>)：
  Init()：
    ├── 按 blockIdx 与 formerNum/formerLength/tailLength 算出本核的起始偏移与长度
    ├── SetGlobalBuffer 建立 xGm / yGm / zGm（偏移以元素为单位）
    └── InitBuffer：x/y/z 三个队列各 2 块缓冲，每块 tileLength × sizeof(T)
  Process()：
    └── while (offset < coreLength)
          ├── CopyIn() ：AllocTensor → DataCopy(x/y) → EnQue
          ├── Compute()：DeQue → AscendC::Sub → EnQue → FreeTensor
          └── CopyOut()：DeQue → DataCopy → FreeTensor
```

---

## 3. 实现方案

### 3.1 核间切分：按 32B 对齐块均分

多核切分的粒度不是「元素」而是「32B 对齐块」，目的是让每个核拿到的数据量都是搬运对齐单位的整数倍，从而让核内所有 `DataCopy` 天然满足 32B 对齐要求，不需要任何 padding 分支。

以 fp32 为例，`chunkElems = 32 / 4 = 8`，即 8 个元素为一个块。

切分采用**大核 + 小核**的两段式均分：

| 参数 | 含义 | 取值 |
|------|------|------|
| `blockNum` | 实际启动的核数 | `min(availableCoreNum, chunks)` |
| `baseChunks` | 每核至少分到的块数 | `chunks / blockNum` |
| `remainChunks` | 平分后剩余的块数 | `chunks % blockNum` |
| `formerNum` | 大核个数 | `remainChunks` |
| `formerLength` | 大核的元素数 | `(baseChunks + 1) × chunkElems` |
| `tailLength` | 小核的元素数 | `baseChunks × chunkElems` |

核 `k` 的起始偏移：

$$
\text{start}(k)=
\begin{cases}
k \times \text{formerLength}, & k < \text{formerNum}\\[4pt]
\text{formerNum} \times \text{formerLength} + (k-\text{formerNum}) \times \text{tailLength}, & k \ge \text{formerNum}
\end{cases}
$$

如此保证 $\text{formerNum}\times\text{formerLength} + (\text{blockNum}-\text{formerNum})\times\text{tailLength} = \text{chunks}\times\text{chunkElems}$，切片首尾相接、无重叠、无空洞。

**举例**：case0 的 shape 为 `(8, 2048)`、fp32，则 `totalLength = 16384`、`chunkElems = 8`、`chunks = 2048`。若平台上报 40 个 AIV 核，则 `blockNum = 40`、`baseChunks = 51`、`remainChunks = 8`，即前 8 个核各处理 52 块（416 元素），后 32 个核各处理 51 块（408 元素）：$8\times416 + 32\times408 = 16384$。

### 3.2 核内切分：固定 tile + 双缓冲

核内以固定的 `TILE_LENGTH = 2048` 个元素为一片循环处理，最后一片按实际剩余量收尾：

```
while (offset < coreLength) {
    count = min(coreLength - offset, tileLength);   // 尾片自动收缩
    CopyIn(offset, count); Compute(count); CopyOut(offset, count);
    offset += count;
}
```

`TILE_LENGTH` 取 2048 的理由：它是 fp32（8 元素/32B）、fp16（16 元素/32B）、int32（8 元素/32B）对齐粒度的公倍数，并且 2048 个元素 × 4B = 8 KB，三个队列双缓冲后总占用仍在 UB 容量之内（见 3.5）。

由于 `coreLength` 是 `chunkElems` 的整数倍、`TILE_LENGTH` 也是 `chunkElems` 的整数倍，尾片长度必然仍是 `chunkElems` 的整数倍 —— 这是核内不需要处理非对齐尾块的根据。

### 3.3 TilingData 结构体

**文件位置**：`kernel.asc`

```cpp
struct SubCustomTilingData {
    uint32_t totalLength;   // 展平后的总元素数
    uint32_t blockNum;      // 实际启动的核数
    uint32_t formerNum;     // 大核个数
    uint32_t formerLength;  // 大核负责的元素数
    uint32_t tailLength;    // 小核负责的元素数
    uint32_t tileLength;    // 核内单片 tile 的元素数
    int32_t  dtype;         // 元素 dtype（0=fp32 1=fp16 2=bf16 3=int8 … 5=int32 …）
};
```

该结构体作为核函数参数**按值传递**给 `sub_custom`（直调形态下不走 GM tiling buffer），因此不需要 `GET_TILING_DATA_WITH_STRUCT` 之类的宏解析。

### 3.4 数据类型分派

核函数内按 `tiling.dtype` 做三分支实例化：

```cpp
if (tiling.dtype == 1)       KernelSub<half>    op;   // fp16
else if (tiling.dtype == 5)  KernelSub<int32_t> op;   // int32
else                         KernelSub<float>   op;   // fp32（默认）
```

Host 侧同步的 `elemSize` 映射为：`dtype == 1 → 2 字节`，其余一律 `4 字节`。两处必须保持一致，源码注释中也做了强调。

### 3.5 API 映射

| 计算步骤 | Ascend C API | 关键参数 | 说明 |
|---------|-------------|---------|------|
| 队列初始化 | `TPipe::InitBuffer` | `(que, BUFFER_NUM, tileLength * sizeof(T))` | `BUFFER_NUM = 2` 双缓冲 |
| 数据搬入 | `DataCopy` | `(xLocal, xGm[offset], count)` | `count` 为元素数，天然 32B 对齐 |
| 计算 | `Sub` | `(zLocal, xLocal, yLocal, count)` | 逐元素相减 |
| 数据搬出 | `DataCopy` | `(zGm[offset], zLocal, count)` | — |
| 线程索引 | `GetBlockIdx()` | — | 决定本核切片 |
| Kernel 启动 | `<<<blockNum, nullptr, stream>>>` | — | 直调启动语法 |

### 3.6 内存管理（UB 占用）

每个核的 UB 占用 = 3 个队列 × 2 块缓冲 × `TILE_LENGTH × sizeof(T)`：

| dtype | 单块大小 | UB 占用 |
|-------|---------|---------|
| fp32 / int32 | 2048 × 4 = 8 KB | 3 × 2 × 8 KB = **48 KB** |
| fp16 | 2048 × 2 = 4 KB | 3 × 2 × 4 KB = **24 KB** |

Ascend910B 单核 UB 为 192 KB，48 KB 的占用留有充足余量，因此该 tile 尺寸在任意输入规模下都不会溢出 —— 输入变大只会增加循环次数，不增加 UB 占用。

---

## 4. 性能优化

### 4.1 并行策略

- **多核并行**：按 32B 对齐块均分到最多 `availableCoreNum` 个 AIV 核，核间负载差不超过一个对齐块。
- **核内流水**：`tileLength` 允许的情况下，`DataCopy`（MTE2/MTE3）与 `Sub`（Vector）在不同缓冲上重叠执行。

### 4.2 流水线设计

- **双缓冲**：`BUFFER_NUM = 2`，第 $i$ 片的计算与第 $i+1$ 片的搬入可并行，隐藏搬运时延。
- **一次申请、循环复用**：队列缓冲在 `Init()` 中一次性申请，`Process()` 内仅做 Alloc/DeQue/Free，不产生重复的 UB 管理开销。
- **对齐换效率**：用「按 32B 块切分」替代「按元素切分」，省掉了核内对非对齐尾块的特判分支，让热路径完全整齐。

### 4.3 可优化空间

- 当前实现「大核/小核」两种长度，负载差最大为 1 个对齐块；若追求极致均衡可改为按块交错分配（每核处理不相邻的多个块），但会牺牲访存连续性，当前方案是按「连续性优先」做的取舍。
- 未使用 `DataCopyPad`，因此不支持非 32B 对齐的尾部；若后续要支持任意长度，需要引入 `DataCopyPad` 并处理尾部 padding。

---

## 5. 风险与约束

### 5.1 功能约束

| 约束 | 表现 | 原因 / 影响 |
|------|------|------------|
| **尾部不足一个 32B 块的元素不参与计算** | `chunks = totalLength / chunkElems` 为整除，`totalLength % chunkElems` 个尾元素不会被任何核处理 | 未使用 `DataCopyPad`，为保证所有搬运对齐而做的取舍。fp32 下最多丢 7 个元素，fp16 下最多丢 15 个元素 |
| **总长小于一个 32B 块时不启动核** | `chunks < 1` 时 `run_kernel` 直接返回，输出 buffer 保持原样 | 同上；此时计算规模已无意义，但输出内容未定义 |
| **`run_kernel` 用 x 的 shape 决定总长** | `info_y`、`info_z` 被 `(void)` 显式忽略 | 直调约定下三个张量 shape 一致，不重复校验；若传入形状不一致的 y 会越界读 |
| **dtype 白名单外一律按 fp32 处理** | 传入 int8/int16/bf16 等会被当作 4 字节类型分派到 `KernelSub<float>` | Host 侧 `elemSize` 与核内 dispatch 只覆盖 fp16/fp32/int32 三种；这是明确的规格边界 |

### 5.2 精度风险

| 风险 | 说明 |
|------|------|
| 无 | 该算子为纯减法，fp32/int32 为单次精确运算；fp16 的舍入由硬件 `Sub` 指令决定，与参考实现 `x - y` 的 fp16 语义一致 |

### 5.3 校验点位

- `init.dtype` 与 `elemSize` 的映射必须同步修改，否则会出现「按 2 字节切分、按 4 字节计算」的错配。
- `TILE_LENGTH` 必须是所有支持 dtype 对齐粒度的公倍数，否则尾片会破坏对齐假设。

---

## 6. 测试方案

### 6.1 用例设计

| 项 | 值 |
|----|----|
| case 数量 | 1（case0） |
| 输入 shape | `(8, 2048)` |
| dtype | float32 |
| 数据范围 | `np.random.uniform(low=1, high=10)`，固定 `np.random.seed(42)` |
| 输入字节数 | 16384 × 4 = 65536 B（与 `main.asc` 中写死的 65536 一致） |

数据范围取 `[1, 10)` 而非含 0 或负数的区间，是为了避免减法结果落在 0 附近时相对误差判据失效。

### 6.2 参考实现与判据

参考实现（`scripts/sub.py`）：

```python
def impl(x, y):
    return x - y
```

判据（`scripts/verify_result.py`）：case0 的 `z` 按 `np.float32` 读取，要求

$$
|\text{output} - \text{golden}| \le \text{atol} + \text{rtol}\cdot|\text{golden}|,\qquad \text{rtol}=\text{atol}=10^{-4}
$$

且允许的失配比例 `tol = 1e-4`（即最多万分之一的元素可以超出上式）。比对前会校验输出与 golden 的元素数是否一致，输出偏小时截断 golden 并打印告警。

### 6.3 执行方式

```bash
export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
bash run.sh
```

`run.sh` 中 `sub_custom` 以 `timeout 120` 运行，超时或非零退出都判为失败。

---

## 7. 交付件清单

| 文件 | 说明 |
|------|------|
| `CMakeLists.txt` | 构建配置（`find_package(ASC)`，目标 `sub_custom`，默认 `--npu-arch=dav-2201`） |
| `main.asc` | 本地测试宿主 |
| `kernel.asc` | 核函数 + Host 侧切分/启动逻辑 |
| `data_utils.h` | 二进制文件读写工具 |
| `run.sh` | 一键构建/生成数据/运行/校验脚本 |
| `scripts/sub.py` | 参考实现 |
| `scripts/gen_data.py` | 测试数据与 golden 生成 |
| `scripts/verify_result.py` | 精度校验 |
| `README.md` | 提交信息 |
