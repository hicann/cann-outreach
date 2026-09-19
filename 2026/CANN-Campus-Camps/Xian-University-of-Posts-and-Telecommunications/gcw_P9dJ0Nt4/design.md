# aclblasCaxpy 950 算子开发任务书

* | 项目 | 内容 |
* | --- | --- |
* | 算子名称 | `acblascaxpy` |
* | 提交者 | `leezui`|
* | 单位 | 西安邮电大学 |
* | 目标硬件 | Ascend 950PR |
* | 软件版本 | CANN 9.1.0 |
* | 数据类型 | COMPLEX64（实部、虚部均为 FLOAT32） |
* | 开发方式 | Ascend C Kernel 直调，句柄绑定 Stream |
* | 代码规划目录 | `blas/caxpy/arch/950pr/` |
* | 文档状态 | 设计完成，代码与真机数据待实现和验证 |

# 需求背景（required）

## 需求来源
### aclblasCaxpy算子实现优化
基于昇腾 NPU（Ascend 950PR）使用 Ascend C/CATLASS 编程语言实现单精度复数向量线性组合更新算子 `aclblasCaxpy`，语义对标 cuBLAS `cublasCaxpy` 与 Netlib BLAS `caxpy`，完成算子设计、开发、测试全流程工作，验收通过后合入昇腾算子开源仓。

参考链接：
- ops-blas 开源仓：https://gitcode.com/cann/ops-blas
- Netlib BLAS caxpy 参考实现：https://www.netlib.org/blas/caxpy.f
- cuBLAS cublasCaxpy 参考文档：https://docs.nvidia.com/cuda/cublas/index.html#cublas-t-axpy

### 算子现状分析
本项目为新增 BLAS 算子开发。ops-blas 仓当前 caxpy 仅有硬编码冒烟测试（`test/axpy/caxpy/arch22/caxpy_test.cpp`），尚无 CSV 驱动测试工程与 950PR（arch35）实现。本任务需开发 `aclblasCaxpy` 句柄式 BLAS 接口，代码放 `blas/axpy/arch35/`。

| 参数 | 参数含义 | 输入/输出 | 数据类型 | 支持数据类型 | 约束 | 形状 |
| --- | --- | --- | --- | --- | --- | --- |
| handle | 库上下文句柄，携带 stream | input | scalar | - | 有效句柄 | - |
| n | 向量元素个数 | input | scalar | int | n ≥ 0 | - |
| alpha | 复数标量指针 | input | scalar | COMPLEX64 | 实/虚部 FLOAT32 全集 | - |
| x | 复数向量，Device 内存只读 | input | tensor | COMPLEX64 | 物理长 1+(n-1)*\|incx\| | 逻辑一维 [n] |
| incx | x 元素间步长 | input | scalar | int | incx ≠ 0 | - |
| y | 复数向量，原地更新 alpha*x+y | output | tensor | COMPLEX64 | 物理长 1+(n-1)*\|incy\| | 逻辑一维 [n] |
| incy | y 元素间步长 | input | scalar | int | incy ≠ 0 | - |

计算公式：`y[j] = alpha * x[k] + y[j]`（i=1..n，k=1+(i-1)*incx，j=1+(i-1)*incy，1-based 索引兼容 Fortran）

### 算子功能分析
- 功能：单精度复数向量线性组合更新，y 原地更新
- 复数乘法语义：`(a+bi)(c+di) = (ac-bd) + (ad+bc)i`
- 复数加法：实部/虚部分别相加
- 输入：alpha、x、y；输出：y（原地）
- 支持数据类型：COMPLEX64（实/虚部各 float32）
- 广播支持：不涉及，本算子为两个一维向量的逐元素线性组合

# 需求分析（required）
## 需求描述
使用 Ascend C kernel 直调方式，基于 ops-blas 开源仓工程框架，实现 `aclblasCaxpy` 句柄式 BLAS 接口，通过 handle 绑定 stream 直调 NPU kernel，实现代码放在 `blas/axpy/arch35/`，接口声明放入 `include/cann_ops_blas.h`，禁止定义 950PR 私有平行接口。

## 需求拆解
1. 实现 `aclblasCaxpy(handle, n, alpha, x, incx, y, incy)` 接口，参数语义同 cublasCaxpy
2. 支持 COMPLEX64 数据类型，实部/虚部分别计算
3. 支持 incx/incy 正负步长非连续访问
4. 参数合法性校验（n≥0；incx≠0；incy≠0；空指针处理）
5. 精度达到生态算子开源精度标准
6. 性能达到任务书标杆耗时

# 详细设计（required）
## 算子分析
### 数学公式
```
y[j] = alpha * x[k] + y[j]
其中：i = 1..n，k = 1+(i-1)*incx，j = 1+(i-1)*incy
复数乘法：(a+bi)(c+di) = (ac-bd) + (ad+bc)i
```

### 支持数据类型
COMPLEX64（单精度复数，实部/虚部各 float32）

### 支持形状
逻辑一维向量 [n]，物理长度 `1+(n-1)*|inc|`；非连续访问由 incx/incy 表达，不支持额外 leading dimension padding；不涉及广播。

## 算子实现
### 实现方案
#### 3.2.1 host侧设计：
采用 Ascend C kernel 直调方式，host 侧通过 handle 绑定 stream 直调 NPU kernel。

**参数校验**：
按接口规格逐项校验，异常行为返回对应状态码：
- handle 为 nullptr → `ACLBLAS_STATUS_HANDLE_IS_NULLPTR`
- n < 0 → `ACLBLAS_STATUS_INVALID_VALUE`
- n > 0 且 alpha/x/y 为 nullptr → `ACLBLAS_STATUS_INVALID_VALUE`
- incx = 0 或 incy = 0 → `ACLBLAS_STATUS_INVALID_VALUE`
- n = 0 为合法 no-op，直接返回成功且不修改 y

**tiling策略**：
算子计算不感知维度，host 侧将 x、y 视为一维向量，仅处理元素个数 n 与步长 incx/incy。将 n 个元素按 AI Core 数量均分，每 Core 处理 `blockLength = ceil(n / blockNum)`，最后一个 block 处理剩余部分。

**任务均分**：
coreNum 根据输入长度 n 与块大小动态调整，优先满核使用；核间不能均分时，将余出数据块分配到前几个核。

**批量搬运**：
通过 `tileBlockNum`、`tileDataNum` 计算单次搬运数据量，将多次搬运合并为批量操作；尾块处理逻辑确保不完整块合并进计算流程，避免数据碎片。步长处理：incx/incy ≠ 1 时按 `1+(i-1)*|inc|` 索引，支持负步长反向取数。

**异步调度**：
通过 `aclblasSetStream` 将 kernel 提交到 stream，host 不阻塞等待；读回 Device 结果前调用 stream 同步。

#### 3.2.2 kernel侧设计：
进行 Init 和 Process 两个阶段，其中 Process 包括数据搬入（CopyIn）、计算（Compute）、搬出（CopyOut）三个阶段。

1. **CopyIn（数据搬入）**：GlobalMemory → LocalTensor(UB)，按 tiling 分块 DataCopy；获取 block 编号 `blockIdx`、总 block 数 `blockDim`，计算本 block 起始偏移与元素数；处理 incx/incy 非连续索引与负步长。
2. **Compute（计算）**：UB 内执行复数乘加：
   - 取 alpha 实部 ar、虚部 ai；x 实部 xr、虚部 xi；y 实部 yr、虚部 yi
   - 结果实部：`outRe = ar*xr - ai*xi + yr`
   - 结果虚部：`outIm = ar*xi + ai*xr + yi`
3. **CopyOut（数据搬出）**：LocalTensor(UB) → GlobalMemory 写回 y。

**关键技术选择**：
- 数据搬运用 DataCopy 批量搬入 UB，减少 GM 反复访问；
- 采用多级迭代 + double buffer 流水线，隐藏搬运与计算时延；
- 复数实虚部分离标量计算，保证正确性；
- 按 vector 对齐切分，满足 950 内存对齐约束。

## 支持硬件
| 支持的芯片版本 | 涉及勾选 |
| --- | --- |
| Ascend 950PR（arch35） | √ |
| CANN 版本 | 9.1.0 |

## 算子约束限制
1. 不支持额外 leading dimension padding 场景；向量非连续由 incx/incy 表达；
2. 不涉及 broadcast；
3. 不要求 dynamic shape（n 为运行时入参）；
4. y 原地更新，不返回视图；
5. 不要求确定性计算；
6. 异步执行依赖 `aclblasSetStream` 绑定 stream，读回结果前须同步 stream。

# 可维可测分析
## 精度标准/性能标准
| 验收标准 | 描述(不涉及说明原因) | 标准来源 |
| --- | --- | --- |
| 精度标准 | 实部/虚部分别按 FLOAT32 判定，rtol=2⁻¹⁰≈9.77e-4，atol=2⁻¹⁶≈1.53e-5，matched_ratio≥0.99，max_abs_error≤1e-2 或 32×ULP；golden 由 cblas（Netlib BLAS caxpy）生成 | 生态算子开源精度标准 |
| 性能标准 | 先 warmup 再有效采样>50 次取平均，平均单次耗时 ≤ 标杆耗时 | 任务书 §3.3 |

性能目标（Avg time，us）：
| case | n | incx | incy | 标杆耗时(us) |
| --- | --- | --- | --- | --- |
| 1 | 1048576 | 1 | 1 | 14.53 |
| 2 | 2097152 | 1 | 1 | 28.36 |
| 3 | 4194304 | 1 | 1 | 89.76 |

## 兼容性分析
新增 BLAS 算子，接口声明放入 `include/cann_ops_blas.h`，供其他产品线共用，不定义 950PR 私有平行接口，不涉及存量兼容性风险。

## 测试方案
1. **用例生成**：`gen_csv.py` 固定随机种子生成 1200 条用例（1000 精度 + 200 性能）
2. **功能测试覆盖**（精度用例）：
   - L0 基础：小尺寸 × 步长 ±1/±2
   - L1 尺寸扫描：1 → 1048576（含质数、2 的幂±1、非对齐）
   - L2 步长组合：incx × incy ∈ {±1,±2,±3}
   - L3 标量特殊值：alpha = (0,0)/(1,0)/纯虚/负/大值 1e10
   - L5 填充：均匀/全零/交替/极端/Inf/NaN
   - L5b 对齐：x/y 对齐偏移组合
   - L6 边界/负向：n=0 no-op、空指针、incx=0、incy=0、负 n（期望 INVALID_VALUE）
   - EX 扩展：尺寸×步长×标量×对齐确定性采样
3. **精度验收**：`verify_accuracy.py` 编译 C++ GTest，解析逐条 PASS/FAIL，实部/虚部分别比对
4. **性能验收**：`verify_performance.py` 执行 TC_PF 用例，与 `gpu_baseline.csv` 比对
5. 精度 golden 由 cblas（Netlib BLAS 复数实现）生成

## 风险与对策
- 风险1：复数非连续（步长）访问导致计算逻辑复杂、索引计算错误
  对策：按 `1+(i-1)*|inc|` 索引，负步长反向取数，L2 步长组合用例全覆盖验证。
- 风险2：精度不达标（复数乘加舍入误差累积）
  对策：本算子为逐元素线性运算，误差来源仅为单次复数乘加浮点舍入，正常实现应远优于阈值；与 cblas golden 比对，实虚部分别判定。
- 风险3：性能不达标杆（访存带宽瓶颈）
  对策：DataCopy 批量搬运、double buffer 流水线隐藏时延，按 vector 对齐切分，多核均分任务。
- 风险4：cannsim 仿真只能做功能验证，无法做真实性能验收
  对策：功能阶段用 cannsim 仿真调试，最终性能在真实 Ascend 950PR 上验收。

## 开发计划
1. 环境部署：申请 Ascend 950PR 算力 + CANNLab CPU 仿真环境，克隆 ops-blas 源码，安装依赖
2. 设计文档：按本模板完成 design.md，提交 cann-ops-competitions 仓库评审
3. host 侧开发：接口声明、参数校验、tiling 分块、stream 绑定
4. kernel 侧开发：Init、CopyIn、Compute、CopyOut，复数乘加逻辑，步长处理
5. 仿真验证：cannsim 跑通全部功能用例（精度）
6. 真机性能调优：Ascend 950PR 上达成 §性能目标
7. 自测报告 + 提交 PR：4 个交付件齐全后提交验收合入 ops-blas
