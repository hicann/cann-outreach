# gcd123 算子工程（Ascend C / CANN 自定义算子）

基于 **Ascend C 编程框架**实现的 4 维 broadcast 最大公约数（GCD）自定义算子。

## 1. 算子规格

| 项目 | 说明 |
| --- | --- |
| 算子类型（OpType） | `Gcd123` |
| 核函数名 | `gcd123_custom` |
| 输入 | `self`（fp16，ND）、`other`（fp16，ND） |
| 输出 | `out`（fp16，ND） |
| 维数 | 4 维，shape 形如 `[N4, N3, N2, N1]` |
| shape 约束 | `self` 与 `other` 满足 broadcast 关系；`out` 的 shape 为二者 broadcast 后的 shape |
| 数据类型 | `float16`（fp16） |
| 数据格式 | `ND` |

**计算语义**（与整数 gcd 一致）：

```
gcd(a, b) = |a|                    若 b == 0
          = gcd(b, fmod(a, b))     若 b != 0
gcd(0, 0) = 0
```

> 说明：标准 gcd 定义在整数域上，本算子按需求使用 fp16 存储。对 **|x| ≤ 2048 的整数输入**（fp16 可精确表示的整数范围）结果**精确**；输入超出该范围或为非整数时按浮点精度近似，见第 6 节。

## 2. 目录结构

```
gcd123/
├── gcd123.json                  # 算子原型定义（msopgen 生成算子工程的输入）
├── CMakePresets.json            # 编译配置（需按环境修改 ASCEND_CANN_PACKAGE_PATH）
├── op_host/                     # host 侧实现
│   ├── gcd123_tiling.h          # Tiling 数据结构定义（host / kernel 共用）
│   └── gcd123.cpp               # 算子原型注册 + broadcast shape 推导 + Tiling 计算
├── op_kernel/                   # kernel 侧实现
│   └── gcd123.cpp               # Ascend C 核函数（broadcast 搬运 + 矢量 GCD）
├── kernel_invocation/           # 直接调用（Kernel Launch）验证工程，免装算子包
│   ├── main.cpp                 # host 测试程序（CPU 仿真 / NPU 真机两种模式）
│   ├── CMakeLists.txt           # ccec（NPU）/ tikicpulib（CPU 仿真）双目标编译
│   ├── run.sh                   # 一键：生成数据 → 编译 → 运行 → 比对结果
│   ├── gen_data_and_tiling.py   # 生成输入 / 黄金输出 / tiling.bin
│   ├── gcd123_tiling_def.h      # tiling 结构体定义（host 侧）
│   └── gcd123_tiling_direct.h   # tiling 解析宏（kernel 侧）
├── scripts/
│   ├── gen_data.py              # 生成输入与黄金输出数据（7 组内置用例）
│   └── verify_result.py         # 输出与黄金数据比对
└── st/
    └── gcd123_broadcast_case.json   # ST 测试用例示例（shape / 文件路径按需修改）
```

## 3. 环境要求

- CANN 8.0.RC2 及以上（`ascend-toolkit` 开发套件），已执行 `source set_env.sh`；
- 编译机安装 cmake ≥ 3.19；
- NPU 真机模式：Atlas 800I A2（Ascend 910B）等昇腾 AI 处理器；
- CPU 仿真模式：x86 Linux 上安装 CANN 开发套件即可，**无需 NPU 硬件**。

## 4. 怎么运行

### 方式 A：直接调用验证（推荐先跑这个，最简单）

不需要安装算子包，用 aclrt 直接拉起核函数，编译快、便于调试，支持 **CPU 仿真** 和 **NPU 真机** 两种模式：

```bash
cd kernel_invocation

# 1. 生成数据（输入/黄金输出/tiling）。默认 shape：[1,3,4,1] x [2,1,4,8]
python3 gen_data_and_tiling.py

# 2. 一键编译并运行（默认 ascend910b + npu 真机）
bash run.sh ascend910b npu

# 或者：CPU 仿真模式（没有 NPU 卡也能跑，用于功能调试）
bash run.sh ascend910b cpu
```

运行成功会打印 `test pass: 192 elements all match golden`。换 shape 测试：

```bash
python3 gen_data_and_tiling.py --self 1,1,1,4096 --other 1,1,1,1   # N1>2048 行内切块
bash run.sh ascend910b npu
```

> 芯片适配：`run.sh` 支持 `ascend910 / ascend310p / ascend910b`。若你的 CANN 版本中 ccec 架构参数（`--cce-aicore-arch`）或 tikicpulib 目标名与脚本中不一致，按编译报错提示微调 `kernel_invocation/CMakeLists.txt` 即可。

### 方式 B：正式算子工程流程（msopgen 工程 + 安装算子包 + ST 测试）

### 4.1 生成算子工程

用 msOpGen 基于算子原型 json 生成算子工程（生成 `build.sh`、顶层 `CMakeLists.txt`、`op_host/op_kernel` 目录下 CMakeLists.txt 等模板文件）：

```bash
${INSTALL_DIR}/python/site-packages/bin/msopgen gen \
    -i gcd123.json -f aclnn -c ai_core-ascend910b -lan cpp -out .
```

> `-c` 的计算资源按实际芯片填写（`npu-smi info` 查询 Name，如 `ascend910b`、`ascend310p`）；
> 不同 CANN 版本的 msopgen 参数略有差异，以 `msopgen gen --help` 为准。
> 若生成目录结构与预期不符，可将本工程的 `op_host/`、`op_kernel/` 文件直接覆盖到生成的同名目录下。

### 4.2 替换算子实现文件

用本工程的实现覆盖生成工程中的同名文件：

```bash
cp op_host/gcd123_tiling.h <工程>/op_host/
cp op_host/gcd123.cpp        <工程>/op_host/
cp op_kernel/gcd123.cpp      <工程>/op_kernel/
```

### 4.3 修改编译配置

编辑工程根目录 `CMakePresets.json`：

- `ASCEND_CANN_PACKAGE_PATH`：CANN 软件包安装路径（如 `/usr/local/Ascend/latest`）；
- `ASCEND_COMPUTE_UNIT`：目标芯片（如 `ascend910b`）。

### 4.4 编译并部署算子包

```bash
./build.sh                                  # 编译，生成 build_out/custom_opp_<os>_<arch>.run
./build_out/custom_opp_<os>_<arch>.run      # 安装算子包到 OPP 算子库
```

### 4.5 ST 测试（msOpST）

```bash
# 1) 基于 host 实现生成 ST 测试用例
${INSTALL_DIR}/python/site-packages/bin/msopst create -i op_host/gcd123.cpp -out ./st

# 2) 编辑 st/<用例>.json，按需填写 shape / 输入输出文件名
#    （可参考 st/gcd123_broadcast_case.json）

# 3) 生成测试数据（输入 + 黄金输出）
python3 scripts/gen_data.py --self 1,3,4,1 --other 2,1,4,8 --outdir ./st/out/test_data

# 4) 执行 ST 测试
export DDK_PATH=${INSTALL_DIR}
export NPU_HOST_LIB=${INSTALL_DIR}/<arch-os>/devlib
${INSTALL_DIR}/python/site-packages/bin/msopst run \
    -i st/<用例>.json -soc Ascend910B -out ./st/out

# 5) 校验结果
python3 scripts/verify_result.py --out ./st/out/.../out.bin --golden ./test_data/golden.bin
```

`scripts/gen_data.py` 内置 7 组用例（双端 broadcast、shape 一致、N1>2048 行内切块、标量 broadcast、N1 非 16 对齐等），直接 `python3 scripts/gen_data.py` 即可全量生成。

## 5. 实现原理

### 5.1 Broadcast 规则与 stride 计算（host 侧）

- 4 维 shape `[N4,N3,N2,N1]` 从外到内逐维比较：相等取该值；一方为 1 取另一方；否则不可广播（InferShape / Tiling 均会校验并报错）。
- 对每个输入计算**有效 stride**：某维被广播（该维 size == 1）时 stride 置 0，否则使用该输入自身连续排布的 stride。
  例：`self=[1,3,4,1]`、`out=[2,3,4,8]` → self 有效 stride `[0,4,1,0]`，输出坐标 `(i4,i3,i2,i1)` 映射到 self 的线性偏移 `i3*4 + i2*1`。

### 5.2 多核切分（Tiling）

- 输出按 1 维展平后按核切分：`blockDim = min(8, ceil(totalLength / 2048))`；
- 每个核处理 `coreLen` 个连续元素，tile 大小 ≤ 2048（UB 容量约束）；
- `N1 < 2048` 时按**整行 tile** 切分（行数 = `min(2048/pitch, N2)`）；`N1 ≥ 2048` 时按 **2048 元素的行内切块**处理。

### 5.3 Broadcast 数据搬运（kernel 侧，无需 Gather）

tile 起点 4 维坐标由标量单元维护（跨核起点做一次除法分解，之后每 tile 增量推进并处理维度进位）。输入偏移 = 坐标点乘有效 stride。行对齐模式下输入行只有 4 种模式：

| s3（最内维 stride） | s2（dim2 stride） | 搬运方式 |
| --- | --- | --- |
| 1 | N1 | 整段连续 DataCopy（无广播快路径） |
| 1 | 0 | 搬入一行后逐行 UB 内复制（dim2 广播） |
| 0 | 0 | 读 1 个元素后 Duplicate 填充整 tile（标量广播） |
| 0 | 1 | 行值在源中连续，先搬行值向量再逐行 Duplicate（dim3 广播） |

UB 中每行按 32B 对齐存放（`pitch = AlignUp(N1, 16)`），保证 Duplicate / 矢量指令操作数满足 32 字节对齐约束；搬出时逐行写回 GM 连续地址。全部使用 DataCopy / Duplicate / GetValue 等基础指令，无版本差异风险。

### 5.4 GCD 计算（kernel 侧，全矢量）

- `a = |self|`，`b = |other|`；固定 **64 次迭代**（欧几里得算法，无数据相关分支，便于矢量流水）：
  1. `safeB = (b==0) ? 1 : b`（Compare + Select 做掩码，避免除零）；
  2. `q = trunc(a / safeB)`（矢量 Div 后 Cast 到 int32 截断，再转回 half）；
  3. `r = a - b*q`，随后修正到 `[0, b)`：`r<0` 时 `+b`，`r≥b` 时 `-b`；
  4. `(a, b) ← (b==0 ? a : b, b==0 ? 0 : r)`。
- 收敛后 `a` 即结果；`gcd(a,0)=|a|`、`gcd(0,0)=0` 均由掩码路径保证。

### 5.5 UB 规划

tile ≤ 2048 元素：15 个 half 缓冲（输入 ×2、ping-pong 4、q/t/r/r±b/safeB 等）+ 1 个 int32 缓冲 + 3 个 uint8 掩码缓冲，合计约 76KB，远小于 910B 的 192KB UB。

## 6. 精度说明与限制

| 项目 | 说明 |
| --- | --- |
| 精确范围 | 输入为 **|x| ≤ 2048 的整数**（fp16 精确整数域）时结果精确 |
| 近似情形 | 非整数或超范围输入按浮点精度近似（GCD 本身定义于整数域，建议按整数语义使用） |
| shape | 4 维静态 shape（动态 shape 已做推导透传，但 Tiling 暂不支持 -1 维） |
| 性能 | 模板实现优先保证正确性与可读性；如需极致性能可自行增加双缓冲流水线、调大 blockDim 等 |

## 7. 常见问题

1. **编译报 `ccec` / `bisheng` 找不到**：确认 `ASCEND_CANN_PACKAGE_PATH` 正确、已 source `set_env.sh`。
2. **`custom_opp_*.run` 安装后调用报找不到算子**：确认安装的是同一 CANN 环境；指定 `--install-path` 安装时需 `source <path>/vendors/customize/bin/set_env.bash`。
3. **ST 测试报 shape 错误**：确认用例 json 中两个输入的 shape 可广播，且与 gen_data.py 生成的数据一致。
4. **芯片型号不对**：`msopgen -c`、`msopst -soc`、`CMakePresets.json` 的 `ASCEND_COMPUTE_UNIT` 三处需与实际芯片一致（如 `Ascend910B`）。
