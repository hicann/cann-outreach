# add_custom 算子（Ascend C 实现）

## 1. 算子定义

| 项目 | 内容 |
| --- | --- |
| 算子名称 | `add_custom` |
| 功能描述 | 对两个输入张量 `x`、`y` 执行逐元素加法，返回结果张量 `z` |
| 数学公式 | `z = x + y`（`x`、`y`、`z` 均为张量） |
| 输入 | `x[N2, N1]`、`y[N2, N1]`，数据类型 `float16`，数据格式 `ND` |
| 输出 | `z[N2, N1]`，数据类型 `float16`，数据格式 `ND` |
| shape | 2 维（`[N2, N1]`），两输入 shape 一致；Host 侧按元素总个数 `N2 * N1` 切分 |

## 2. 工程结构

```
.
├── add_custom.cpp         # 算子 Kernel 侧实现（Ascend C），包含核函数入口 add_custom
├── add_custom_tiling.h    # Tiling 参数结构体（Host / Kernel 共用）
├── main.cpp               # Host 侧主流程：初始化、数据搬运、Tiling 下发、Kernel Launch、精度比对
├── data_utils.h           # Host 侧文件读写辅助函数
├── fp16_utils.h           # Host 侧 float16 <-> float 转换（不依赖 CANN，便于 Host 编译）
├── CMakeLists.txt         # 构建脚本（ascendc_library + 可执行程序）
├── run.sh                 # 一键编译 + 运行 + 校验
└── scripts
    ├── gen_data.py        # 生成输入数据 input_x.bin / input_y.bin 与真值 golden_z.bin
    ├── verify_result.py   # 独立校验 output_z.bin 的精度
    ├── fp16_selftest.pl   # 离线自检：float16 <-> float 转换逻辑
    └── tiling_selftest.pl # 离线自检：切分逻辑的对齐/覆盖/边界不变量
```

运行期目录（脚本自动创建）：

```
input/input_x.bin     输入 x（float16，行优先连续排布）
input/input_y.bin     输入 y（float16）
input/golden_z.bin    真值 z = x + y（float16）
output/output_z.bin   算子运行结果
```

## 3. 依赖与构建

- CANN（Ascend-CANN-Toolkit）已安装并 `source` 了 `set_env.sh`，即环境变量 `ASCEND_HOME_PATH` 可用；
- 宿主机具备 `cmake >= 3.16`、`g++`（C++17）、`make`、`python3`（含 `numpy`）；
- 采用官方 **Kernel Launch 直调** 方式：`ascendc_library()` 编译算子并生成 `aclrtlaunch_add_custom.h`，Host 程序通过 `ACLRT_LAUNCH_KERNEL(add_custom)` 直接下发核函数。

一键运行：

```bash
bash run.sh [N2] [N1] [blockDim]     # 默认 8 2048 8
```

手工构建与运行：

```bash
mkdir -p input output build && cd build
cmake ..
make -j
cd ..
python3 scripts/gen_data.py 8 2048
./build/add_custom 8 2048 8
python3 scripts/verify_result.py 8 2048
```

`CMakeLists.txt` 会在 `${ASCEND_HOME_PATH}` 下按候选路径自动查找 `ascendc.cmake` 与 `libascendcl`，适配不同 CANN 版本的目录差异；也可显式指定：

```bash
cmake .. -DASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
```

不同 CANN 版本下 `ascendc_library` 的库类型（STATIC / SHARED）与宏名存在差异，构建脚本已做兼容：找不到 `ascendc_include_directories` 时自动退回普通 include 设置；若链接阶段报 `aclrtlaunch_add_custom` 相关未定义符号，可切换库类型后重新构建：

```bash
cmake .. -DADD_CUSTOM_KERNEL_LIB_TYPE=STATIC
```

## 4. 实现说明

### 4.1 多核切分（核间）

- 元素总数 `totalLength = N2 * N1`，按 `blockLength` 均分到 `blockNum` 个 AI Core；
- `blockLength` 按 32Byte（16 个 `float16`）向上对齐，保证每个核的起始地址 32Byte 对齐，从而使用高效的 `DataCopy`；
- 第 `coreIdx` 个核处理 `[coreIdx * blockLength, ...)` 区间；超出总长度的核直接返回；最后一个核数据量不足 `blockLength` 时按实际长度处理（核间尾块）；
- 数据量较小时（`< ADD_CUSTOM_MIN_ELEMS_PER_CORE * coreNum`）自动降低核数，避免单核数据过少导致流水收益不足。

### 4.2 核内切分与流水（核内）

- 核内数据再切成 `tileLength` 大小的 tile，配合 `TPipe` / `TQue` 的 double buffer（`BUFFER_NUM = 2`）形成 `CopyIn -> Compute -> CopyOut` 流水，掩盖 GM/UB 搬运开销；
- `tileLength` 由 `blockLength / ADD_CUSTOM_TILE_NUM_TARGET` 推导并 32Byte 对齐，限制在 `[256, 8192]`；上限 8192 个 `float16`（16KB）保证 3 个队列 × 2 份 buffer 共 96KB，处于 UB 容量（192KB）以内。

### 4.3 尾块与对齐

- 长度正好为 `tileLength` 的 tile 使用 `DataCopy`（`len` 是 32Byte 的整数倍）；
- 核内最后一个不足一个 tile 的尾块使用 `DataCopyPad` 精确搬运 `len * sizeof(half)` 字节，避免越界读写；
- 向量计算 `Add` 按 32Byte 对齐长度下发（`calCount = AlignUp(len, 16)`），多算出的元素停留在 UB 中、不回写 GM，不影响结果正确性。

### 4.4 数据格式 ND

逐元素加法与 shape 无关，2 维 `shape` 仅用于 Host 侧推导元素总个数；Kernel 侧对 ND 连续内存按一维处理，因此 `[N2, N1]` 的任意取值都能正确计算。

### 4.5 精度校验

- 运行前由 `scripts/gen_data.py` 生成 `golden_z.bin`（`float32` 相加后舍入回 `float16`，等价于逐元素加法的正确舍入结果）；
- `main.cpp` 运行结束后自动与 `golden_z.bin` 逐元素比对：`float16` 机器精度约 1e-3，判定门限为「相对误差 > 1e-3 且绝对误差 > 1e-3」记为误差元素，误差元素数必须为 0 才 PASS；
- `scripts/verify_result.py` 提供进程外的独立校验（同时与 `input_x.bin + input_y.bin` 现场重算的真值比对）。

### 4.6 离线自检脚本（不依赖 CANN / 编译器）

`fp16_utils.h` 的转换逻辑与 `main.cpp` 的切分逻辑均已用脚本逐行等价移植并离线验证：

```bash
perl scripts/fp16_selftest.pl       # float<->float16 转换：特殊值、次正规数、全量往返
perl scripts/tiling_selftest.pl     # 切分不变量：覆盖性、对齐性、边界、核内循环
```

两者均已通过：fp16 转换 15 组定点用例正确，且 `0x0000~0xFFFF` 全量往返（排除 Inf/NaN）稳定；切分逻辑在 20 组 shape × 5 组核数下满足「全覆盖、无重复、不越界、每核起始地址 32Byte 对齐」。

## 5. 已知限制

- 本工程在当前开发机上**未执行编译与上板运行**（该机器无 CANN 工具链、无 C++ 编译器、无可用 python），代码按官方直调样例范式编写，请在 Ascend 环境上按第 3 节步骤验证；若 `cmake ..` 报找不到 `ascendc.cmake`，请用 `-DASCEND_HOME_PATH=` 显式指定 CANN 路径；
- 当前按「两输入 shape 一致、任意 2 维 shape」实现，未做 broadcast 与 shape 校验（如需广播，需在 Host 侧按对齐后的 shape 计算索引并在 Kernel 侧引入索引搬移）；
- 仅针对 `float16` 与 `ND` 格式；若需 `float32`/`bfloat16`，仅需把 Kernel 中的 `half` 与 `ADD_CUSTOM_DTYPE_BYTE` 一并替换（对齐元素个数需按 32Byte 重新折算）。

## 6. 扩展为自定义算子工程

若需交付为算子包（`op_host` / `op_kernel` 分离、支持 `msopgen` 与图模式调用），可在本实现基础上：

1. 用 `msopgen` 生成算子工程骨架，把 `add_custom.cpp` 的类实现搬入 `op_kernel/add_custom.cpp`，把 `ComputeTiling` 的逻辑搬入 `op_host/add_custom.cpp` 的 `TilingFunc`；
2. 在 `op_host` 的算子原型（`*_op.cpp` 中的 `OpDef`）中声明输入 `x`、`y` 与输出 `z` 的 `float16` / `ND` 属性；
3. 用 `AddCustomTiling` 结构体配合 `GET_TILING_DATA`（或在 TilingFunc 中使用 `context->GetTilingData`）传递切分参数。

Kernel 侧的切分与流水逻辑无需修改。
