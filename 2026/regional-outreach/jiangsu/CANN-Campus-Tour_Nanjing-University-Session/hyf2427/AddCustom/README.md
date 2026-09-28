# add_custom 算子（Ascend C）

逐元素加法算子：`z = x + y`

| 项目 | 规格 |
|------|------|
| 算子名 | add_custom（注册类型名 `AddCustom`） |
| 输入 | `x`, `y`：2D `[N2, N1]`，`float16`，`ND` |
| 输出 | `z`：shape/dtype 同输入 |
| 框架 | Ascend C（CANN 自定义算子工程） |
| 数据切分 | 按 AI Core 均分（blockDim=8），每核 8 个 tile，双缓冲三级流水 |

## 目录结构

```
AddCustom/                  # 算子工程（模板由 msopgen 生成，以下文件为本项目实现）
├── add_custom.json         # 算子原型定义（msopgen gen 的输入）
├── op_kernel/
│   └── add_custom.cpp      # 【实现】kernel 核函数：搬入/计算/搬出三级流水
└── op_host/
    ├── add_custom_tiling.h # 【实现】Tiling 参数定义（totalLength, tileNum）
    └── add_custom.cpp      # 【实现】原型注册、InferShape、InferDataType、TilingFunc
AddCustom_st/
└── AddCustom_case.json     # msOpST 功能测试用例（[8, 2048] float16）
```

## 开发流程（在昇腾 NPU 服务器的 CANN 环境中执行）

### 1. 生成算子工程模板

```bash
# 将 add_custom.json 拷贝到服务器（例如 $HOME/sample/add_custom.json）
# ${INSTALL_DIR} 为 CANN 安装路径，如 $HOME/Ascend/ascend-toolkit/latest
# <soc_version> 通过 npu-smi info 查询 Name（如 Ascend910B）
${INSTALL_DIR}/python/site-packages/bin/msopgen gen \
    -i $HOME/sample/add_custom.json \
    -c ai_core-<soc_version> \
    -lan cpp \
    -out $HOME/sample/AddCustom
```

msopgen 生成的工程包含 `build.sh`、`cmake/`、`CMakeLists.txt`、`scripts/` 等模板文件。
**用本目录下 `op_kernel/add_custom.cpp`、`op_host/add_custom.cpp`、
`op_host/add_custom_tiling.h` 覆盖模板中对应的三个文件**（模板其余文件保持不动）。

> 若你的 CANN 版本生成的模板文件结构有差异（如 8.3/8.5 版本），
> 以模板为准，把本项目的核心逻辑（TilingData 字段、TilingFunc、kernel 类、
> InferShape/InferDataType、OpDef 注册）合并进模板即可。

### 2. 编译

```bash
cd $HOME/sample/AddCustom
./build.sh
# 成功后生成 build_out/custom_opp_<os>_<arch>.run
```

### 3. 部署

```bash
./build_out/custom_opp_<os>_<arch>.run --install-path="$HOME/sample/AddCustom/installed"

# 使用前加载环境（将自定义算子追加到 ASCEND_CUSTOM_OPP_PATH）
source $HOME/sample/AddCustom/installed/vendors/customize/bin/set_env.bash
```

### 4. 功能测试（msOpST）

```bash
export DDK_PATH=${INSTALL_DIR}
export NPU_HOST_LIB=${INSTALL_DIR}/{arch-os}/devlib    # 如 x86_64-linux
cd ${INSTALL_DIR}/python/site-packages/bin
./msopst run -i $HOME/AddCustom_st/AddCustom_case.json -soc <soc_version> -out $HOME/AddCustom_st
```

预期输出：`success count: 1 / failed count: 0`，详细结果见生成的 `st_report.json`。

## 实现要点

1. **2D → 1D 展平**：ND 格式下 `[N2, N1]` 张量在 GM 中按行主序连续存放，
   故 Tiling 直接用 `GetShapeSize()`（= N2*N1）作为一维长度，kernel 按元素数切分，
   对任意 `[N2, N1]` 形状通用。
2. **三级流水 + 双缓冲**：`TPipe` + `TQue`（VECIN/VECOUT）管理 UB 内存，
   `BUFFER_NUM=2` 使 DataCopy（MTE2/MTE3 引擎）与 Add（VEC 引擎）重叠执行。
3. **核间切分**：`blockLength = totalLength / GetBlockNum()`，
   每核从 `blockLength * GetBlockIdx()` 偏移处开始处理，互不重叠。
4. **切分约束**：`totalLength / BLOCK_DIM / TILE_NUM / BUFFER_NUM` 必须整除
   （本实现按官方样例取整切分；`[8, 2048]`=16384 元素，
   16384/8/8/2=128 整除，满足约束）。

## 注意事项

- `op_host/add_custom.cpp` 中 `AddConfig` 注册了 ascend910b/310p/310b/910 四款芯片，
  请按实际硬件保留对应项（工程模板生成参数 `-c ai_core-<soc_version>` 决定默认值）。
- 若 shape 不满足整除约束（如奇数维），运行期 kernel 会少算尾部数据；
  课程场景 `[8, 2048]` 不受影响。如需支持任意 shape，可在 TilingFunc 中
  增加余数处理逻辑（按行/按块分配 tile）。
- float16 累加精度：本算子为单步加法无累积误差，
  `AscendC::Add` 内部按 float16 计算，符合课程对 dtype 的要求。
