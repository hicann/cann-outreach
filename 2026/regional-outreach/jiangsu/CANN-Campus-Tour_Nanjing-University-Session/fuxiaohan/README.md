# InplaceRsqrt 算子（AscendC）

功能：`self = 1 / sqrt(self)`，结果**原位覆盖**输入显存的算子，不申请任何额外输出内存与 workspace。

## 1. 规格

| 项 | 说明 |
| --- | --- |
| 算子类型 | `InplaceRsqrt` |
| 输入 / 输出 | 同一个张量 `self`（原地，输入输出同一块 GM 地址） |
| Shape | 4 维 `[N4, N3, N2, N1]`，Format = `ND`，按连续内存展平后按元素处理 |
| 数据类型 | `float16`（half） |
| 计算方式 | `AscendC::Rsqrt`（底层映射 `vrsqrt` 向量指令），整块向量化，无标量循环 |
| 尾块 | 非 32B 对齐的尾块用 `DataCopyPad`，因此 `N1` 不需要是 16 的倍数 |
| workspace | 0 字节 |
| UB 占用 | `BUFFER_NUM(2) x tileLength(8192) x 2B = 32KB`（默认 tiling） |

## 2. 文件清单

| 文件 | 说明 |
| --- | --- |
| `op_kernel/inplace_rsqrt.h` | **核心实现**：`InplaceRsqrt<T>` 类，TQueBind 原地流水 + 多核切分 + 尾块处理 |
| `op_kernel/inplace_rsqrt.cpp` | 框架调用方式的 kernel 入口 `inplace_rsqrt(...)`，读取 tiling |
| `op_host/inplace_rsqrt_tiling.h` | `InplaceRsqrtTilingData`（host / kernel 共用） |
| `op_host/inplace_rsqrt.cpp` | host 侧 tiling 函数 + `IMPL_OP_OPTILING` 注册 |
| `InplaceRsqrt.json` | 算子原型（msopgen 输入），1 输入 1 输出同名 => 引用（原地） |
| `build.sh` | 路线 A 的编译脚本（生成骨架 -> 覆盖实现 -> 编译算子包） |
| `test/direct_launch/inplace_rsqrt_direct.cpp` | 路线 B：核函数直接调用入口（复用同一份 kernel 实现） |
| `test/direct_launch/main.cpp` | 路线 B：ACL 主机程序，原地跑一遍并与 `1/sqrt(x)` 逐元素比对 |
| `test/direct_launch/{CMakeLists.txt,build.sh}` | 路线 B 的编译脚本 |

## 3. 实现要点

1. **原地语义**：kernel 只使用一个 `GlobalTensor<T> selfGm_`，`DataCopy` 读进来、算完再 `DataCopy` 写回**同一地址**，全程没有第二块显存，也没有 GM <-> GM 的额外拷贝。若框架给输出分配了独立地址（`self_out != self`），kernel 会自动多做一次搬出，两种情况结果都正确。
2. **UB 内原地计算**：`Rsqrt(x, x, count)` 的 dst 与 src 是同一块 UB。逐元素向量指令没有数据相关，允许原地。
3. **TQueBind 而不是 TQue**：原地算子的输入 Buffer 就是输出 Buffer，用 `TQueBind<VECIN, VECOUT, 2>` 让同一块 UB 既作为 VECIN 入队、又作为 VECOUT 出队，从而在**不额外占用 UB** 的前提下保留双缓冲流水（搬运 / 计算 / 搬出 三级并行）。
   - 万一某个 CANN 版本的 `Rsqrt` 不允许 dst == src：不要用裸 `TBuf` 承接结果（CopyOut 读 TBuf 与下一轮 Compute 写 TBuf 会形成 WAR 竞争），正确改法是再加一条 `TQue<TPosition::VECOUT, 2>` 输出队列，在 `Compute` 里 `AllocTensor` 出 `y`、执行 `Rsqrt(y, x, count)` 后 `EnQue(y)`，`CopyOut` 里从该队列 `DeQue` 搬出，`selfQueue_` 只负责输入。
4. **对齐**：`DataCopy` 要求 32B 对齐。核间切分时每核数据段长度向上对齐到 16 个 half（32B），因此每核的段首地址恒为 32B 对齐；只有整个张量最后一个 tile 的长度可能非对齐，走 `DataCopyPad`。
5. **多核切分**：按核号平分连续数据段，超出数据范围的核直接返回；数据量小于一个 tile 时只用一个核（host tiling 控制 `blockDim`）。
6. **精度**：硬件 `vrsqrt` 是近似实现，`float16` 下相对误差约在 1e-3 量级。若对精度敏感，打开一次牛顿迭代收敛（默认关闭）：

   ```
   y1 = y0 * (1.5 - 0.5 * x * y0 * y0)
   ```

   开启方式：把 `op_kernel/inplace_rsqrt.h` 里的 `#define INPLACE_RSQRT_NEWTON_REFINE 0` 改成 `1`，或在骨架工程的 `op_kernel/CMakeLists.txt` 中加 `add_compile_definitions(INPLACE_RSQRT_NEWTON_REFINE=1)`。
   代价：2 块 UB 临时缓冲 + 5 条向量指令（tile 8192 时 UB 占用从 32KB 增到 64KB）。
7. **边界情况**（与 `torch.rsqrt` 语义一致，不额外处理）：
   - `x = 0` -> `+inf`
   - `x < 0` -> `NaN`
   - `x = +inf` -> `0`
8. **限制**：tiling 用 `uint32_t` 传元素总数，单次下发的张量元素数需 < 2^32；4 维 shape 需连续（ND）内存。

## 4. 编译命令

### 路线 A：编译成自定义算子包（推荐，标准算子交付方式）

```bash
# 0) 环境
source /usr/local/Ascend/ascend-toolkit/set_env.sh     # 让 ASCEND_HOME_PATH / msopgen 可用
cd <本仓库根目录>

# 1) 一键：msopgen 生成骨架 -> 覆盖实现 -> 编译
bash build.sh ascend910b          # 参数为 soc_version：ascend910b / ascend910_93 / ascend310p ...

# 等价的手工命令：
#   msopgen gen -i InplaceRsqrt.json -c ai_core-ascend910b -lan cpp -out build/skeleton
#   cp -f op_kernel/inplace_rsqrt.h op_kernel/inplace_rsqrt.cpp  <骨架>/op_kernel/
#   cp -f op_host/inplace_rsqrt_tiling.h op_host/inplace_rsqrt.cpp <骨架>/op_host/
#   cd <骨架> && bash build.sh

# 2) 产物与安装
ls build/skeleton/*/build_out/                      # custom_opp_<soc>.run
./build/skeleton/*/build_out/custom_opp_*.run --install
```

编译细节：
- 用 `msopgen` 生成骨架是为了让 cmake / OpDef / aclnn 适配代码与**本机 CANN 版本**匹配，避免手写模板带来的版本差异。
- kernel 入口名与算子类型对应（`InplaceRsqrt` -> `inplace_rsqrt`）。若生成的骨架入口名或参数列表不同，以骨架为准，只改函数名/参数，实现逻辑不用动。
- 想看真实的编译器命令行（bisheng/ccec 参数、`--cce-aicore-arch` 取值），可在骨架工程里用 `make VERBOSE=1` 或查看 `${ASCEND_HOME_PATH}/compiler/tikcpp/ascendc_kernel_cmake/`。

### 路线 B：核函数直接调用（快速验证，不走算子框架）

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd test/direct_launch
bash build.sh 8            # 参数是 blockDim，默认 8
./build/inplace_rsqrt_test 8
```

若 `ascendc.cmake` 的 `ascendc_library` 接口与你的 CANN 版本对不上，可以直接用 bisheng 编译（`--cce-aicore-arch` 请按 SoC 调整，最稳妥的做法是从路线 A 的编译日志里拷贝）：

```bash
bisheng -O2 -xcce --cce-aicore-arch=dav-c220-vec \
  -I${ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw \
  -I${ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw/impl \
  -I${ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw/interface \
  -I${ASCEND_HOME_PATH}/include \
  test/direct_launch/inplace_rsqrt_direct.cpp test/direct_launch/main.cpp \
  -L${ASCEND_HOME_PATH}/lib64 -lascendcl -lpthread -ldl \
  -o inplace_rsqrt_test
./inplace_rsqrt_test 8
```

路线 B 的主机程序做的事：`aclrtMalloc` 一块 device 显存 -> 拷入正数输入 -> **用同一个地址**做输入输出调用 `inplace_rsqrt_do()` -> 拷回主机 -> 与 `1/sqrt(x)`（float32 计算）比对相对误差，并打印最大相对误差。测试 shape 故意取 `[4,7,13,3] = 1092`（`1092 % 16 = 4`），用来覆盖非 32B 对齐的尾块路径。

## 5. 常见编译问题

| 现象 | 处理 |
| --- | --- |
| `GET_TILING_DATA` 报类型/未定义错误 | 换成 `GET_TILING_DATA_WITH_STRUCT(InplaceRsqrtTilingData, tilingData, tiling);`（入口文件里已写好注释） |
| kernel 侧因 `REGISTER_TILING_DATA_CLASS` 报错 | 把该宏从 `op_host/inplace_rsqrt_tiling.h` 挪到 `op_host/inplace_rsqrt.cpp` 的 `optiling` 命名空间内 |
| `KERNEL_TASK_TYPE_DEFAULT` 未定义 | 删掉这一行（老版本 CANN 无此宏，不影响功能） |
| `Rsqrt` 不支持 `dst == src` | 见「实现要点 3」的 `TQue<VECOUT, 2>` 改法 |
| 找不到 `register/tilingdata_base.h` | kernel 编译的 include 路径需要包含 `${ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw`（骨架工程默认已包含） |