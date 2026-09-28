# add_custom 算子

基于 Ascend C（CANN）实现的逐元素加法自定义算子。

## 算子定义

| 项目 | 说明 |
|------|------|
| 算子名称 | `add_custom` |
| 功能 | 对两个输入张量 `x`、`y` 执行逐元素加法，返回结果张量 `z` |
| 数学公式 | `z = x + y` |
| shape | 2 维，如 `[N2, N1]` |
| 数据类型 | `float16` |
| 数据格式 | `ND` |
| 输入 | `x` (FLOAT16, ND), `y` (FLOAT16, ND) |
| 输出 | `z` (FLOAT16, ND) |

## 工程结构

```
add_custom/
├── CMakeLists.txt                        # 算子 CMake 入口
├── op_kernel/                            # Device 侧 Kernel 实现（Ascend C）
│   ├── add_custom.cpp                    # kernel 入口（按 tiling key 分发）
│   ├── add_custom.h                      # kernel 主体：TPipe + TQue 双缓冲流水线
│   ├── add_custom_tiling_data.h          # TilingData 结构体（host/kernel 共享）
│   └── add_custom_tiling_key.h           # tiling key 定义（schMode=0: float16）
├── op_host/                              # Host 侧实现
│   ├── add_custom_def.cpp                # 算子定义（输入输出/类型/格式/AICore 配置）
│   ├── add_custom_infershape.cpp         # shape/数据类型推理
│   ├── add_custom_tiling.cpp             # tiling 策略（核切分 + UB 切分）
│   ├── CMakeLists.txt
│   └── config/
│       └── ascend910b/
│           └── add_custom_binary.json    # 算子二进制匹配配置
├── op_graph/                             # Graph 侧 proto 注册
│   ├── add_custom_proto.h
│   └── CMakeLists.txt
└── examples/
    └── test_aclnn_add_custom.cpp         # aclnn 两段式调用示例
```

## 实现要点

### Kernel（op_kernel）

- **编程模型**：`TPipe` + `TQue` 流水线，2 个 `VECIN` 输入队列 + 1 个 `VECOUT` 输出队列，`BUFFER_NUM = 2` 双缓冲，使 GM↔UB 搬运与 UB 内向量计算流水重叠。
- **数据搬运**：`DataCopyPad`（块首 256B 对齐、块长 32B 对齐的 MTE 搬运约束）。
- **计算**：`AscendC::Add(zLocal, xLocal, yLocal, currentNum)` 执行 `z = x + y`。
- **切分**：host 侧已按核切分，kernel 中每个 AI Core 只处理 `[blockFactor * blockIdx, blockFactor * (blockIdx + 1))` 区间，Core 内再按 `ubFactor` 分块循环，尾块处理余数。
- **2 维处理**：`[N2, N1]` 按元素总数 `N2 * N1` 一维化计算（逐元素加法与维度布局无关，ND 格式下安全）。

### Tiling（op_host/add_custom_tiling.cpp）

1. 获取平台信息：AIV 核数、每核 UB 大小。
2. 校验输入/输出均为 2 维、类型仅 `FLOAT16`。
3. 核切分：`blockFactor = CeilDiv(totalNum, coreNum)`，尽量多用核并行。
4. UB 切分：`ubFactor = FloorAlign(FloorDiv(ubSize / 2 /*float16*/, 6 /*2输入+1输出×双缓冲*/), ubBlockSize)`。
5. 设置 tiling key：`float16 -> schMode 0`。

## 环境要求

- 昇腾 Atlas 硬件（Ascend 310 / 910B 等，按 `add_custom_def.cpp` 中 `AICore().AddConfig` 保留的条目）
- CANN 软件包（与硬件匹配的 toolkit + kernels），已配置 `install.cfg`
- CANN 参考工程结构（本目录需放入参考工程的 `ops/` 或 `examples/` 下参与构建）

## 编译构建

本算子需放在 CANN 参考工程（如 `Ascend/samples/ops` 或自建 ops 工程）中构建。
假设参考工程根目录为 `<ops_project>`，本目录已拷贝到 `<ops_project>/add_custom`：

```bash
cd <ops_project>
# 编译全部算子
bash build.sh --pkg --soc=${soc_version} --vendor_name=${vendor_name}
# 或仅编译本算子
bash build.sh --pkg --soc=${soc_version} --vendor_name=${vendor_name} --ops=add_custom
```

说明：
- `soc_version` 按实际芯片填写，如 `ascend910b`（与 `op_host/config/` 下目录名一致）。
- 编译产物：`add_custom` 算子包（含 kernel bin 与 host 插件）。
- 算子 JSON（`config/ascend910b/add_custom_binary.json`）中 `bin_filename` 由
  `AscendC` 按算子名 + tiling key 自动生成（`AddCustom_<md5>`），与官方样例规则一致。

## 调用示例

`examples/test_aclnn_add_custom.cpp` 演示 aclnn 两段式调用（float16、2 维 [64, 64]）：

```cpp
#include "aclnn_add_custom.h"

// 第一段：计算 workspace 大小 + 创建 executor
aclnnAddCustomGetWorkspaceSize(x, y, z, &workspaceSize, &executor);
// 第二段：执行算子
aclnnAddCustom(workspaceAddr, workspaceSize, executor, stream);
```

单独编译示例程序（需先完成算子编译并生成 `aclnn_add_custom.h`）：

```bash
g++ -std=c++17 test_aclnn_add_custom.cpp -o test_aclnn_add_custom \
    -I$ASCEND_HOME/include \
    -I<算子包输出目录>/include \
    -L$ASCEND_HOME/lib64 -L<算子包输出目录>/lib64 \
    -lacl_rt -lascendcl -lascendcl_opapi -lopapi \
    -Wl,-rpath,$ASCEND_HOME/lib64
./test_aclnn_add_custom
```

预期输出（前 10 个元素，x=1.0 + y=2.0）：

```
add_custom first input[0] is: 1.000000, second input[0] is: 2.000000, result[0] is: 3.000000
...
```

## 扩展说明

- 如需支持其他数据类型（如 float32 / int32）：
  1. `add_custom_tiling_key.h` 增加 `schMode` 枚举值；
  2. `add_custom.cpp` 增加对应 `if constexpr` 分发分支（如 `NsAddCustom::AddCustom<float>`）；
  3. `add_custom_def.cpp` / `add_custom_proto.h` 增加对应 `DataType`；
  4. `add_custom_tiling.cpp` 增加 dtype 分支与 `TYPE_SIZE` 处理；
  5. `add_custom_binary.json` 增加对应 op_list 条目。
- 如需支持动态 shape / 更多维度，参考 CANN《AI 算子开发指南》中 dynamic shape 章节。
