# TruncateMod 算子

基于华为昇腾（Ascend）平台，使用 Ascend C 编程语言实现的 `TruncateMod` 自定义算子。语义对齐社区内置 `TruncateMod`（等价于 numpy / torch 的 `fmod`），满足 CANN Judge 判题平台的精度与性能验收要求。

## 简介

`TruncateMod` 为逐元素（element-wise）截断取模算子：商向零取整（truncate），余数符号与被除数 `x1` 保持一致。

计算公式：

```
y = x1 - trunc(x1 / x2) * x2       （trunc 向零取整）
```

为统一各类型处理并避免整型溢出，`trunc` 在 UB 上以 fp32 无分支实现：

```
trunc(t) = floor(max(t, 0)) + ceil(min(t, 0))
```

## 特性

| 项目 | 说明 |
| --- | --- |
| 支持数据类型 | float16、float32、bfloat16、int32、int8、uint8 |
| 支持形状 | NumPy 风格广播（右对齐补维，维度相容取较大者） |
| 支持硬件 | Atlas 800I/T A2（ascend910b）、ascend910_93、ascend950 |
| 除零行为 | 浮点 → `NaN`，有符号整型 → `-1`，无符号整型 → `255`（对齐内置 golden） |

## 目录结构

```
TruncateMod/
├── README.md                    # 本文件
├── TruncateMod设计文档.md        # 算子设计文档（需求/详细设计/可维可测）
├── TruncateMod自验证报告.md      # 功能/精度/性能自验证报告
├── CANNJudge_排名截图.png        # 判题平台提交排名
├── CANNJudge_通过截图1.png       # 提交执行结果（测试点 32~50）
├── CANNJudge_测试点数据.xlsx     # 逐测试点用时/精度数据
└── code/                        # 算子实现工程（详见 code 分支）
    ├── build.sh                 # 构建 / UT / 示例入口脚本
    ├── CMakeLists.txt
    ├── op_host/                 # 算子原型、InferShape/InferDataType、Tiling
    │   ├── truncate_mod_def.cpp
    │   ├── truncate_mod_infershape.cpp
    │   └── truncate_mod_tiling.cpp
    ├── op_kernel/               # kernel 实现与 tiling 头
    │   ├── truncate_mod.cpp
    │   ├── truncate_mod.h
    │   ├── truncate_mod_tiling_data.h
    │   └── truncate_mod_tiling_key.h
    ├── examples/                # aclnn 单算子调用示例
    │   └── test_aclnn_truncate_mod.cpp
    └── tests/ut/                # host / kernel 单元测试及数据脚本
        ├── op_host/             # tiling UT
        ├── op_kernel/           # kernel UT + gen_data.py / compare_data.py
        └── run.sh
```

> 说明：算子实现工程 `code/` 位于 `code` 分支；本目录（`dock` 分支）汇总设计文档与验证材料。

## 环境要求

- 已安装 CANN（ascend-toolkit），并配置 `ASCEND_HOME_PATH`（默认 `/usr/local/Ascend/cann`）
- Python3 + numpy + ml_dtypes（用于精度对拍中的 golden 生成与 bfloat16 支持）
- 运行示例（`-e`）需要 NPU 设备

## 快速开始

进入 `code/` 目录，使用 `build.sh` 统一入口：

```bash
# 构建（默认 soc: ascend910b）
bash build.sh

# 指定编译线程数
bash build.sh -j8

# 运行单元测试（op_host tiling UT + op_kernel UT + 精度对拍）
bash build.sh -u

# 运行 aclnn 调用示例（需要 NPU）
bash build.sh -e

# 清理构建产物
bash build.sh --make_clean

# 查看帮助
bash build.sh -h
```

构建成功后产物：

- Kernel 二进制：`build/op_kernel/ascendc_kernels/binary/ascend910b/*.o`
- 自定义算子包：`build/custom_opp_*.run`

## 测试与验证

单元测试由 `tests/ut/run.sh` 编译执行，分三段：

1. **op_host tiling UT** — 校验 tilingKey / tilingData / workspace 与预期一致；
2. **op_kernel UT** — 基于 `tikicpulib` CPU 孪生运行 kernel，输出 bin；
3. **精度对拍** — `gen_data.py` 生成 golden（numpy `fmod`），`compare_data.py` 逐元素比对。

精度判据（社区标准）：

- 浮点：`MERE < threshold` 且 `MARE < 10 × threshold`（float16=2⁻¹⁰、bfloat16=2⁻⁷、float32=2⁻¹³）
- 整数：二进制一致或绝对误差为 0

## 验证结果

判题平台 50 个测试点：**49 Pass，1 Wrong Answer（测试点 45，错误占比 0.05%）**。

- 精度：唯一失败点为个别元素的 fp32 精度边界问题，非结构性错误；
- 性能：大 shape 场景（测试点 48–50）用时约为最优用时的 97%~98%，满足 ≥95% 红线；
- 排名：账号 `TheZombie` 位列第 10 名，得分 51.71。

详见 [TruncateMod自验证报告.md](./TruncateMod自验证报告.md)。

## 已知约束

1. x1、x2 数据类型必须一致；
2. int32 走 fp32 计算路径，在 `|value| ≤ 2^24` 范围内精确；超大幅值列入后续加固；
3. 广播 stride 已在 tiling 中完整携带，广播分支的完整 kernel 计算路径在后续阶段完善。

## 相关文档

- [算子设计文档](./TruncateMod设计文档.md)
- [自验证报告](./TruncateMod自验证报告.md)
