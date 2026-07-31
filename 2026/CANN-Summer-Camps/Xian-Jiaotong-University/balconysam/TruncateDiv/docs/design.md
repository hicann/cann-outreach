# TruncateDiv 算子设计文档

## 1 文档概述
### 1.1 文档目的
本文档为 TruncateDiv 算子 Ascend C 实现的正式设计文档，完整覆盖需求背景、TBE实现现状分析、功能定义、Host侧设计、Kernel侧设计、Tiling与分核策略、数据搬运策略、性能优化方案、精度与性能标准、兼容性分析等内容，确保算子实现可与内置TBE版本功能、精度、性能全维度对齐，通过文档评审后可进入开发阶段。
### 1.2 术语与缩写
| 缩写             | 全称 / 含义                                    |
| :--------------- | :--------------------------------------------- |
| TBE              | Tensor Boost Engine，昇腾历史 DSL 算子开发语言 |
| AIV              | AI Vector Core，向量核                         |
| UB               | Unified Buffer，统一缓冲                       |
| GM               | Global Memory，全局显存                        |
| ND               | Normal Data，通用维度格式                      |
| FP16/BF16/FP32   | 半精度/脑浮点/单精度浮点                       |
| INT8/UINT8/INT32 | 8位有符号/8位无符号/32位有符号整数              |
| Tiling           | 算子分块切分策略                               |
| broadcasting     | 张量广播                                       |

---

## 2 需求背景（required）
### 2.1 需求来源
本需求来源于 TruncateDiv 算子 Ascend C 实现优化任务书，属于昇腾基础算子开发与性能优化类社区任务。依据任务书明确的基础信息、功能实现要求、测试标准、性能与精度指标，完成 TruncateDiv 算子的设计文档输出，文档需通过评审后合入 cann-ops-competitions 仓库。
### 2.2 背景介绍
在深度学习张量运算体系中，除法运算是基础算术算子的核心组成单元，截断除法（向零取整除法）广泛应用于整数张量计算、特征维度缩放、索引坐标换算、数据量化反量化等场景，是 AI 算子库中不可或缺的基础运算组件。
当前 TruncateDiv 算子的历史版本为 TBE 实现，在部分业务场景下存在算力利用率不足、执行性能有待提升的空间。为进一步挖掘昇腾 AI 处理器的硬件算力潜力，提升算子执行效率与资源使用效率，需基于 Ascend C 编程语言对 TruncateDiv 算子进行优化实现，在保证功能与精度对齐的前提下，达成更优的性能表现。
### 2.3 需求描述
使用 Ascend C 编程语言实现 TruncateDiv 算子，支持 float16、float32、int32、bfloat16、int8、uint8 数据类型，支持广播功能，算子精度与性能均不低于现有 TBE 版本，适配 Atlas 800I/T A2 芯片平台。
### 2.4 需求拆解
1. 支持 float16、float32、bfloat16、int8、uint8、int32 全量数据类型
2. 支持张量广播功能，对齐 TBE 版本广播规则
3. 支持泛化输入场景，覆盖各类合法 shape、数据类型组合的输入，验收泛化数据用例全部通过
4. 运算性能不低于 TBE 版本
5. 运算精度不低于 TBE 版本
6. 适配 Atlas 800I/T A2 芯片硬件平台
### 2.5 验收标准概括
- **功能验收**：算子功能逻辑与 TBE 版本完全对齐，支持全部指定数据类型，广播机制符合规范，向零截断规则准确，所有合法输入、边界输入场景下运算结果符合预期，无功能缺陷。
- **精度验收**：算子精度满足 CANN Judge 平台对应题目的默认精度阈值，各数据类型误差均在允许范围内，无精度退化问题。
- **性能验收**：算子在 Atlas 800I/T A2 芯片上，所有核参与计算场景下，性能不低于 TBE 版本的 95%；对于小 shape（10μs 以下场景）若性能无法达到 95%，允许与 TBE 版本相差 3μs 以内，但需提供性能仿真图与分析结论，证明 Ascend C 实现与原 TBE 实现性能一致或更优。
- **交付验收**：设计文档内容完整、格式规范，符合仓库合入要求，可通过评审并合入 cann-ops-competitions 仓库。

---

## 3 TBE实现现状与功能分析
### 3.1 分析环境与参考文件
本次分析基于 CANN 9.0.0 版本的内置 TruncateDiv TBE 源码、Ascend 910B Kernel 配置和算子原型，参考文件路径如下：
1. 动态 TBE 实现：`$ASCEND_HOME_PATH/opp/built-in/op_impl/ai_core/tbe/impl/ops_legacy/dynamic/truncate_div.py`
2. 静态 TBE 实现：`$ASCEND_HOME_PATH/opp/built-in/op_impl/ai_core/tbe/impl/ops_legacy/truncate_div.py`
3. 算子原型：`$ASCEND_HOME_PATH/opp/built-in/op_proto/inc/ops_proto_legacy.h`、`$ASCEND_HOME_PATH/opp/built-in/op_proto/inc/elwise_calculation_ops.h`
4. Ascend 910B Kernel 配置：`$ASCEND_HOME_PATH/opp/built-in/op_impl/ai_core/tbe/kernel/config/ascend910b/ops_legacy/truncate_div.json`
5. Ascend 910B 预编译 Kernel 目录：`$ASCEND_HOME_PATH/opp/built-in/op_impl/ai_core/tbe/kernel/ascend910b/ops_legacy/truncate_div/`

安装包中同时存在动态、静态 TBE Python 实现和 Ascend 910B 高性能预编译 Kernel，TruncateDiv 是 CANN 9.0.0 的内置 AI Core 算子，可作为本次 Ascend C 实现的功能、精度和性能基线。
### 3.2 算子功能与准确语义
TruncateDiv 对两个输入执行逐元素除法，并支持广播。整数输入和浮点输入需分别处理，语义与 TensorFlow `tf.raw_ops.TruncateDiv` 一致，与 `FloorDiv`（向负无穷取整）严格区分：
1. **整数输入**：整数输出等于 x 除以 y 后向零方向截断的商，公式为 `z = trunc(x/y)`。TBE 实现中负数部分执行 ceil，正数部分执行 floor，两部分相加得到向零截断结果，等价于 `z = ceil(min(x/y, 0)) + floor(max(x/y, 0))`。例如 -7 除以 3 的 TruncateDiv 结果为 -2，而 FloorDiv 结果为 -3，不可用 FloorDiv 代替整数 TruncateDiv。
2. **浮点输入**：CANN 9.0.0 的 TBE 浮点路径直接执行向量除法，输出普通浮点除法结果，不执行 floor 或 ceil。例如 7.0 除以 3.0 的结果保留约 2.333，而非取整为 2.0。
### 3.3 输入输出参数定义
正式算子原型为 `REG_OP(TruncateDiv)`，参数定义如下：
| 名称 | I/O  | 描述                           | 数据类型                             | 格式 |
| :--- | :--- | :----------------------------- | :----------------------------------- | :--- |
| x1   | 输入 | 被除数张量                     | FP16, FP32, BF16, INT8, UINT8, INT32 | ND   |
| x2   | 输入 | 除数张量，数据类型与 x1 相同   | 同 x1                                | ND   |
| y    | 输出 | 截断除法结果，数据类型与 x1 相同 | 同 x1                                | ND   |
### 3.4 支持数据类型与格式
算子原型层声明支持 FLOAT32、FLOAT16、BFLOAT16、INT8、UINT8、INT32、INT16、UINT16、DOUBLE、INT64、COMPLEX64、COMPLEX128 类型，但 Ascend 910B 实际注册以下六种同类型组合：
| 数据类型        | 910B 支持情况 | 计算路径 |
| :-------------- | :-----------: | :------- |
| bfloat16 (BF16) |       ✅       | 浮点路径 |
| float16 (FP16)  |       ✅       | 浮点路径 |
| float32 (FP32)  |       ✅       | 浮点路径 |
| int32           |       ✅       | 整数路径 |
| int8            |       ✅       | 整数路径 |
| uint8           |       ✅       | 整数路径 |
其余类型（INT64、DOUBLE、INT16/UINT16、复数类型）未在 910B 注册，本次实现不做支持。所有支持类型均为 ND 格式，opMode 为 dynamic，format_match_mode 为 FormatAgnostic。
### 3.5 广播规则
TruncateDiv 支持 NumPy 标准双向广播：
1. 输入 shape 从右向左对齐，缺失维度按 1 处理。
2. 任一维度满足 `d1 == d2`、`d1 == 1` 或 `d2 == 1` 之一即视为兼容。
3. 输出每维取 `max(d1, d2)`。
4. 任一维度既不相等也不为 1，则 Host 侧 InferShape 与 Tiling 阶段均拒绝并返回错误。
支持场景包括：两个输入 shape 相同、单边广播（如 2×3 张量与长度为 3 的张量）、双向多维广播（如 2×1×4 与 1×3×1，输出为 2×3×4）、标量与张量广播。
### 3.6 TBE计算流程
1. **整数路径（INT32/INT8/UINT8）**：
   - 将两个输入转换为 FLOAT32（INT8/UINT8 部分平台先转 INT32 再处理）
   - 将两个输入广播到公共 Shape
   - 执行 FLOAT32 向量除法
   - 对商的负数部分执行 ceil，正数部分执行 floor
   - 将两部分相加得到向零截断的商
   - 将结果转换回原输入类型
2. **浮点路径（BF16/FP16/FP32）**：
   - 低精度类型（FP16/BF16）提升为 FLOAT32 进行中间计算
   - 将两个输入广播到公共 Shape
   - 直接执行向量除法
   - 将结果转换回原输入类型，保留小数结果
### 3.7 算子约束与边界行为
算子约束如下：
1. 两个输入的数据类型必须相同，输出数据类型与输入一致
2. 两个输入 Shape 必须满足广播规则
3. 仅支持 ND 数据格式
4. 支持动态 Shape 和动态 Rank，最大支持 8 维输入
5. 支持算子融合

边界行为严格对齐内置 TBE，不在算子内部新增除零检查或主动改写已有行为：
| 场景              | 行为（与 TBE 一致）                             |
| :---------------- | :---------------------------------------------- |
| 整数除数为 0      | 硬件除法指令产生饱和/未定义结果，由上层框架捕获 |
| 浮点除以 ±0       | 产生 ±Inf 或 NaN（IEEE 754 标准行为）           |
| NaN/Inf 输入      | 按 IEEE 754 规则传播                            |
| INT32 最小值 / -1 | 中间 float32 计算产生的精度损失与 TBE 行为一致  |
| INT8/UINT8 边界值 | 经类型转换后溢出按目标类型饱和，行为与 TBE 一致 |

---

## 4 总体设计
### 4.1 算子目录结构
```
truncate_div/
├── op_host/
│   ├── CMakeLists.txt
│   ├── truncate_div_def.cpp          # 算子原型注册
│   ├── truncate_div_infershape.cpp   # Shape 与 dtype 推导
│   └── truncate_div_tiling.cpp       # Tiling 策略实现
└── op_kernel/
    ├── CMakeLists.txt
    ├── truncate_div.cpp              # Ascend C Kernel 实现
    ├── truncate_div_tiling_data.h    # Host/Kernel 共享 Tiling 数据结构
    └── truncate_div_tiling_key.h     # TilingKey 模板参数声明
```
### 4.2 模块划分
| 模块           | 文件                          | 职责                                                                        |
| :------------- | :---------------------------- | :-------------------------------------------------------------------------- |
| 算子原型注册   | `truncate_div_def.cpp`        | 注册x1/x2/y 参数、数据类型、格式、AICore 配置                               |
| Shape 推导     | `truncate_div_infershape.cpp` | 双向广播 shape 推导，非法 shape 校验拒绝                                    |
| Tiling 策略    | `truncate_div_tiling.cpp`     | 平台信息查询、合法性校验、分核、UB tile 量化、stride 表生成、TilingKey 编码 |
| Kernel 计算    | `truncate_div.cpp`            | 分类型 Kernel 模板实现，按 TilingKey 分派计算路径                           |
| Tiling 数据    | `truncate_div_tiling_data.h`  | Host 写、Kernel 读的共享 Tiling 参数结构                                    |
| TilingKey 声明 | `truncate_div_tiling_key.h`   | 模板参数与选择器声明，编译期生成多版本 Kernel                               |
### 4.3 算子调用时序
```
┌──────────┐  InferShape   ┌──────────────┐
│  GE 图   │ ────────────> │  InferShape  │
│  编译期  │               └──────────────┘
└────┬─────┘
     │ TilingFunc (Host)
     v
┌──────────┐  TilingData   ┌──────────┐
│  Tiling  │ ────────────> │  Kernel  │
│  (Host)  │  TilingKey    │ (AIV)    |
└──────────┘               └──────────┘
     │                          │
     │ SetBlockDim              │ DataCopy GM<->UB
     │ SetTilingKey             │ Div / Trunc / Cast
     │ SetWorkspace             │ DataCopy UB->GM
     v                          v
```

---

## 5 Host侧设计
### 5.1 算子原型注册
通过 `OpDef` 注册算子原型，注册支持的 6 种数据类型、ND 格式，配置 AICore 关键参数：
| 配置项                             | 值                           | 说明                                |
| :--------------------------------- | :--------------------------- | :---------------------------------- |
| `DynamicCompileStaticFlag`         | true                         | 支持动态编译静态                    |
| `DynamicFormatFlag`                | false                        | 仅 ND 格式                          |
| `DynamicRankSupportFlag`           | true                         | 支持 -1/-2 动态 rank                |
| `DynamicShapeSupportFlag`          | true                         | 支持动态 shape                      |
| `NeedCheckSupportFlag`             | false                        | 不需要框架级 support 校验           |
| `PrecisionReduceFlag`              | true                         | 允许精度降低（BF16/FP16 中转 FP32） |
| `ExtendCfgInfo("isAscendC.value")` | "true"                       | 标识为 Ascend C 实现                |
| `AddConfig`                        | "ascend910b", "ascend910_93" | 双硬件平台注册                      |
### 5.2 Shape 推导
实现双向广播 Shape 推导逻辑：
1. 获取 x1、x2 的 StorageShape，计算输出 Rank 为两个输入 Rank 的最大值
2. 自左向右遍历输出轴，将轴索引右对齐映射到两个输入的维度，缺失维度按 1 处理
3. 未知维度（-1）直接通过广播检查，输出该维保持 -1
4. 校验维度兼容性：任一维度既不相等也不为 1 时，记录错误日志并返回 `GRAPH_FAILED`
5. 合法维度输出值取两个输入对应维度的最大值，写回输出 Shape
6. 数据类型由 OpDef 自动传播，无需额外 InferDataType 实现
### 5.3 Tiling 实现
Tiling 函数完成十步流水化处理：
1. **平台信息获取**：查询物理 AIV 核数、系统 Workspace 大小，校验核数合法
2. **UB 大小获取**：查询可用 UB 大小，预留 4096 字节给栈与框架开销
3. **数据类型校验**：校验两个输入数据类型一致且在支持列表内，映射 dtypeMode 与单元素字节数
4. **Shape 处理**：将输入 Shape 右对齐填充至 8 维，推导输出 Shape 与总元素数，分类广播模式（相同shape/x1标量/x2标量/单边广播/双向广播），计算连续非广播后缀长度用于搬运优化
5. **Workspace 设置**：仅使用系统 Workspace，不额外申请算子私有 Workspace
6. **UB tile 量化**：根据单元素 UB 占用计算单次 tile 最大元素数，按 256 字节对齐
7. **分核计算**：按分核策略计算实际使用核数与单核处理元素长度
8. **Stride 表生成**：计算两个输入的维度步长，广播维步长设为 0，供通用广播场景偏移计算
9. **标量倒数优化**：x2 为编译期常量标量且为浮点类型时，预计算倒数并执行 Newton-Raphson 迭代精化，启用乘法代替除法的优化路径
10. **Tiling 数据写入**：填充 TilingData 结构，设置 BlockDim 与 TilingKey
### 5.4 TilingData 结构
Host 与 Kernel 共享的 Tiling 参数结构：
| 字段               | 类型   | 含义                              |
| :----------------- | :----- | :-------------------------------- |
| `totalLength`      | uint64 | 输出元素总数                      |
| `coreLength`       | uint64 | 单核元素数（对齐）                |
| `tileLength`       | uint32 | 单次 UB tile 元素数               |
| `usedCoreNum`      | uint32 | 实际使用 AIV 核数                 |
| `dtypeMode`        | uint32 | 0~5 数据类型编号                  |
| `broadcastMode`    | uint32 | 0~4 广播模式编号                  |
| `canUseMul`        | uint32 | 1 表示走标量倒数乘法路径          |
| `rank`             | uint32 | 标准化后有效 rank                 |
| `reciprocal`       | float  | 编译期倒数（仅优化路径有效）      |
| `contiguousSuffix` | uint32 | 最内轴连续非广播元素数            |
| `x1Shape[8]`       | uint64 | 右对齐输入 x1 shape               |
| `x2Shape[8]`       | uint64 | 右对齐输入 x2 shape               |
| `outputShape[8]`   | uint64 | 右对齐输出 shape                  |
| `x1Stride[8]`      | uint64 | x1 维度步长，广播维为 0           |
| `x2Stride[8]`      | uint64 | x2 维度步长，广播维为 0           |
### 5.5 TilingKey 编码
通过 TilingKey 编译期生成多版本 Kernel，避免运行时分支开销：
```
bits[0]    : canUseMul    (0/1)
bits[2:1]  : broadcastMode (0-4, 3 bits)
bits[5:3]  : dtypeMode    (0-5, 3 bits)
key = (dtypeMode << 3) | (broadcastMode << 1) | canUseMul
```
共生成 6 × 5 × 2 = 60 个 Kernel 二进制变体，运行时根据输入场景选择对应变体执行。

---

## 6 Kernel侧设计
### 6.1 Kernel 总体结构
Kernel 入口按 TilingKey 分派至四类实现路径：
| 路径         | 适用场景                        | 计算流程                       |
| :----------- | :------------------------------ | :----------------------------- |
| 标量倒数优化 | 浮点类型 + x2 为编译期常量标量  | `Muls(x1, reciprocal) → Trunc` |
| 浮点通用     | FP16/FP32/BF16 通用场景         | `Div(x1, x2) → Trunc`          |
| INT8/UINT8   | 8位整数类型                     | `Cast→half→Div→Trunc→Cast`     |
| INT32        | 32位整数类型                    | `Cast→float→Div→Trunc→Cast`    |
### 6.2 浮点通用 Kernel
模板参数支持 half、float、bfloat16 三种类型，核心流程：
1. 初始化阶段计算当前核的偏移与处理元素数，设置 GM 缓冲区，初始化双缓冲队列
2. 循环处理每个 tile：
   - 按广播模式搬运数据：相同shape直接搬运x1/x2；标量场景取标量后在UB内填充；通用广播场景按stride计算偏移搬运
   - 执行向量除法指令，低精度类型在指令内部以FP32完成中间计算
   - 执行Trunc指令完成向零截断
   - 将结果回写至GM，释放UB缓冲区
3. 采用双缓冲队列，实现数据搬运与向量计算的流水并行，隐藏访存延迟
### 6.3 INT8/UINT8 Kernel
INT8/UINT8 不支持直接除法指令，经 half 类型中转计算：
1. 搬运 INT8/UINT8 数据至 UB
2. 类型转换为 half 类型
3. 执行 half 向量除法（内部以FP32计算）
4. 执行 Trunc 指令向零截断
5. 类型转换回 INT8/UINT8（四舍五入模式对齐TBE行为）
6. 结果回写 GM
中间缓冲通过单个 TBuf 切分复用，节省 UB 占用。
### 6.4 INT32 Kernel
INT32 经 float 类型中转计算：
1. 搬运 INT32 数据至 UB
2. 类型转换为 float 类型
3. 执行 float 向量除法
4. 执行 Trunc 指令向零截断
5. 类型转换回 INT32
6. 结果回写 GM
### 6.5 标量倒数优化路径
当 x2 为编译期常量浮点标量时启用：
1. 仅搬运 x1 数据至 UB，省去 x2 的搬运与填充开销
2. 执行向量标量乘法（乘以预计算的倒数），吞吐高于向量除法
3. 执行 Trunc 指令向零截断
4. 结果回写 GM
倒数在 Host 端已通过 Newton-Raphson 迭代精化，精度损失控制在允许范围内。
### 6.6 Kernel 入口分派
Kernel 入口函数接收 TilingData，根据 canUseMul 与 dtypeMode 分派到对应模板实现，无运行时动态分支开销。

---

## 7 Tiling与分核策略
### 7.1 Tiling 维度划分
TruncateDiv 为典型逐元素算子，采用"输出空间线性切分"策略：
- **核间切分**：将输出总元素数按元素序线性切分到 AIV 核，每核负责连续一段输出
- **核内切分**：每核再将负责段按 tileLength 切分为多个 UB tile，循环执行搬运-计算-回写
广播偏移由 Host 端 stride 表预计算，Kernel 内对快速路径做特化优化，通用路径按 stride 计算元素偏移。
### 7.2 UB tile 量化
根据可用 UB 大小计算单 tile 最大元素数，按 256 字节对齐，保证 DataCopy burst 效率：
```
usableUb     = ubSize - 4096（预留开销）
perElemUb    = 双缓冲占用 + 中间计算缓冲占用 + 类型转换额外缓冲
tileLength   = (usableUb / perElemUb / 对齐元素数) * 对齐元素数
```
192KB UB 下各类型单 tile 元素数约为：FP16/BF16 ~6K，FP32/INT32 ~4K，INT8/UINT8 ~7K。
### 7.3 尾块处理
- 核间尾块：最后一核处理剩余元素，长度可能小于 coreLength，Kernel 自动按实际长度循环
- 核内尾块：最后一个 tile 处理剩余元素，非对齐部分由 DataCopy 硬件自动处理
### 7.4 分核策略
1. **小 shape 单核**：总元素数 ≤ 2048 时强制单核执行，避免多核启动开销超过计算耗时
2. **启发式核数计算**：以"每核约16KB工作量"为初始估计，按元素大小折算目标核数
3. **核数边界保护**：核数不超过物理 AIV 核数，不小于1
4. **对齐与负载均衡**：单核长度按 256 字节对齐，除最后一核外各核负载一致，最后一核处理尾段
### 7.5 分核示例（Atlas 800I A2，48 AIV核）
| 总元素数 | 元素字节数 | 目标核数 | 单核长度 | 实际使用核数 |
| :-------: | :--------: | :------: | :------: | :----------: |
|   1024    |     4      |    1     |   1024   |      1       |
|   100K    |     4      |    19    |   5280   |      19      |
|    1M     |     2      |    39    |  26112   |      39      |
|   10M     |     4      |    48    |  2097152 |      48      |

---

## 8 数据搬运策略
### 8.1 搬运通道
- GM ↔ UB：通过 DataCopy 双向搬运，32字节对齐 burst 传输
- 标量广播：通过 GetValue 从 GM 取单元素 + Duplicate 在 UB 内填充为向量
- 结果回写：通过 DataCopy 将 UB 内结果写回 GM
### 8.2 缓冲与流水
所有 Kernel 路径均使用 BUFFER_NUM=2 的双缓冲队列，实现 GM↔UB 搬运与 UB 内计算的跨 tile 流水并行，隐藏全局访存延迟。
### 8.3 广播场景搬运优化
| 广播模式         | x1 搬运            | x2 搬运            | 优化点                     |
| :--------------: | :----------------- | :----------------- | :------------------------- |
| 相同 shape       | DataCopy           | DataCopy           | 最快连续搬运路径           |
| x1 为标量        | GetValue+Duplicate | DataCopy           | x1 单元素广播，无重复搬运  |
| x2 为标量        | DataCopy           | GetValue+Duplicate | 可触发标量倒数乘法优化     |
| 单边/双向广播    | DataCopy           | DataCopy           | stride 表预计算偏移        |
### 8.4 对齐优化
tile 长度与核偏移均按 256 字节对齐，保证 DataCopy 单次 burst 至少 8 个 32B 块，最大化 GM 带宽利用率；非对齐尾块由硬件自动处理。

---

## 9 性能优化思路
1. **多核并行**：输出空间线性切分，最大化 AIV 核利用率；小 shape 单核避免启动开销，大 shape 满核并行
2. **双缓冲流水**：双缓冲队列实现数据搬运与计算重叠，隐藏访存延迟
3. **标量倒数优化**：x2 为常量标量时用乘法代替除法，提升计算吞吐，倒数预计算无运行时开销
4. **中间类型优化**：整数路径统一用 FP32 中间计算对齐 TBE 精度；INT8/UINT8 经 half 中转节省 UB 占用
5. **模板特化**：编译期生成 60 个 Kernel 变体，运行时无类型/场景分支判断开销
6. **UB 紧凑复用**：中间计算缓冲通过单个 TBuf 切分复用，减少 UB 浪费，提升单 tile 元素数
7. **对齐搬运**：256 字节对齐提升 DataCopy burst 效率，最大化带宽利用率

性能预期：全核场景下性能不低于 TBE 版本的 95%；小 shape（<10μs）场景与 TBE 版本耗时差 ≤3μs。

---

## 10 精度与性能标准
### 10.1 精度标准
1. **整数路径**：INT8/UINT8/INT32 计算结果严格满足向零截断规则，与 TBE 版本同输入下输出逐比特一致
2. **浮点路径**：FP32 最大绝对/相对误差 ≤1e-5；FP16/BF16 最大绝对/相对误差 ≤1e-3，符合 CANN Judge 默认阈值
3. **边界场景**：除零、NaN/Inf、整数极值等边界行为与 TBE 版本完全一致
### 10.2 性能标准
| 场景              | 性能要求                                 |
| :---------------- | :--------------------------------------- |
| 全核参与计算      | 性能不低于 TBE 版本的 95%                |
| 小 shape（<10μs） | 与 TBE 版本耗时差 ≤3μs，提供性能分析结论 |
性能测试方法：相同硬件、相同输入下对比 Ascend C 实现与 TBE 实现的多次运行中位数耗时，覆盖全部数据类型与广播场景。

---

## 11 兼容性分析
### 11.1 数据类型兼容性
本实现支持的 6 种数据类型与 Ascend 910B TBE 注册类型完全一致，未注册类型均不实现，无类型兼容差异。
### 11.2 格式兼容性
仅支持 ND 格式，与 TBE FormatAgnostic 配置一致；通过 AutoContiguous 保证非连续输入在框架层自动转连续。
### 11.3 Shape 兼容性
支持动态 Shape、动态 Rank（最大8维），覆盖全部广播场景；不兼容 Shape 在 InferShape 与 Tiling 阶段双重校验拒绝。
### 11.4 硬件兼容性
- ✅ Atlas 800I/T A2（ascend910b）：主目标平台，全功能支持
- ✅ Atlas A3（ascend910_93）：同步注册支持
- ❌ 其他硬件平台：不在任务范围内，不做注册
### 11.5 行为对齐风险与缓解
| 风险点              | 缓解措施                                     |
| :------------------ | :------------------------------------------- |
| INT8/UINT8 中转路径差异 | 以 TBE 输出为基线做逐比特验证                |
| 浮点类型舍入差异    | 误差控制在 CANN Judge 默认阈值内             |
| 标量倒数优化精度    | Newton 迭代精化倒数，误差控制在阈值内        |
| INT32 极值精度损失  | 行为与 TBE 完全对齐，不额外修改逻辑          |

---

## 12 参考资料
1. TruncateDiv 算子开发任务书：`TruncateDiv_task_doc.md`
2. CANN 内置 `truncate_div.py`（动态/静态 TBE 实现）
3. CANN 内置 `truncate_div.json`（Ascend 910B Kernel 配置）
4. [Ascend C 算子开发文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850/opdevg/Ascendcopdevg/atlas_ascendc_map_10_0002.html)
5. [TBE 算子开发文档](https://www.hiascend.com/document/detail/zh/canncommercial/850/opdevg/tbeaicpudevg/atlasopdev_10_0001.html)
6. [Ascend C 算子开发接口文档](https://www.hiascend.com/document/detail/zh/canncommercial/850/API/ascendcopapi/atlasascendc_api_07_0003.html)
7. [CANN Judge 平台](https://cannjudge.cn/home)
8. [ops-math 开源仓](https://gitcode.com/cann/ops-math)
9. TensorFlow TruncateDiv 算子文档：https://www.tensorflow.org/api_docs/python/tf/raw_ops/TruncateDiv
10. PyTorch 除法算子文档：https://pytorch.org/docs/stable/generated/torch.div.html