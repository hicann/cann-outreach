# Gelu 算子设计文档

> 目录名为 `op_02_mul`，但本目录实际提交的算子实现是 **Gelu**（激活函数），并非 Mul。
> 文档按实际实现内容编写。

## 1. 概述

### 1.1 基本信息

| 项目 | 内容 |
|-----|------|
| 算子名称 | Gelu（目录 `op_02_mul`，算子原型名 `Gelu`） |
| 算子类别 | Elementwise（逐元素一元运算） |
| 支持数据类型 | FP16 / FP32 |
| 支持格式 | ND |
| 目标芯片 | Ascend910B |
| 目标架构 | dav-2201（arch22） |
| 开发形态 | 标准自定义算子工程（`op_host` + `op_kernel`，打包为算子包） |
| 提交团队 | 西安邮电大学 |
| 提交者 | 刘昊（will13546） |

### 1.2 算子功能

Gelu 是神经网络中常用的激活函数。本实现采用**精确形式（erf 形式）**，而非 tanh 近似形式。输出 `output` 的 shape 与 dtype 与输入 `input_x` 完全一致，逐元素计算，无广播、无属性参数。

### 1.3 数学公式

$$
\text{gelu}(x) = 0.5 \cdot x \cdot \left(1 + \operatorname{erf}\!\left(\frac{x}{\sqrt{2}}\right)\right)
$$

其中

$$
\operatorname{erf}(z) = \frac{2}{\sqrt{\pi}}\int_0^{z} e^{-t^2}\,dt
$$

**关键常数**：

| 常量 | 值 | 说明 |
|-----|-----|------|
| `inv_sqrt2` ($1/\sqrt{2}$) | 0.70710678 | 换算成标准正态分布的自变量 |
| 系数 | 0.5 | 与 `1 + erf(...)` 相乘的缩放因子 |

**实现上的等价改写**（对应 Kernel 中五个连续向量指令）：

$$
\text{gelu}(x) = \underbrace{\left(0.5 \cdot \left(1 + \operatorname{erf}\left(\underbrace{x \cdot \tfrac{1}{\sqrt 2}}_{\text{Muls}}\right)\right)\right)}_{\text{Erf} \to \text{Adds} \to \text{Muls}} \cdot x
$$

这样改写的原因是 Ascend C 没有「`0.5 * x * (1 + erf)`」的单条融合指令，拆成 `Muls / Erf / Adds / Muls / Mul` 五步后全部是逐元素向量指令，无标量依赖链。

---

## 2. 架构设计

### 2.1 逻辑视图

| 模块 | 职责 | 核心文件 |
|------|------|---------|
| **op_host** | 算子原型注册、Shape/DataType 推导、Tiling 切分计算、TilingKey 选择 | `gelu.cpp` |
| **op_kernel** | Kernel 入口、核间偏移定位、核内分片与双缓冲流水、向量化计算 | `gelu.cpp`、`gelu_tiling.h`、`tiling_key_gelu.h` |

本工程把算子原型定义、Shape 推导、Tiling 实现**合并写在同一个 `op_host/gelu.cpp`** 中（分别位于 `optiling` / `ge` / `ops` 三个命名空间），而不是拆成 `*_def.cpp` / `*_infershape.cpp` / `*_tiling.cpp` 三个文件。

模块依赖：

```
op_host/gelu.cpp
  ├── register/op_def_registry.h        (OpDef / OP_ADD / IMPL_OP_OPTILING)
  ├── tiling/platform/platform_ascendc.h (核数、UB 容量查询)
  └── ../op_kernel/gelu_tiling.h        (GeluTilingData 结构体，Host/Device 共享)
      ../op_kernel/tiling_key_gelu.h    (DT_INPUT_X 模板参数声明)

op_kernel/gelu.cpp
  ├── kernel_operator.h                 (Ascend C 编程接口)
  ├── gelu_tiling.h                     (GeluTilingData)
  └── tiling_key_gelu.h                 (DT_INPUT_X)
```

`GeluTilingData` 与 `tiling_key_gelu.h` 被 Host 与 Kernel 两侧共同包含，是两侧的**唯一契约**：Host 写入字段、Kernel 读取字段，任何一侧改结构体都必须同步。

### 2.2 开发视图

```
op_02_mul/
├── CMakeLists.txt              # npu_op_package(custom, TYPE SHARED, ascend910b)
├── README.md
├── op_host/
│   ├── CMakeLists.txt          # npu_op_code_gen + cust_optiling(TILING) + cust_opapi(ACLNN)
│   └── gelu.cpp                # 原型定义 + InferShape/InferDataType + TilingFunc
└── op_kernel/
    ├── CMakeLists.txt          # npu_op_kernel_sources + npu_op_kernel_library
    ├── gelu.cpp                # KernelGelu<T> 类 + __global__ gelu 入口
    ├── gelu_tiling.h           # GeluTilingData
    └── tiling_key_gelu.h       # ASCENDC_TPL_ARGS_DECL/SEL
```

本目录**未包含** `tests/`（UT）与 `examples/`（aclnn 调用样例），验证依赖提交平台的标准用例流程。

### 2.3 运行视图

**数据流**（每个核内部）：

```
GM：input_x
  │  DataCopy (GM → UB，按 32 元素对齐块搬运)
  ▼
UB：inQueue (VECIN，双缓冲)
  │
  │  Muls(inv_sqrt2) → Erf → Adds(1.0) → Muls(0.5) → Mul(x)
  ▼
UB：outQueue (VECOUT，双缓冲)
  │  DataCopy (UB → GM)
  ▼
GM：output
```

**执行流程**：

```
Host 侧 TilingFunc()：
  ├── GetCoreNumAiv() / GetCoreMemSize(UB)
  ├── length = input_x.GetShapeSize()
  ├── ASCENDC_TPL_SEL_PARAM(context, dt_input_x)   按 dtype 选模板
  ├── SetBlockDim(num_cores_aiv) / workspace = 0
  ├── 核间切分：按 32 元素对齐块均分 → tailBlockNum / bigCoreDataNum / smallCoreDataNum
  ├── 核内分片：按 UB 容量算 tileDataNum
  └── 写回 GeluTilingData（9 个字段）

Device 侧 gelu<DT_INPUT_X>()：
  ├── GetTilingData → tiling_data
  └── KernelGelu<DT_INPUT_X> op; op.Init(...); op.Process();
        Init()    : 算本核 GM 起始偏移 → SetGlobalBuffer → InitBuffer(双缓冲)
        Process() : for tile : CopyIn → Compute → CopyOut
```

---

## 3. 实现方案

### 3.1 模板划分

模板参数由 `op_kernel/tiling_key_gelu.h` 声明：

```cpp
ASCENDC_TPL_ARGS_DECL(Gelu,
    ASCENDC_TPL_DATATYPE_DECL(DT_INPUT_X, C_DT_FLOAT16, C_DT_FLOAT),
);

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(DT_INPUT_X, C_DT_FLOAT16, C_DT_FLOAT),
    ),
);
```

| 模板 | 触发条件 | 实例化类型 | 说明 |
|-----|---------|-----------|------|
| FP16 | `input_x.dtype == DT_FLOAT16` | `KernelGelu<half>` | half 直算 |
| FP32 | `input_x.dtype == DT_FLOAT` | `KernelGelu<float>` | float 直算 |

本算子采用**数据类型直接作为模板参数**（DT 模板）而非「dtype 编号 + 内部分支」，Host 侧通过 `ASCENDC_TPL_SEL_PARAM(context, dt_input_x)` 传入 `ge::DataType` 的数值，由框架自动匹配实例化。相比 `relu` 使用的 `schMode` 枚举模板（UINT 模板），DT 模板少了一层人工映射，但要求 Host 侧传入的枚举值与 `ASCENDC_TPL_DATATYPE_DECL` 的顺序一致。

输入输出的 `DataType` 列表必须同序：

```cpp
this->Input("input_x").DataType({ge::DT_FLOAT16, ge::DT_FLOAT}).Format({ND, ND});
this->Output("output") .DataType({ge::DT_FLOAT16, ge::DT_FLOAT}).Format({ND, ND});
```

### 3.2 TilingData 结构体

**文件位置**：`op_kernel/gelu_tiling.h`

```cpp
struct GeluTilingData {
    uint32_t length;            // 输入总元素个数
    uint32_t smallCoreDataNum;  // 小核负责的元素个数
    uint32_t bigCoreDataNum;    // 大核负责的元素个数
    uint32_t finalSmallTileNum; // 小核需要处理的 tile 个数
    uint32_t finalBigTileNum;   // 大核需要处理的 tile 个数
    uint32_t tileDataNum;       // 每个 tile 的元素个数
    uint32_t smallTailDataNum;  // 小核最后一片 tile 的元素个数
    uint32_t bigTailDataNum;    // 大核最后一片 tile 的元素个数
    uint32_t tailBlockNum;      // 大核个数
};
```

字段分成三组：

- **总量**：`length`
- **核间切分**：`tailBlockNum`（大核个数）、`bigCoreDataNum` / `smallCoreDataNum`（大/小核元素数）
- **核内分片**：`tileDataNum`、`finalBigTileNum` / `finalSmallTileNum`、`bigTailDataNum` / `smallTailDataNum`

### 3.3 Tiling 计算逻辑

**文件**：`op_host/gelu.cpp`

#### 3.3.1 核间切分（按 32 元素对齐块）

```cpp
constexpr uint32_t kBlockSize = 32;
aligned_length = ceil(length / 32) * 32;          // 向上对齐到 32 元素
block_count    = aligned_length / 32;             // 对齐块总数
base_blocks_per_core = block_count / num_cores_aiv;
extra_blocks         = block_count % num_cores_aiv;

small_core_data_num = base_blocks_per_core       * 32;
big_core_data_num   = (base_blocks_per_core + 1) * 32;
tailBlockNum        = extra_blocks;               // 前 extra_blocks 个核为「大核」
```

切分粒度为 **32 个元素**（而非 32 字节）。对 fp32 而言 32 元素 = 128 B、对 fp16 而言 32 元素 = 64 B，两者都是 32 B 的整数倍，因此搬运对齐要求同样满足；取固定元素数而不是固定字节数，是为了让「核间切分」与「dtype」解耦，Host 侧这段逻辑无需按 dtype 分支。

核间负载差最大为 32 个元素（一个大核比小核多一个对齐块）。

#### 3.3.2 核内分片（按 UB 容量）

```cpp
uint32_t available_ub   = ub_size * 3 / 4;                       // 只用 3/4 UB
uint32_t tile_data_num  = available_ub / (2 * 2 * dtype_size);   // 2 队列 × 2 缓冲 = 4 份
tile_data_num = clamp(tile_data_num, 32, aligned_length);
tile_data_num = ceil_align(tile_data_num, 32);                   // 同时向上对齐到 32 元素
```

- 分母 `2 * 2` 的含义：inQueue 双缓冲 2 份 + outQueue 双缓冲 2 份，共 4 份 tile 空间。
- 只使用 3/4 UB 是预留余量，避免把整块 UB 分配干净。
- 下限 32 元素：保证至少能覆盖一个对齐块，不会出现 `tileDataNum = 0` 导致 `InitBuffer` 分配 0 字节。
- 上限 `aligned_length`：输入很小时不做无意义的超大 tile。

随后派生：

```
finalSmallTileNum = ceil(smallCoreDataNum / tileDataNum)
finalBigTileNum   = ceil(bigCoreDataNum   / tileDataNum)
smallTailDataNum  = (smallCoreDataNum % tileDataNum == 0) ? tileDataNum : (smallCoreDataNum % tileDataNum)
bigTailDataNum    = (bigCoreDataNum   % tileDataNum == 0) ? tileDataNum : (bigCoreDataNum   % tileDataNum)
```

注意 `tile_data_num` 在「对齐到 32 的倍数」这一步之后可能略大于 `aligned_length`（例如 `aligned_length = 40`、对齐后 `tile = 64`）。此时 `finalXxxTileNum = 1`、`xxxTailDataNum = coreDataNum`，Kernel 按尾片大小处理，逻辑仍然自洽。

#### 3.3.3 其他配置

| 项 | 值 |
|----|----|
| `SetBlockDim` | `num_cores_aiv`（总是启动全部 AIV 核） |
| `workspace_sizes[0]` | 0（本算子不需要额外 workspace） |

### 3.4 Kernel 实现

#### 3.4.1 Init：核间偏移定位

核间采用**大核在前、小核在后**的排布，因此不能直接用 `core_num × 本核长度` 求偏移，而要先按**大核步长**算，再对小核做差量修正：

```cpp
uint32_t global_buffer_index = big_core_data_num * core_num;   // 先按大核步长

this->coreDataNum = (core_num < tail_block_num) ? big_core_data_num : small_core_data_num;
this->tileNum     = (core_num < tail_block_num) ? finalBigTileNum  : finalSmallTileNum;
this->tailDataNum = (core_num < tail_block_num) ? bigTailDataNum   : smallTailDataNum;

if (core_num >= tail_block_num) {
    global_buffer_index -= (big_core_data_num - small_core_data_num) * (core_num - tail_block_num);
}
```

推导：核 $k \ge \text{tailBlockNum}$ 的真实起点为

$$
\text{tailBlockNum}\times \text{big} + (k - \text{tailBlockNum})\times \text{small}
= k\times\text{big} - (\text{big}-\text{small})\times(k-\text{tailBlockNum})
$$

即代码中的「乘大核步长再减差量」。这样所有核的切片首尾相接，总覆盖长度为 `tailBlockNum × big + (cores - tailBlockNum) × small = aligned_length`。

随后 `SetGlobalBuffer` 建立 `xGm` / `yGm`（输入与输出窗口长度都是 `coreDataNum`），并按

```cpp
pipe.InitBuffer(inQueue,  BUFFER_NUM, tileDataNum * sizeof(TYPE_X));
pipe.InitBuffer(outQueue, BUFFER_NUM, tileDataNum * sizeof(TYPE_Y));
```

申请双缓冲队列。

#### 3.4.2 Process：分片主循环

```cpp
for (uint32_t i = 0; i < tileNum; ++i) {
    const uint32_t current_tile_size = (i + 1 == tileNum) ? tailDataNum : tileDataNum;
    CopyIn(i, current_tile_size);
    Compute(current_tile_size);
    CopyOut(i, current_tile_size);
}
```

只有**最后一片**可能不足 `tileDataNum`，其余片一律按满片处理 —— 这是「核内不做逐片长度判断」的根据，热路径里只有一个分支。

#### 3.4.3 Compute：向量化计算

```cpp
AscendC::Muls(yLocal, xLocal, inv_sqrt2, current_tile_size);            // x / √2
AscendC::Erf (yLocal, yLocal, current_tile_size);                        // erf(x / √2)
AscendC::Adds(yLocal, yLocal, static_cast<TYPE_Y>(1.0f), current_tile_size);  // 1 + erf(...)
AscendC::Muls(yLocal, yLocal, static_cast<TYPE_Y>(0.5f), current_tile_size);  // 0.5 × (...)
AscendC::Mul (yLocal, yLocal, xLocal, current_tile_size);                // × x
```

五步全部是原地/双操作数向量指令，`yLocal` 既是输入又是输出，Ub 中只占用一份输出缓冲。`inv_sqrt2` 是编译期常量（`static_cast<TYPE_X>(0.70710678)`），随模板实例化按下标类型取精度。

### 3.5 API 映射

| 计算步骤 | Ascend C API | 参数签名 | 说明 |
|---------|-------------|---------|------|
| 数据搬入 | `DataCopy` | `(xLocal, xGm[progress * tileDataNum], current_tile_size)` | 元素数为单位 |
| 除以 √2 | `Muls<TYPE_X>` | `(dst, src, inv_sqrt2, count)` | 标量乘 |
| 误差函数 | `Erf<TYPE_X>` | `(dst, src, count)` | 硬件 erf 指令 |
| 加 1 | `Adds<TYPE_Y>` | `(dst, src, 1.0f, count)` | 标量加 |
| 乘 0.5 | `Muls<TYPE_Y>` | `(dst, src, 0.5f, count)` | 标量乘 |
| 乘 x | `Mul<TYPE_Y>` | `(dst, src0, src1, count)` | 张量乘 |
| 数据搬出 | `DataCopy` | `(yGm[progress * tileDataNum], yLocal, current_tile_size)` | — |
| Tiling 解析 | `GET_TILING_DATA_WITH_STRUCT` | `(GeluTilingData, tiling_data, tiling)` | 从 GM tiling buffer 还原 |

### 3.6 内存管理（UB 占用）

| 缓冲区 | 大小 | 说明 |
|-------|------|------|
| `inQueue` | `tileDataNum × sizeof(TYPE_X) × 2` | VECIN，双缓冲 |
| `outQueue` | `tileDataNum × sizeof(TYPE_Y) × 2` | VECOUT，双缓冲 |
| **合计** | `4 × tileDataNum × size` | 由 Tiling 保证 $\le \tfrac{3}{4} \times \text{UB}$ |

由于 `tileDataNum = (UB × 3/4) / (4 × dtypeSize)` 在 Host 侧已经按这个等式反解得到，**任何输入规模下 UB 占用都不超过 3/4 UB**，输入变大只会增加 tile 循环次数，不会增加 UB 占用。该设计不需要 `InitBuffer` 失败时的降级分支。

---

## 4. 性能优化

### 4.1 并行策略

- **多核并行**：Host 侧把对齐块均分到全部 AIV 核，核间负载差 ≤ 32 个元素。
- **核间偏移零开销**：`Init()` 内一次性算出 `global_buffer_index`，`Process()` 中所有地址都是 `progress * tileDataNum` 的简单乘加，没有逐片重算。
- **全向量化**：五步计算全部走 Vector 单元，无 scalar 循环、无逐元素取值。

### 4.2 流水线设计

- **双缓冲**：inQueue / outQueue 各 2 块缓冲，第 $i$ 片的 `CopyIn`（MTE2）与第 $i-1$ 片的 `Compute`（Vector）在不同缓冲上重叠。
- **单次搬运多计算**：tile 尺寸按「吃满 3/4 UB」反解，单次搬运的粒度尽可能大，摊薄同步开销；同时仍受 `aligned_length` 上限约束，小输入不会浪费。

### 4.3 可优化空间

- **空核浪费**：`SetBlockDim` 恒等于 `num_cores_aiv`，当 `aligned_length` 很小时（例如只有 1 个对齐块）其余核仍会被启动并空跑一次 `Init()`。可改为 `SetBlockDim(min(num_cores_aiv, block_count))` 消除这部分调度开销。
- **重复计算**：`tileDataNum` 在 Host 与 Kernel 两侧都参与派生（Host 算 `finalXxxTileNum`、Kernel 直接读），属于必要的冗余；如需进一步精简 TilingData 字段数，可以把 `finalBigTileNum` / `bigTailDataNum` 挪到 Kernel 内计算，只下发 `tileDataNum` 一个字段。

---

## 5. 风险与约束

### 5.1 功能约束

| 约束 | 说明 |
|------|------|
| 仅支持 FP16 / FP32 | 原型注册、TilingKey 声明、Kernel 模板均只覆盖这两种 dtype |
| 仅支持 ND 格式 | 原型中 `Format({ND, ND})`；不支持 NZ 等分形格式 |
| 不支持广播 | `InferShape` 直接复制输入 shape，两输入不等形状的语义未定义（本算子只有单输入，无此问题） |
| 无属性参数 | 本算子为 erf 精确形式，不提供 `approximate="tanh"` 之类的开关 |

### 5.2 精度风险

| 风险 | 影响 | 应对措施 |
|-----|------|---------|
| `inv_sqrt2` 常量按 `TYPE_X` 取精度 | fp16 路径下 `0.70710678` 被截断为 half（约 0.70703），相对误差约 1e-4，会线性传递到 `erf` 的自变量 | 本算子参考判据为 relative error ≤ 2⁻¹⁰ ≈ 9.8e-4（fp16 社区标准），该量级在容差之内；若需进一步收紧，可改为先在 fp32 下做 `x/√2` 再 Cast 回 half |
| half 精度下 `Erf` 的逼近误差 | fp16 的 erf 实现为多项式逼近，绝对值误差集中在 $|x|$ 较大处 | fp16 判据为 2⁻¹⁰，留有量级余量 |
| `Adds(1.0f)` / `Muls(0.5f)` 的标量在 half 下可精确表示 | 1.0 与 0.5 都是 2 的幂次，half 下无舍入 | 无需处理 |

### 5.3 越界风险

| 风险 | 影响 | 应对措施 |
|-----|------|---------|
| 尾部 padding 越界读写 | 核间切分按 `aligned_length = ceil(length/32) × 32` 对齐，当 `length` 不是 32 的整数倍时，最后一个核的 GM 窗口会覆盖到第 `length` 到 `aligned_length-1` 这段**逻辑张量之外**的元素：读时读到非法值（计算结果被丢弃，不影响正确性），**写时向输出张量末尾之外写最多 31 个元素** | 依赖 Device 侧显存按较大粒度（512 B 及以上）分配的既有事实，实际不会破坏其他数据；严格做法是 Host 侧额外下发 `length`，Kernel 在 `CopyOut` 时对尾片做长度裁剪。**当前实现未做该裁剪，属于已知的越界写窗口** |
| `length = 0` | `aligned_length = 0`、`block_count = 0`，但 `big_core_data_num = (0+1) × 32 = 32`、`tailBlockNum = 0`，导致所有核都走「小核」分支：`coreDataNum = 0`、`tileNum = 0`，`Process()` 不发生任何访存 | 已安全；但沿小核修正公式反算偏移时 `global_buffer_index` 会归零，因此不会出现越界。属「空输入不崩溃、输出未定义」 |

### 5.4 工程一致性风险

- 目录名 `op_02_mul` 与实际算子 `Gelu` 不一致，命名容易误导阅读者与自动化脚本（例如按目录名推断算子的构建/测试流水线）。
- `GeluTilingData` 是 Host/Kernel 共享结构体，字段新增/重排时两侧必须同时修改；由于 `op_kernel/gelu_tiling.h` 被 Host 侧以相对路径 `../op_kernel/gelu_tiling.h` 包含，编译期即可发现结构体不一致，风险可控。

---

## 6. 测试方案

本目录未提交 `tests/` 与 `examples/`，验证依赖提交平台的标准用例流程（按固定 shape 生成输入、跑通算子包、按社区精度标准比对）。若需本地自测，建议按以下用例设计补充：

| 用例类型 | 建议 shape | dtype | 关注点 |
|---------|-----------|-------|--------|
| 基础用例 | 与题面一致的标准 shape | fp32 / fp16 | 主流程正确性 |
| 对齐边界 | 元素数为 32 的整数倍 | fp32 / fp16 | 无 padding 越界 |
| 非对齐边界 | 元素数为 $32k+1$ 或 $32k+31$ | fp32 / fp16 | 尾部 padding 分支 |
| 小规模 | 元素数 < 32、恰好 = 32 | fp32 / fp16 | `tileDataNum` 下限与上限约束 |
| 空输入 | 元素数 = 0 | fp32 / fp16 | 不崩溃、不误写 |
| 极值 | 含 $\pm\inf$、大正负数 | fp32 | erf 饱和段（$|x| > 6$ 时输出应趋近 0 或 x） |

精度判据（社区标准）：

| dtype | 阈值 | error metric |
|-------|------|-------------|
| FP16 | $2^{-10} \approx 9.77\times10^{-4}$ | MERE < threshold 且 MARE < 10 × threshold |
| FP32 | $2^{-13} \approx 1.22\times10^{-4}$ | 同上 |

---

## 7. 交付件清单

| 文件 | 说明 |
|------|------|
| `CMakeLists.txt` | 顶层构建（`npu_op_package(custom, TYPE SHARED)`，`ASCEND_COMPUTE_UNIT=ascend910b`） |
| `op_host/CMakeLists.txt` | Host 侧编译（`cust_optiling` / `cust_opapi` 两个动态库） |
| `op_host/gelu.cpp` | 算子原型 + InferShape/InferDataType + Tiling |
| `op_kernel/CMakeLists.txt` | Kernel 侧编译（`ascendc_kernels` 二进制） |
| `op_kernel/gelu.cpp` | `KernelGelu<T>` 类与 `__global__ gelu` 入口 |
| `op_kernel/gelu_tiling.h` | `GeluTilingData` 结构体 |
| `op_kernel/tiling_key_gelu.h` | `DT_INPUT_X` 模板参数声明与选择 |
| `README.md` | 提交信息 |
