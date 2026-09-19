# Relu 算子设计文档

## 1. 概述

### 1.1 基本信息

| 项目 | 内容 |
|-----|------|
| 算子名称 | Relu（目录 `op_03_relu`，算子原型名 `Relu`，算子包 `relu_custom`） |
| 算子类别 | Elementwise（逐元素一元运算 / 激活函数） |
| 支持数据类型 | FP32 / FP16 |
| 支持格式 | ND |
| 目标芯片 | Ascend910B / Ascend910_93（arch32）、Ascend950（arch35） |
| 目标架构 | dav-2201 及以上 |
| 开发形态 | 标准自定义算子工程（`op_host` + `op_kernel` + `tests/ut` + `examples`） |
| 提交团队 | 西安邮电大学 |
| 提交者 | 刘昊（will13546） |

### 1.2 算子功能

Relu（Rectified Linear Unit）是最基础的激活函数：把输入中的负值截断为 0，非负值原样保留。输出 `y` 与输入 `x` 的 shape、dtype 完全一致，逐元素计算。

$$
y_i = \max(0, x_i)
$$

本实现除了算子本体之外，还提交了完整的 `tests/ut`（Host Tiling UT + Kernel CPU 桩 UT + 精度比对脚本）与 `examples`（aclnn 两段式调用样例），是可独立构建、测试、安装的完整工程。

---

## 2. 架构设计

### 2.1 逻辑视图

| 模块 | 职责 | 核心文件 |
|------|------|---------|
| **op_host / 原型** | 算子输入输出注册（名称、dtype、格式、`AutoContiguous`） | `relu_def.cpp` |
| **op_host / 形状推导** | `InferShape`：输出 shape = 输入 shape | `relu_infershape.cpp` |
| **op_host / Tiling** | 平台信息查询、多核切分、UB 切分、TilingKey 选择 | `relu_tiling.cpp` |
| **op_kernel / 入口** | 按 `schMode` 模板参数分派到具体 dtype 实例 | `relu.cpp` |
| **op_kernel / 实现** | `NsRelu::Relu<T>` 类：核间偏移、分片循环、双缓冲 | `relu.h` |
| **op_kernel / 契约** | `ReluTilingData` 结构体、`schMode` 模板参数声明 | `relu_tiling_data.h`、`relu_tiling_key.h` |
| **tests/ut** | Host Tiling UT、Kernel CPU 桩 UT、数据生成与精度比对 | `tests/ut/**` |
| **examples** | aclnn 两段式接口调用样例 | `examples/test_aclnn_relu.cpp` |

模块依赖：

```
op_host/relu_def.cpp ──┐
op_host/relu_tiling.cpp ┴── register/op_def_registry.h（IMPL_OP_OPTILING / OP_ADD）
op_host/relu_infershape.cpp ── register/op_impl_registry.h（IMPL_OP_INFERSHAPE）

op_host/relu_tiling.cpp ──▶ ../op_kernel/relu_tiling_data.h   （TilingData 契约）
                        ──▶ ../op_kernel/relu_tiling_key.h    （schMode 契约）
op_kernel/relu.cpp ──▶ relu.h ──▶ relu_tiling_data.h / relu_tiling_key.h
```

TilingData 与 TilingKey 两个头文件被 Host 与 Kernel **共同包含**，构成两侧的唯一契约：Host 决定写什么、Kernel 决定读什么，任何一侧修改都会在编译期暴露。

### 2.2 开发视图

```
op_03_relu/
├── build.sh                    # 构建脚本（-j 线程数 / --make_clean / -u 跑 UT / -e 跑 example）
├── CMakeLists.txt              # npu_op_package(relu_custom, TYPE RUN)
├── README.md
├── op_host/
│   ├── CMakeLists.txt          # cust_optiling / cust_opapi / cust_op_proto 三个库
│   ├── relu_def.cpp            # 算子原型注册
│   ├── relu_infershape.cpp     # 形状推导
│   └── relu_tiling.cpp         # Tiling 实现
├── op_kernel/
│   ├── CMakeLists.txt
│   ├── relu.cpp                # __global__ 入口（按 schMode 分派）
│   ├── relu.h                  # NsRelu::Relu<T> 类实现
│   ├── relu_tiling_data.h      # ReluTilingData
│   └── relu_tiling_key.h       # schMode 0/1
├── examples/
│   ├── CMakeLists.txt          # 链接 cust_opapi / libascendcl / libnnopbase / libopapi
│   ├── run.sh
│   └── test_aclnn_relu.cpp     # aclnnReluGetWorkspaceSize + aclnnRelu
└── tests/ut/
    ├── run.sh                  # op_host UT → 生成数据 → op_kernel UT → compare_data.py → op_api UT
    ├── cmake/BuildGoogleTest.cmake
    ├── common/                 # Tiling/InferShape CaseExecutor、Context/Platform 伪造
    ├── op_host/
    │   ├── test_op_host_main.cpp
    │   └── test_relu_tiling.cpp
    └── op_kernel/
        ├── relu_tiling.h       # CPU 桩侧 GET_TILING_DATA 宏
        ├── test_relu.cpp       # ICPU_RUN_KF 直接跑核函数
        └── relu_data/
            ├── gen_data.py     # 生成输入与 golden
            └── compare_data.py # MARE / MERE 精度比对
```

`build.sh` 的四种用法：

| 命令 | 行为 |
|------|------|
| `bash build.sh` | 构建算子包（默认 `ASCEND_COMPUTE_UNIT=ascend910b`），产出 `custom_opp_*.run` |
| `bash build.sh -j8` | 指定编译线程数 |
| `bash build.sh --make_clean` | 清理 `build/` 与 `build_out/` |
| `bash build.sh -u` | 进入 `tests/ut` 跑单元测试 |
| `bash build.sh -e` | 构建后进入 `examples` 跑 aclnn 样例（需 NPU） |

### 2.3 运行视图

**数据流**（每个核内部）：

```
GM：x
  │  DataCopy (GM → UB)
  ▼
UB：inputQueueX (VECIN，双缓冲)
  │
  │  AscendC::Relu(yLocal, xLocal, currentNum)
  ▼
UB：outputQueueY (VECOUT，双缓冲)
  │  DataCopy (UB → GM)
  ▼
GM：y
```

**执行流程**：

```
Host 侧 ReluTilingFunc()：
  ├── GetPlatformInfo()  → ubSize、coreNum（任一为 0 直接报错返回）
  ├── GetWorkspaceSize() → workspace[0] = 0
  ├── inputShape.GetStorageShape() → totalNum（GetDimNum()==0 的标量按 {1} 处理）
  ├── inputDesc.GetDataType() → typeSize（fp32=4 / fp16、bf16=2）
  ├── 多核切分：usedCoreNum（从 min(ceil(totalNum/1024), coreNum) 向下回退到整除且 32B 对齐）
  ├── UB 切分：ubFactor（预留 8KB，按 4 份 tile 空间反解）
  ├── SetBlockDim(usedCoreNum)
  └── SetTilingKey(GET_TPL_TILING_KEY(schMode))

Device 侧 relu<schMode>()：
  ├── REGISTER_TILING_DEFAULT + GET_TILING_DATA_WITH_STRUCT
  ├── if constexpr (schMode == 0) NsRelu::Relu<half>  op;  op.Init(); op.Process();
  └── if constexpr (schMode == 1) NsRelu::Relu<float> op;  op.Init(); op.Process();
        Init()    : 校验并夹紧 ubLength_ → 算核间偏移 → SetGlobalBuffer → InitBuffer(双缓冲)
        Process() : while (progress < blockLength_) { CopyIn → Compute → CopyOut }
```

---

## 3. 实现方案

### 3.1 模板划分

模板参数在 `op_kernel/relu_tiling_key.h` 中声明为**枚举型 UINT 模板**（`schMode`）：

```cpp
#define RELU_TPL_SCH_MODE_0 0   // half
#define RELU_TPL_SCH_MODE_1 1   // float

ASCENDC_TPL_ARGS_DECL(
    Relu,
    ASCENDC_TPL_UINT_DECL(schMode, 1, ASCENDC_TPL_UI_LIST, RELU_TPL_SCH_MODE_0, RELU_TPL_SCH_MODE_1));

ASCENDC_TPL_SEL(ASCENDC_TPL_ARGS_SEL(
    ASCENDC_TPL_UINT_SEL(schMode, ASCENDC_TPL_UI_LIST, RELU_TPL_SCH_MODE_0, RELU_TPL_SCH_MODE_1)));
```

`ASCENDC_TPL_UINT_DECL` 的第二个参数 `1` 是位宽（bit 数），两个取值各占 1 bit，因此 TilingKey 的低位直接编码 `schMode`。

| schMode | 触发条件 | 实例化类型 |
|:-------:|---------|-----------|
| 0 | `dtype == DT_FLOAT16` 或 `DT_BF16` | `NsRelu::Relu<half>` |
| 1 | 其他（`DT_FLOAT`） | `NsRelu::Relu<float>` |

Kernel 入口用 `if constexpr` 分派：

```cpp
template <uint32_t schMode>
__global__ __aicore__ void relu(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    REGISTER_TILING_DEFAULT(ReluTilingData);
    GET_TILING_DATA_WITH_STRUCT(ReluTilingData, tilingData, tiling);

    if constexpr (schMode == RELU_TPL_SCH_MODE_0) { NsRelu::Relu<half>  op; op.Init(x, y, &tilingData); op.Process(); }
    if constexpr (schMode == RELU_TPL_SCH_MODE_1) { NsRelu::Relu<float> op; op.Init(x, y, &tilingData); op.Process(); }
}
```

`if constexpr` 保证每个 `schMode` 实例化时只保留一条分支，不会把两种 dtype 的代码都编进去。

### 3.2 TilingData 结构体

**文件位置**：`op_kernel/relu_tiling_data.h`

```cpp
struct ReluTilingData {
    int64_t totalNum    = 0;   // 总元素数量
    int64_t blockFactor = 1;   // 每个核处理的元素数量
    int64_t ubFactor    = 0;   // 每次 UB 循环处理的元素数量
};
```

字段刻意保持最少：核间切分只下发 `blockFactor`，核内分片只下发 `ubFactor`。「每核起始偏移」由 Kernel 自己按 `blockFactor × GetBlockIdx()` 计算；「分片个数」由 `ceilDiv(blockLength_, ubLength_)` 在 `Process()` 的 `while` 循环里隐式决定 —— 因为切分策略保证了 `totalNum % usedCoreNum == 0`，所以每个核的长度都严格等于 `blockFactor`，不需要尾核特判，也不需要下发「尾核长度」字段。

### 3.3 Tiling 计算逻辑

**文件**：`op_host/relu_tiling.cpp`

#### 3.3.1 输入校验

| 校验项 | 不通过时的行为 |
|-------|--------------|
| `platformInfoPtr != nullptr` | `OP_CHECK_NULL_WITH_CONTEXT` 报错返回 |
| `coreNum != 0` | `OP_LOGE("coreNum is 0")`，返回 `GRAPH_FAILED` |
| `ubSize != 0` | `OP_LOGE("ubSize is 0")`，返回 `GRAPH_FAILED` |
| `tiling != nullptr` | `OP_CHECK_NULL_WITH_CONTEXT` 报错返回 |
| `inputShape != nullptr` | 同上 |
| `totalNum > 0` | `OP_LOGE("invalid totalNum")`，返回 `GRAPH_FAILED` |
| `inputDesc != nullptr` | `OP_CHECK_NULL_WITH_CONTEXT` 报错返回 |

标量输入（`GetDimNum() == 0`）通过 `EnsureNotScalar()` 归一化为 shape `{1}`，避免 `GetShapeSize()` 返回 0。

#### 3.3.2 数据类型映射

```cpp
int64_t typeSize = 4;                                        // 默认 fp32
if (dataType == DT_FLOAT16 || dataType == DT_BF16) typeSize = 2;

constexpr int64_t BLOCK_BYTES = 32;
const int64_t alignNum = BLOCK_BYTES / typeSize;             // fp32→8，fp16/bf16→16
```

| dtype | `typeSize` | `alignNum` |
|-------|:----------:|:----------:|
| DT_FLOAT | 4 | 8 |
| DT_FLOAT16 | 2 | 16 |
| DT_BF16 | 2 | 16 |

`alignNum` 是 32 B 搬运粒度对应的元素数，是**所有对齐操作的唯一依据**（核间切分、`ubFactor` 都按它取整）。

#### 3.3.3 多核切分

切分的目标是「**每个核拿到完全相等且 32B 对齐的长度**」，从而让 Kernel 侧不做任何余数处理：

```cpp
int64_t usedCoreNum = ceilDiv(totalNum, MIN_SPLIT_THRESHOLD);   // 每核目标 ≥ 1024 元素
if (usedCoreNum > coreNum) usedCoreNum = coreNum;
if (usedCoreNum < 1) usedCoreNum = 1;

// 向下回退，直到满足「整除 + 对齐」两个条件
while (usedCoreNum > 1) {
    if ((totalNum % usedCoreNum == 0) && ((totalNum / usedCoreNum) % alignNum == 0)) break;
    --usedCoreNum;
}

const int64_t blockFactor = totalNum / usedCoreNum;
```

两处取整的含义：

- **`MIN_SPLIT_THRESHOLD = 1024`**：数据量不大时不铺满所有核，减少核启动与调度开销。例如 92160 个元素按 1024 目标切分得到 90，超过物理核数 64 后被截断为 64；而 10000 个元素只会用 10 个核。
- **向下回退循环**：优先保证「整除 + 32B 对齐」，而不是「核数最多」。极端情况下（`totalNum` 为质数）回退到 `usedCoreNum = 1`，全部数据由单核处理，属于用并行度换实现简洁性的取舍。

> 该循环一定会终止：`usedCoreNum = 1` 时两个条件恒成立（任何数都能被 1 整除，且 `totalNum % alignNum == 0` 对满足本算子前提的输入成立），因此不会死循环。

#### 3.3.4 UB 切分

```cpp
constexpr uint64_t UB_RESERVED_BYTES = 8 * 1024;
const uint64_t usableUb = (ubSize > UB_RESERVED_BYTES) ? (ubSize - UB_RESERVED_BYTES) : ubSize;

int64_t maxUbFactor = usableUb / (4ULL * typeSize);   // 4 份：in 队列 ×2 缓冲 + out 队列 ×2 缓冲
maxUbFactor = (maxUbFactor / alignNum) * alignNum;    // 对齐到 32B
if (maxUbFactor < alignNum) maxUbFactor = alignNum;

int64_t ubFactor = min(blockFactor, maxUbFactor);
ubFactor = (ubFactor / alignNum) * alignNum;
if (ubFactor <= 0) ubFactor = alignNum;
```

分母中的 `4` 与 Kernel 的缓冲申请严格对应（见 3.6）：`inputQueueX` 2 块 + `outputQueueY` 2 块。

预留 8 KB UB 而不是把 UB 用满，是给框架自身的临时空间留余量，避免与其他隐式分配冲突。

#### 3.3.5 TilingKey 与 BlockDim

```cpp
context->SetBlockDim(static_cast<uint32_t>(usedCoreNum));

uint64_t tilingKey;
if (dataType == DT_FLOAT16 || dataType == DT_BF16)
    tilingKey = GET_TPL_TILING_KEY(RELU_TPL_SCH_MODE_0);   // half 实例
else
    tilingKey = GET_TPL_TILING_KEY(RELU_TPL_SCH_MODE_1);   // float 实例
context->SetTilingKey(tilingKey);
```

`SetBlockDim` 用的是**回退之后**的 `usedCoreNum`（不是平台物理核数），保证实际启动的核数、TilingData 中的 `blockFactor`、Kernel 中的 `GetBlockIdx()` 取值范围三者一致。

### 3.4 Kernel 实现

#### 3.4.1 Init：参数夹紧与偏移定位

```cpp
blockLength_ = tilingData->blockFactor;
ubLength_    = tilingData->ubFactor;

if (ubLength_ <= 0 || ubLength_ > blockLength_) ubLength_ = blockLength_;  // 夹紧 ①
constexpr int64_t maxTileElems = 4096;
if (ubLength_ > maxTileElems) ubLength_ = maxTileElems;                    // 夹紧 ②
if (ubLength_ <= 0) ubLength_ = 1;                                         // 夹紧 ③

const int64_t gmOffset = blockLength_ * static_cast<int64_t>(GetBlockIdx());
inputGMX.SetGlobalBuffer((__gm__ T*)x + gmOffset, blockLength_);
outputGMY.SetGlobalBuffer((__gm__ T*)y + gmOffset, blockLength_);

pipe.InitBuffer(inputQueueX,  BUFFER_NUM, ubLength_ * sizeof(T));
pipe.InitBuffer(outputQueueY, BUFFER_NUM, ubLength_ * sizeof(T));
```

三道夹紧是**防御性设计**，源码注释中说明了动机：

| 夹紧 | 触发场景 | 作用 |
|:----:|---------|------|
| ① `ubLength_ > blockLength_` | 工程自带的 Kernel UT 手工构造 `ubFactor = totalNum`（见 `tests/ut/op_kernel/test_relu.cpp`） | 避免 UB 按超大值申请而溢出 |
| ② `maxTileElems = 4096` | 同上，兜底 UB 申请上限（4096 × 4B × 4 份 = 64 KB） | 保证任何输入下 UB 都不会被分配干净 |
| ③ `ubLength_ <= 0` | TilingData 结构体默认值为 `ubFactor = 0` | 避免除零与 0 字节 `InitBuffer` |

也就是说，Kernel 不假设 Host 一定给出合法 `ubFactor` —— 这是**把正确性建立在 Kernel 自洽之上**，而不是依赖 Host 的自觉。代价是 Host 若算出过大的 `ubFactor`，Kernel 会静默用 4096 替代，Host 与 Kernel 对 tile 尺寸的认知可能出现不一致（本算子的 `Process()` 是按 `ubLength_` 自行推进的，不依赖 Host 的 tile 个数，因此不影响正确性）。

#### 3.4.2 Process：分片主循环

```cpp
int64_t progress = 0;
while (progress < blockLength_) {
    int64_t currentNum = blockLength_ - progress;
    if (currentNum > ubLength_) currentNum = ubLength_;

    CopyIn(progress, currentNum);
    Compute(currentNum);
    CopyOut(progress, currentNum);

    progress += currentNum;
}
```

核内同样只有「满片」与「尾片」两种情形，尾片通过 `min(剩余, ubLength_)` 自然收缩。

#### 3.4.3 Compute

```cpp
LocalTensor<T> xLocal = inputQueueX.DeQue<T>();
LocalTensor<T> yLocal = outputQueueY.AllocTensor<T>();

AscendC::Relu(yLocal, xLocal, static_cast<uint32_t>(currentNum));

outputQueueY.EnQue(yLocal);
inputQueueX.FreeTensor(xLocal);
```

这里显式写 `AscendC::Relu` 而非 `Relu`，是因为当前类的模板名也叫 `Relu`（`NsRelu::Relu<T>`），不加命名空间限定会发生名称冲突 —— 源码注释中对此做了说明。这是一个典型的「类名与算子 API 同名」问题，后续若重命名类（例如 `KernelRelu<T>`）即可去掉限定。

### 3.5 API 映射

| 计算步骤 | Ascend C API | 参数签名 | 约束说明 |
|---------|-------------|---------|---------|
| 数据搬入 | `DataCopy<T>` | `(xLocal, inputGMX[progress], currentNum)` | `currentNum` 为元素数；本算子切分策略保证对齐 |
| 激活 | `AscendC::Relu<T>` | `(yLocal, xLocal, currentNum)` | 逐元素 `max(0, x)` |
| 数据搬出 | `DataCopy<T>` | `(outputGMY[progress], yLocal, currentNum)` | — |
| 队列管理 | `AllocTensor` / `EnQue` / `DeQue` / `FreeTensor` | — | VECIN / VECOUT 位置 |
| Tiling 解析 | `GET_TILING_DATA_WITH_STRUCT` | `(ReluTilingData, tilingData, tiling)` | 由 `REGISTER_TILING_DEFAULT` 配合 |

### 3.6 内存管理（UB 占用）

| 缓冲区 | 大小 | 说明 |
|-------|------|------|
| `inputQueueX` | `ubLength_ × sizeof(T) × 2` | VECIN，双缓冲 |
| `outputQueueY` | `ubLength_ × sizeof(T) × 2` | VECOUT，双缓冲 |
| **合计** | `4 × ubLength_ × sizeof(T)` | 与 Host 侧 `maxUbFactor` 的分母 `4` 严格对应 |

**容量验证**（以 fp32、`ubSize` 预留 8 KB 为例）：

```
maxUbFactor = floor((ubSize - 8192) / (4 × 4))   → 向下对齐到 8 的倍数
ubFactor    = min(blockFactor, maxUbFactor)       → 再向下对齐到 8 的倍数
实际占用    = 4 × ubFactor × 4 bytes ≤ ubSize - 8192 < ubSize   ✅
```

由于 `ubFactor` 在 Host 侧已经按「4 份」反解并两次向下对齐，Kernel 侧的 `InitBuffer` 必然成功，不需要分配失败的降级分支。

---

## 4. 性能优化

### 4.1 并行策略

- **自适应核数**：不是无脑铺满物理核，而是按 `totalNum` 以 1024 元素/核为目标决定核数，小规模数据避免多核调度开销。
- **整除切分**：核数向下回退到「`totalNum` 能被整除且每核长度 32B 对齐」为止，使 Kernel 侧完全没有余数分支，热路径只有 `min` 一次判断。
- **全向量化**：计算路径只有一条 `AscendC::Relu` 向量指令，没有 scalar 循环或 `GetValue/SetValue`。

### 4.2 流水线设计

- **双缓冲**：in/out 队列各 2 块缓冲，第 $i$ 片的 `CopyIn` 与第 $i-1$ 片的 `Compute` 在不同缓冲上重叠，实现搬运（MTE）与计算（Vector）并行。
- **单缓冲收敛的可能**：当前实现恒为双缓冲。当 `blockLength_ == ubLength_`（每核只有一片）时双缓冲不产生流水收益，理论上可降为单缓冲减少 UB 占用，但对本算子收益有限，未做区分。

### 4.3 可优化空间

| 方向 | 说明 | 代价 |
|------|------|------|
| 回退循环的粒度 | 当前回退是「逐核递减」，`totalNum` 为大质数时会退到 1 个核，牺牲并行度 | 改为「允许核间长度差 1 个对齐块」可保持满核，但 Kernel 需增加尾核特判 |
| 非对齐输入 | 当前不对齐就回退核数，无法处理任意长度的高效切分 | 引入 `DataCopyPad` 后即可按元素切分，Kernel 需处理尾部 padding |
| 单缓冲收敛 | 每核只有一片时用单缓冲 | 增加一个分支，收益仅在小输入场景 |

---

## 5. 风险与约束

### 5.1 功能约束

| 约束 | 说明 |
|------|------|
| 仅 FP32 / FP16 | `relu_def.cpp` 中 `DataType({ge::DT_FLOAT, ge::DT_FLOAT16})`；Tiling 里虽然按 `DT_BF16` 处理了 `typeSize`，但原型未注册 BF16，BF16 输入不会被接受 |
| 仅 ND 格式 | 原型与 `UnknownShapeFormat` 均为 `FORMAT_ND` |
| 依赖 `AutoContiguous()` | 输入输出均声明为自动连续化，非连续输入会由框架插入拷贝 |
| 不支持原地输出 | 输入 `x` 与输出 `y` 是两个独立张量，不支持 `y` 与 `x` 共用同一块显存 |

### 5.2 精度风险

| 风险 | 影响 | 说明 |
|-----|------|------|
| 无 | — | Relu 是「取最大值」操作，不产生任何舍入误差：正数原样透传、负数写 0，fp16/fp32 下结果与参考实现 **bit-exact** |

### 5.3 输入形状风险

`InferShape` 的实现直接复制输入 shape 到输出：

```cpp
const gert::Shape* input_shape = (1 > 0) ? context->GetInputShape(0) : nullptr;
for (size_t i = 0; i < 1; i++) {
    gert::Shape* output_shape = context->GetOutputShape(i);
    if (output_shape == nullptr) return ge::GRAPH_FAILED;
    const gert::Shape* in_shape = (i < 1) ? context->GetInputShape(i) : input_shape;
    ...
    *output_shape = *in_shape;
}
```

其中的 `(1 > 0)` 与 `i < 1` 是**自动生成模板留下的常量条件**（`1` 表示输入个数），逻辑正确但对阅读者不友好：常量真值条件恒成立、空指针分支不可达。本算子只有一个输入一个输出，行为正确；若后续扩展为多输入算子，这段代码需要重写。**源码中的 `// TODO: 实现形状推导逻辑` 标注与上述常量条件说明这一点尚未手工整理。**

### 5.4 工程一致性风险

`tests/ut/op_host/test_relu_tiling.cpp` 中的期望值与当前 Tiling 实现不一致（详见 6.2「已知问题」），需要在提交前校正。

---

## 6. 测试方案

### 6.1 三层测试结构

`tests/ut/run.sh` 串起三层测试，任一层失败即整体失败：

```
[1] op_host UT      ./build/op_host/relu_op_host_ut
      └── gtest + TilingContextPara 伪造 Tiling 上下文，直接调 ReluTilingFunc
          校验：tilingKey、序列化后的 TilingData、workspace size
[2] op_kernel UT     ./build/op_kernel/relu_op_kernel_ut
      └── CPU 桩（tikicpulib）上 ICPU_RUN_KF 直接执行核函数
          手工构造 TilingData + ICPU_SET_TILING_KEY，输出落盘为 bin
[3] 精度比对         python3 relu_data/compare_data.py
      └── 生成数据（gen_data.py）与 Kernel 输出逐元素比对 MARE / MERE
[4] op_api UT        ./build/op_api/relu_op_api_ut    （`op_api/` 目录存在时才执行）
```

`tests/ut/common/` 下提供了完整的伪造设施：`tiling_context_faker`、`infershape_context_faker`、`tiling_case_executor`、`infershape_case_executor`、`any_value.h`。

### 6.2 用例设计

#### op_host Tiling UT

| 项 | 值 |
|----|----|
| 用例名 | `relu_0` |
| 输入 | shape `{45, 2048}`，`DT_FLOAT`，`FORMAT_ND`（共 92160 元素） |
| 输出 | shape `{45, 2048}`，`DT_FLOAT`，`FORMAT_ND` |
| 平台参数 | `maxAIVNum = 64`、`ubSize = 262144`、`tilingDataMaxSize = 4096`、soc `Ascend910B` |
| 期望 tilingKey | `1`（float → `RELU_TPL_SCH_MODE_1`） |
| 期望 workspace | `{0}` |

选择 `{45, 2048}` 而非题面的 `{8, 2048}`，是为了贴合工程自带用例并覆盖「元素数为 2048×45、核数回退到 64」的切分路径（`92160 = 64 × 1440`，`1440 % 8 == 0`，一次命中整除+对齐条件）。

**已知问题（待修正）**：用例中的 `expectTilingData` 为 `"0 1 0 "`，恰好等于 `ReluTilingData` 三个字段的**默认值**（`totalNum=0, blockFactor=1, ubFactor=0`），而不是按上面这组平台参数算出的实际结果。按 `relu_tiling.cpp` 的逻辑推算：

```
totalNum     = 45 × 2048 = 92160
usedCoreNum  = min(ceil(92160/1024), 64) = min(90, 64) = 64
               92160 % 64 == 0 且 1440 % 8 == 0  → 一次命中，不回退
blockFactor  = 92160 / 64 = 1440
maxUbFactor  = (262144 - 8192) / (4 × 4) = 15872，对齐后 15872
ubFactor     = min(1440, 15872) = 1440
→ 序列化结果应为 "92160 1440 1440 "
```

即该用例的 `expectTilingData` **仍是初始模板的占位值**（用例文件中保留的注释 `// TODO: 以下期望值基于初始模板实现，修改 tiling 逻辑后请更新` 也印证了这一点），而 `tiling_case_executor.cpp` 中执行的是 `EXPECT_EQ(tilingDataResult, expectTilingData)` 全串比对，因此该断言无法通过。**提交前应将期望值更新为 `"92160 1440 1440 "`。**（`expectTilingKey = 1`、`expectWorkspaces = {0}` 两项与实现一致，无需修改。）

#### op_kernel UT

| 项 | 值 |
|----|----|
| 用例名 | `ReluKernelTest.test_kernel_run` |
| 元素数 | `92160`（`45 × 2048`） |
| 输入数据 | 全 1 的 float32 |
| TilingData | `totalNum = 92160`、`blockFactor = 92160`、`ubFactor = 92160` |
| TilingKey | `1` |
| 核数 | `1`（`numBlocks = 1`，`AIV_MODE`） |
| 输出 | `float32_output_relu_0.bin` |

该用例刻意把 `ubFactor` 设成等于总元素数（大于 `blockFactor`），正是 3.4.1 中「夹紧 ① / 夹紧 ②」要保护的场景：Kernel 会把 `ubLength_` 夹到 `min(blockFactor, 4096) = 4096`，UB 实际申请 `4 × 4096 × 4 = 64 KB`，可安全执行。

#### 精度比对

`relu_data/gen_data.py` 生成输入与 golden：

```python
def impl(x):
    dtype = x.dtype
    return np.maximum(0, x.astype(np.float64)).astype(dtype)   # float64 往返，对 fp16/fp32 无损
```

`compare_data.py` 按社区标准取阈值：

| dtype | threshold |
|-------|-----------|
| FLOAT16 | $2^{-10} \approx 9.77\times10^{-4}$ |
| FLOAT32 | $2^{-13} \approx 1.22\times10^{-4}$ |
| BFLOAT16 | $2^{-7} \approx 7.81\times10^{-3}$ |

判据为 MARE（最大相对误差）与 MERE（平均相对误差）同时低于阈值。

#### aclnn 样例

`examples/test_aclnn_relu.cpp` 走标准两段式接口：

```
Init(aclInit → aclrtSetDevice → aclrtCreateStream)
  → CreateAclTensor（手工完成 FloatToHalf / FloatToBFloat16 的宿主侧转换）
  → aclnnReluGetWorkspaceSize(x, y, &workspaceSize, &executor)
  → aclrtMalloc(workspace)
  → aclnnRelu(workspaceAddr, workspaceSize, executor, stream)
  → aclrtSynchronizeStream
  → 释放资源
```

输入为 shape `{45, 2048}`、全 1 的 float32。

### 6.3 测试覆盖缺口

| 缺口 | 说明 | 建议补充 |
|------|------|---------|
| **负值路径未覆盖** | op_kernel UT 与 aclnn 样例的输入都是**全 1**（`std::vector<float> xHost(92160, 1)` / `xHostData(92160, 1)`），而 `relu(1) = 1`。也就是说：一个把输入原样输出的空实现（恒等映射）也能通过这些用例，**截断负值这个核心行为没有被任何用例证伪** | 输入改为含正、负、零、`±inf`、`NaN` 的混合数据；至少保证正负各占一半 |
| dtype 覆盖 | 只测了 float32，`schMode = 0` 的 half 路径没有用例 | 增加 fp16 用例 |
| 非整除 shape | 只有 `45 × 2048`（恰好能被 64 整除且对齐） | 增加质数长度、非 32B 对齐长度，覆盖核数回退循环 |
| 小规模 / 空输入 | 无 | 增加 `totalNum` 接近 1、小于 `MIN_SPLIT_THRESHOLD` 的用例 |
| UT 期望值过期 | 见 6.2 | 更新 `expectTilingData` |

---

## 7. 交付件清单

| 文件 | 说明 |
|------|------|
| `build.sh` | 构建脚本（构建 / 清理 / 跑 UT / 跑 example） |
| `CMakeLists.txt` | 顶层构建（`npu_op_package(relu_custom, TYPE RUN)`，支持 ascend910b / ascend910_93 / ascend950） |
| `op_host/relu_def.cpp` | 算子原型注册 |
| `op_host/relu_infershape.cpp` | 形状推导 |
| `op_host/relu_tiling.cpp` | Tiling 实现（平台查询 / 核间切分 / UB 切分 / TilingKey） |
| `op_kernel/relu.cpp` | Kernel 入口（`if constexpr` 按 schMode 分派） |
| `op_kernel/relu.h` | `NsRelu::Relu<T>` 类实现 |
| `op_kernel/relu_tiling_data.h` | `ReluTilingData` |
| `op_kernel/relu_tiling_key.h` | `schMode` 模板参数声明 |
| `tests/ut/**` | Host Tiling UT、Kernel CPU 桩 UT、数据生成与精度比对 |
| `examples/**` | aclnn 两段式调用样例 |
| `README.md` | 提交信息 |
