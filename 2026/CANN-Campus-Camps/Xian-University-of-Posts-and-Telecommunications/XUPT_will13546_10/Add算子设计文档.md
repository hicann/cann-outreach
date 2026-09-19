# Add 算子设计文档

## 1. 概述

### 1.1 基本信息

| 项目 | 内容 |
|-----|------|
| 算子名称 | Add（目录 `op_10_add`，算子原型名 `Add`） |
| 算子类别 | Elementwise（逐元素二元运算） |
| 支持数据类型 | FP32 / FP16 |
| 支持格式 | ND |
| 目标芯片 | Ascend910B |
| 目标架构 | dav-2201（arch22） |
| 开发形态 | 标准自定义算子工程（`op_host` + `op_kernel`，打包为算子包） |
| 提交团队 | 西安邮电大学 |
| 提交者 | 刘昊（will13546） |

### 1.2 算子功能

Add 计算两个等形状张量的逐元素和。输入 `x`、`y` 与输出 `z` 的 shape、dtype 完全一致，元素一一对应，无广播语义。

### 1.3 数学公式

$$
z_i = x_i + y_i,\quad i = 0,1,\dots,N-1
$$

其中 $N$ 为张量元素总数（按行主序展平后计）。

---

## 2. 架构设计

### 2.1 逻辑视图

| 模块 | 职责 | 核心文件 |
|------|------|---------|
| **op_host** | 算子原型注册、Shape/DataType 推导、核数与分片数估算 | `add.cpp` |
| **op_kernel** | Kernel 入口、核间偏移定位、核内分片、非对齐搬运与向量化计算 | `add.cpp`、`add_tiling.h`、`tiling_key_add.h` |

与同批次的 `op_02_mul`（Gelu）相同，本工程把算子原型、`InferShape`/`InferDataType` 与 Tiling 实现**合并写在同一个 `op_host/add.cpp`** 中（分别位于 `optiling` / `ge` / `ops` 三个命名空间），而不拆成 `*_def.cpp` / `*_infershape.cpp` / `*_tiling.cpp`。

模块依赖：

```
op_host/add.cpp
  ├── register/op_def_registry.h         (OpDef / OP_ADD / IMPL_OP_OPTILING)
  ├── tiling/platform/platform_ascendc.h (核数查询)
  └── ../op_kernel/add_tiling.h          (AddTilingData，Host/Device 共享契约)
      ../op_kernel/tiling_key_add.h      (DT_X 模板参数声明)

op_kernel/add.cpp
  ├── kernel_operator.h                  (Ascend C 编程接口)
  ├── add_tiling.h                       (AddTilingData)
  └── tiling_key_add.h                   (DT_X)
```

### 2.2 开发视图

```
op_10_add/
├── CMakeLists.txt              # npu_op_package(custom, TYPE SHARED, ascend910b)
├── README.md
├── op_host/
│   ├── CMakeLists.txt          # npu_op_code_gen + cust_optiling(TILING) + cust_opapi(ACLNN)
│   └── add.cpp                 # 原型定义 + InferShape/InferDataType + TilingFunc
└── op_kernel/
    ├── CMakeLists.txt          # npu_op_kernel_sources + npu_op_kernel_library
    ├── add.cpp                 # KernelAdd<DT_X> 类 + __global__ add 入口
    ├── add_tiling.h            # AddTilingData
    └── tiling_key_add.h        # ASCENDC_TPL_ARGS_DECL/SEL
```

本目录**未包含** `tests/`（UT）与 `examples/`（aclnn 调用样例），验证依赖提交平台的标准用例流程。

### 2.3 运行视图

**数据流**（每个核内部）：

```
GM：x, y
  │  DataCopyPad (GM → UB，带对齐填充，支持任意字节数)
  ▼
UB：inQueueX / inQueueY (VECIN，各双缓冲)
  │
  │  Add(z_local, x_local, y_local, current_length)
  ▼
UB：outQueueZ (VECOUT，双缓冲)
  │  DataCopyPad (UB → GM)
  ▼
GM：z
```

**执行流程**：

```
Host 侧 TilingFunc()：
  ├── GetCoreNumAiv()                    → block_dim（total_length < block_dim 时收缩）
  ├── total_length = tensor_x.GetShapeSize()
  ├── tile_num = ceil(ceil(total_length / block_dim) / TILE_LENGTH)   仅用于估算分片数
  ├── ASCENDC_TPL_SEL_PARAM(context, dt_x)   按 dtype 选模板
  ├── 写 AddTilingData{totalLength, tileNum}
  ├── SetBlockDim(block_dim) / workspace = 0
  └── 返回

Device 侧 add<DT_X>()：
  └── KernelAdd<DT_X> op; op.Init(x, y, z, tiling_data); op.Process();
        Init()    : 按 GetBlockIdx/GetBlockNum 重新切分 → 反解 tileLength_ → SetGlobalBuffer → InitBuffer
        Process() : for tile : CopyIn → Compute → CopyOut
```

---

## 3. 实现方案

### 3.1 模板划分

模板参数由 `op_kernel/tiling_key_add.h` 声明：

```cpp
ASCENDC_TPL_ARGS_DECL(Add,
    ASCENDC_TPL_DATATYPE_DECL(DT_X, C_DT_FLOAT, C_DT_FLOAT16),
);

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_DATATYPE_SEL(DT_X, C_DT_FLOAT, C_DT_FLOAT16),
    ),
);
```

| 模板 | 触发条件 | 实例化类型 |
|-----|---------|-----------|
| FP32 | `x.dtype == DT_FLOAT` | `KernelAdd<float>` |
| FP16 | `x.dtype == DT_FLOAT16` | `KernelAdd<half>` |

Host 侧通过 `ASCENDC_TPL_SEL_PARAM(context, dt_x)` 把 `ge::DataType` 的枚举值传给框架完成实例匹配。注意声明顺序 `C_DT_FLOAT, C_DT_FLOAT16` 与 `op_host/add.cpp` 中 `DataType({ge::DT_FLOAT, ge::DT_FLOAT16})` 的顺序**一一对应**，三个输入输出字段（`x` / `y` / `z`）也都保持同序，否则会出现「按 fp32 编译、按 fp16 传参」的错配。

### 3.2 TilingData 结构体

**文件位置**：`op_kernel/add_tiling.h`

```cpp
struct AddTilingData {
    uint32_t totalLength;  // 输入/输出总元素个数
    uint32_t tileNum;      // 每个核上需要循环处理的块数
};
```

只有两个字段，是本批次四个算子中 TilingData 最精简的。核间切分所需的 `blockDim` 通过 `SetBlockDim` 单独下发，不再冗余放进 TilingData；核内分片的**实际 tile 长度**不下发，而是由 Kernel 用 `tileNum` 反解（见 3.4.1）。

### 3.3 Tiling 计算逻辑

**文件**：`op_host/add.cpp`

#### 3.3.1 核数决策

```cpp
uint32_t num_cores_aiv = platform.GetCoreNumAiv();
if (num_cores_aiv == 0) num_cores_aiv = 1;      // 平台查询异常时兜底

uint32_t total_length = tensor_x->GetShapeSize();
uint32_t block_dim = num_cores_aiv;
if (total_length == 0)        block_dim = 1;              // 空张量：单核空跑
else if (total_length < block_dim) block_dim = total_length;  // 元素数少于核数：一核一元素
```

`total_length < block_dim` 时的收缩是必须的：若不收缩，后 `block_dim - total_length` 个核会分到 0 个元素且 `base` 仍然为 0，虽不至于越界，但会产生无意义的核启动。

#### 3.3.2 分片数估算

```cpp
constexpr uint32_t TILE_LENGTH = 2048;              // 仅用于 Host 侧估算
uint32_t block_length = ceilDiv(total_length, block_dim);
uint32_t tile_num    = ceilDiv(block_length, TILE_LENGTH);
if (tile_num == 0) tile_num = 1;
```

这里 `TILE_LENGTH = 2048` 在源码注释中被明确标注「**仅用于 Host 侧估算 tileNum**」：Host 只决定「每个核分几片」，**不决定每片的字节数**。真正决定每次搬运多少元素的是 Kernel 侧的 `tileLength_` 反解，这样 Host 就不需要知道 UB 容量和 dtype 大小，职责更单纯。

#### 3.3.3 其他配置

```cpp
uint32_t dt_x = static_cast<uint32_t>(tensor_x->GetDataType());
ASCENDC_TPL_SEL_PARAM(context, dt_x);               // 模板参数

tiling->totalLength = total_length;
tiling->tileNum     = tile_num;
context->SetBlockDim(block_dim);
current_workspace[0] = 0;                            // 本算子不需要 workspace
```

入参指针均做了空检查：`context == nullptr` 与 `tensor_x == nullptr` 都直接返回 `GRAPH_FAILED`。

### 3.4 Kernel 实现

#### 3.4.1 Init：核间切分与 tile 反解

Kernel **不直接使用** Host 的 `block_length`，而是用 `GetBlockIdx()` / `GetBlockNum()` 现场重算，保证「实际启动的核数」与「参与切分的核数」永远一致：

```cpp
uint32_t block_idx = AscendC::GetBlockIdx();
uint32_t block_num = AscendC::GetBlockNum();
if (block_num == 0) block_num = 1;                          // 兜底

uint32_t base      = tiling.totalLength / block_num;         // 每核至少分到的元素数
uint32_t remainder = tiling.totalLength % block_num;         // 前 remainder 个核各多分 1 个
blockLength_ = base + (block_idx < remainder ? 1 : 0);
uint32_t offset = block_idx * base + (block_idx < remainder ? block_idx : remainder);
```

偏移公式的含义：核 $k$ 之前有 $\min(k, \text{remainder})$ 个「多 1 个元素」的核，因此

$$
\text{offset}(k) = k \times \text{base} + \min(k,\ \text{remainder})
$$

各核区间首尾相接，总覆盖长度恰为 `totalLength`。

随后按「Host 给的分片数」反解每片长度，并对齐到 32 B：

```cpp
uint32_t align_elements = BLOCK_BYTES / sizeof(DT_X);        // 32B 对齐的元素数
uint32_t raw_tile_length = ceilDiv(blockLength_, tiling.tileNum);
tileLength_ = ceilAlign(raw_tile_length, align_elements);     // 向上对齐
tileNum_    = ceilDiv(blockLength_, tileLength_);             // 按新长度重算片数
```

**为什么反解出来的 `tileLength_` 要向上对齐？** 因为除最后一片外，每片都走 `DataCopyPad` 搬运 `tileLength_` 个元素；让 `tileLength_` 是 32 B 的整数倍，前 `tileNum_ - 1` 片就都是整块搬运，只有尾片需要 padding 处理 —— 把「非对齐」集中在唯一一处。

反解后 `tileNum_` 可能比 Host 下发的 `tileNum` 少 1（因为 `tileLength_` 被向上取整放大了），这不影响正确性：`Process()` 用的是反解后的 `tileNum_`。

最后建立 GM 窗口并申请双缓冲队列：

```cpp
xGm.SetGlobalBuffer((__gm__ DT_X *)x + offset, blockLength_);
yGm.SetGlobalBuffer((__gm__ DT_X *)y + offset, blockLength_);
zGm.SetGlobalBuffer((__gm__ DT_X *)z + offset, blockLength_);

pipe.InitBuffer(inQueueX,  BUFFER_NUM, tileLength_ * sizeof(DT_X));
pipe.InitBuffer(inQueueY,  BUFFER_NUM, tileLength_ * sizeof(DT_X));
pipe.InitBuffer(outQueueZ, BUFFER_NUM, tileLength_ * sizeof(DT_X));
```

#### 3.4.2 Process：分片主循环

```cpp
if (blockLength_ == 0 || tileNum_ == 0) return;      // 空核提前返回

for (uint32_t i = 0; i < tileNum_; ++i) {
    uint32_t start = i * tileLength_;
    uint32_t current_length = CurrentLength(start);  // = min(tileLength_, blockLength_ - start)
    if (current_length == 0) continue;
    CopyIn(start, current_length);
    Compute(current_length);
    CopyOut(start, current_length);
}
```

`CurrentLength()` 承担尾片收缩，并用 `start >= blockLength_` 的下界保护防止无符号下溢：

```cpp
__aicore__ inline uint32_t CurrentLength(uint32_t start) const {
    if (start >= blockLength_) return 0;
    uint32_t remain = blockLength_ - start;
    return remain < tileLength_ ? remain : tileLength_;
}
```

#### 3.4.3 CopyIn / CopyOut：DataCopyPad 处理非对齐

本算子的关键实现细节是**使用 `DataCopyPad` 而非 `DataCopy`**：

```cpp
uint32_t copy_bytes = current_length * sizeof(DT_X);
AscendC::DataCopyParams   copy_params{1, static_cast<uint16_t>(copy_bytes), 0, 0};
AscendC::DataCopyPadParams pad_params{false, 0, 0, 0};

AscendC::DataCopyPad(x_local, xGm[start], copy_params, pad_params);
AscendC::DataCopyPad(y_local, yGm[start], copy_params, pad_params);
```

`DataCopyParams{blockCount, blockLen, srcStride, dstStride}` 中 `blockCount = 1`（单段连续搬运）、`blockLen = copy_bytes`（**以字节为单位**，不要求 32 B 对齐），`src/dstStride` 均为 0。

`DataCopyPadParams{isPad, leftPadding, rightPadding, paddingValue}` 中 `isPad = false`：**不做主动填充**，仅由硬件在搬运非 32 B 对齐的尾段时自动处理。这样：

- **读方向**：尾片不足 32 B 的部分由硬件补零，不会读到张量之外的数据；
- **写方向**：`DataCopyPad` 只写 `copy_bytes` 个字节，**不会向输出张量末尾之外写出**。

这正是本算子与同批次 `op_01_sub` 的关键差异：`op_01_sub` 用 `DataCopy` + 32 B 对齐切分，代价是（a）尾部不足一个对齐块的元素不参与计算、（b）对齐 padding 会造成越界写窗口；`op_10_add` 用 `DataCopyPad` 后，**任意元素数都能被完整、精确地处理**。

#### 3.4.4 Compute

```cpp
AscendC::LocalTensor<DT_X> x_local = inQueueX.DeQue<DT_X>();
AscendC::LocalTensor<DT_X> y_local = inQueueY.DeQue<DT_X>();
AscendC::LocalTensor<DT_X> z_local = outQueueZ.AllocTensor<DT_X>();

AscendC::Add(z_local, x_local, y_local, static_cast<int32_t>(current_length));

outQueueZ.EnQue<DT_X>(z_local);
inQueueX.FreeTensor(x_local);
inQueueY.FreeTensor(y_local);
```

`Add` 的 count 参数为 `int32_t`，此处显式转换以免 `uint32_t` 隐式转换告警。由于 `inQueueX` / `inQueueY` / `outQueueZ` 三个队列在 `Init()` 中一次性申请，`Compute` 内的三次操作都是纯队列操作，无额外内存管理开销。

### 3.5 API 映射

| 计算步骤 | Ascend C API | 参数签名 | 约束说明 |
|---------|-------------|---------|---------|
| 数据搬入 | `DataCopyPad<DT_X>` | `(dst, src, DataCopyParams{1, bytes, 0, 0}, DataCopyPadParams{false,0,0,0})` | `blockLen` 为字节数，**不要求 32B 对齐** |
| 加法 | `Add<DT_X>` | `(dst, src0, src1, int32_t count)` | count 为元素数 |
| 数据搬出 | `DataCopyPad<DT_X>` | `(dst, src, DataCopyParams{1, bytes, 0, 0})` | 只写 actual 字节数，不越界 |
| 队列管理 | `AllocTensor` / `EnQue` / `DeQue` / `FreeTensor` | — | VECIN ×2 / VECOUT ×1 |
| 线程索引 | `GetBlockIdx()` / `GetBlockNum()` | — | 核间切分依据 |
| Tiling 解析 | `GET_TILING_DATA_WITH_STRUCT` | `(AddTilingData, tiling_data, tiling)` | 配合 `REGISTER_TILING_DEFAULT` |

### 3.6 内存管理（UB 占用）

| 缓冲区 | 大小 | 说明 |
|-------|------|------|
| `inQueueX` | `tileLength_ × sizeof(DT_X) × 2` | VECIN，双缓冲 |
| `inQueueY` | `tileLength_ × sizeof(DT_X) × 2` | VECIN，双缓冲 |
| `outQueueZ` | `tileLength_ × sizeof(DT_X) × 2` | VECOUT，双缓冲 |
| **合计** | `6 × tileLength_ × sizeof(DT_X)` | 6 = 3 队列 × 2 缓冲 |

**容量验证**（fp32）：`tileLength_` 由 `ceil(ceil(T/I)/K)` 向上对齐得到，其中 `K = ceil(block_length / 2048)`，因此典型值为 2048（对齐后不超过 2048 + 8）。取 `tileLength_ = 2056` 的上界估算：

```
3 × 2 × 2056 × 4 B = 49.3 KB  <  192 KB (Ascend910B 单核 UB)   ✅
```

fp16 下为 24.7 KB。任何输入规模下 `tileLength_` 都不会随总元素数增长（它只与「每核元素数 / tileNum」有关，而 `tileNum` 正比于每核元素数），因此 UB 占用是**有界的常数**，输入变大只会增加循环次数。

---

## 4. 性能优化

### 4.1 并行策略

- **满核启动**：`block_dim` 默认取平台全部 AIV 核，仅在元素数少于核数时才收缩。
- **核间粒度最细**：切分粒度是**单个元素**（`base` + 余数分摊），核间负载差最大为 1 个元素 —— 比同批次 `op_01_sub`（差最多一个 32 B 块）、`op_02_mul`（差最多 32 个元素）都更细。
- **偏移零开销**：`offset` 在 `Init()` 中算一次，`Process()` 内的地址都是 `start = i * tileLength_` 的简单乘加。

### 4.2 流水线设计

- **双缓冲**：三队列各 2 块缓冲，第 $i$ 片的 `CopyIn`（MTE2）、第 $i-1$ 片的 `Compute`（Vector）、第 $i-2$ 片的 `CopyOut`（MTE3）可三级重叠。
- **整块 + 尾片分离**：前 `tileNum_ - 1` 片走整块搬运、尾片走 `DataCopyPad` 收缩，热路径上没有「是否对齐」的判断。

### 4.3 可优化空间

| 方向 | 说明 | 代价 |
|------|------|------|
| **Host 与 Kernel 的 tile 尺寸认知不一致** | Host 用固定的 2048 估算 `tileNum`，Kernel 反解出实际 `tileLength_`。当 `tileLength_` 被向上对齐放大后，Kernel 的 `tileNum_` 会比 Host 下发的 `tileNum` 少 1 | 若让 Host 按真实 UB 容量与 dtype 反解 `tileLength` 并直接下发，可省掉 Kernel 侧的反解逻辑，但 Host 就必须感知 UB 大小与 dtype 尺寸 |
| **`tileLength_` 恒定 2048 附近，与 UB 容量无关** | 当前 tile 尺寸由 Host 的 `TILE_LENGTH` 常量锁定，而非按 UB 容量最大化 | 小 dtype（fp16）下 UB 只用了 24.7 KB，有 3 倍以上的余量可加大 tile、进一步摊薄同步开销 |
| **未做 vec 对齐优化** | `Add` 的 count 为任意元素数，最后一片可能触发硬件的尾块掩码处理 | 可把尾片单独用 `DataCopyPad` 搬成对齐长度后统一满宽计算 |

---

## 5. 风险与约束

### 5.1 功能约束

| 约束 | 说明 |
|------|------|
| 仅 FP32 / FP16 | 原型注册、TilingKey 声明、Kernel 模板均只覆盖这两种 dtype |
| 仅 ND 格式 | 原型中 `Format({ge::FORMAT_ND, ge::FORMAT_ND})`，无 `UnknownShapeFormat` 声明 |
| 不支持广播 | `InferShape` 直接复制第 0 个输入的 shape：`*output_shape = *input_shape`。**三个张量的元素数必须完全一致**，否则越界 |
| 无属性参数 | — |

### 5.2 精度风险

| 风险 | 影响 | 说明 |
|-----|------|------|
| 无 | — | 加法是单次精确运算，fp32/fp16 下的误差仅来自硬件加法本身的舍入，与参考实现 `x + y` 的语义一致 |
| fp16 累加 | 不适用 | 本算子每个输出元素只做一次加法，不存在累加链，无累积误差 |

### 5.3 边界与越界风险

| 场景 | 行为 | 评价 |
|------|------|------|
| `total_length = 0` | `block_dim = 1`，`tile_num = 1`；Kernel 中 `base = 0`、`remainder = 0`、`blockLength_ = 0` → `Process()` 立即返回 | ✅ 安全（输出未定义但无访存） |
| `total_length < block_dim` | 收缩 `block_dim = total_length`，每核 1 个元素；`tileLength_` 反解为 1 个对齐单位（fp32 下 8 个元素） | ✅ 安全；UB 有少量浪费（申请 8 个元素只搬 1 个） |
| `total_length` 为质数 | `base`、`remainder` 精确分摊，核间差 1 个元素；尾片由 `DataCopyPad` 精确处理 | ✅ 安全，**这是本算子相对 `op_01_sub` 的设计优势** |
| 非 32 B 对齐长度 | `DataCopyPad` 处理，读方向硬件补零、写方向精确字节数 | ✅ 无越界读写 |
| `y` 与 `x` 形状不一致 | `InferShape` 只校验第 0 个输入，Kernel 按同一 `offset`/`blockLength_` 访问 `y` | ⚠️ **越界风险**：若 `y` 比 `x` 短会越界读。属于「输入规格由调用方保证」的既有前提 |
| 原地（`z` 复用 `x`） | 未被禁止，但 `CopyIn` 先于 `CopyOut`，同一 tile 内不会读写冲突；跨 tile 时 `z` 的已写区域与 `x` 的未读区域不重叠 | ⚠️ 隐式支持但未声明，不建议依赖 |

### 5.4 工程一致性风险

- `op_host/add.cpp` 的 `InferShape` / `InferDataType` 为空指针检查齐全，但 `TilingFunc` 中通过 `context->GetTilingData<AddTilingData>()` 取得的 `tiling` 指针**未做空检查**，直接解引用写入 `tiling->totalLength`。若框架返回空指针会崩溃。建议补充 `OP_CHECK_NULL_WITH_CONTEXT`。
- 未提交 `tests/`，Host 侧 Tiling 的边界行为（尤其是 `total_length < block_dim` 与大规模输入）缺少可回归的验证手段。

---

## 6. 测试方案

本目录未提交 `tests/` 与 `examples/`，验证依赖提交平台的标准用例流程。若需本地自测，建议按下表补充用例：

| 用例类型 | 建议 shape | dtype | 关注点 |
|---------|-----------|-------|--------|
| 基础用例 | 与题面一致的标准 shape | fp32 / fp16 | 主流程正确性 |
| 非对齐边界 | 元素数为质数、$32k+1$、$32k+31$ | fp32 / fp16 | `DataCopyPad` 尾片路径 |
| 小规模 | 元素数 < 核数；元素数 = 1 | fp32 / fp16 | `block_dim` 收缩分支、一核一元素 |
| 空输入 | 元素数 = 0 | fp32 / fp16 | 不崩溃、不误写 |
| 负值与极值 | 含负数、`±inf`、`NaN` | fp32 / fp16 | 加法符号与特殊值传播 |
| 最大规模 | 接近显存上限 | fp32 | UB 占用不随规模增长（`tileLength_` 有界） |

精度判据（社区标准）：

| dtype | 阈值 |
|-------|------|
| FP16 | $2^{-10} \approx 9.77\times10^{-4}$ |
| FP32 | $2^{-13} \approx 1.22\times10^{-4}$ |

---

## 7. 交付件清单

| 文件 | 说明 |
|------|------|
| `CMakeLists.txt` | 顶层构建（`npu_op_package(custom, TYPE SHARED)`，`ASCEND_COMPUTE_UNIT=ascend910b`） |
| `op_host/CMakeLists.txt` | Host 侧编译（`cust_optiling` / `cust_opapi` 两个动态库） |
| `op_host/add.cpp` | 算子原型 + InferShape/InferDataType + Tiling |
| `op_kernel/CMakeLists.txt` | Kernel 侧编译（`ascendc_kernels` 二进制） |
| `op_kernel/add.cpp` | `KernelAdd<DT_X>` 类与 `__global__ add` 入口 |
| `op_kernel/add_tiling.h` | `AddTilingData` 结构体 |
| `op_kernel/tiling_key_add.h` | `DT_X` 模板参数声明与选择 |
| `README.md` | 提交信息 |
