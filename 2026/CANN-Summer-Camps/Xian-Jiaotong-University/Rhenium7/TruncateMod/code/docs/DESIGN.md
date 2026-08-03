# TruncateMod 算子技术设计（DESIGN.md）

> 版本：v1.1 ｜ 日期：2026-07-31 ｜ 设计者：Ascend C 算子架构设计
> v1.1 变更（设计串讲闭环）：mode2 输出走标准 outQueueY 路径；GM→UB DataCopyPad 改为 4 参原型；策略 2 的 UB Copy 改为 910b mask 形式并处理 repeatTime≤255；翻转概率表述改为数据分布相关；新增 x1/x2 dtype 一致性校验；修正策略 2 适用范围与策略 3 段数上界；明确标量输入分支与 blockIdx 越界防护。

---

## 0. 概述

### 0.0 需求类型判断

**通用**：未限定具体 shape，要求支持 BF16/FP16/FP32 三种 dtype 与任意合法广播组合。

### 0.1 基本信息

| 项目 | 内容 |
|-----|------|
| 算子名称 | TruncateMod（truncate_mod） |
| 算子类别 | Elementwise（二元逐元素）+ Broadcast |
| 需求类型 | 通用 |
| 支持数据类型 | BF16 / FP16 / FP32（op_def 已注册，输入 x1/x2 与输出 y 同 dtype） |
| 支持芯片 | ascend910b（本环境验证目标，Ascend 910B3）；op_def 同时注册 ascend950，方案对 950 兼容（见 §0.3） |
| 特殊约束 | 广播语义为 numpy 右对齐广播；除零按 IEEE 语义输出 Inf/NaN |

### 0.2 用户原始需求

| # | 需求内容 |
|---|---------|
| 1 | TruncateMod 为逐元素除法取模算子，数学定义 `y = x1 - trunc(x1 / x2) * x2`（trunc 为向 0 取整） |
| 2 | 支持数据类型 BF16 / FP16 / FP32 |
| 3 | 支持广播（x1、x2 可为不同 shape，按 numpy 规则广播到公共输出 shape） |
| 4 | 在既有脚手架上完成 Tiling 与 Kernel 实现，不新建工程（op_host / op_kernel / examples / tests 位于工作目录根） |
| 5 | 输出 docs/DESIGN.md 与 docs/PLAN.md 双文件 |

### 0.3 环境与架构（由 `/npu-arch` skill 查得）

| 项 | 值 | 来源 |
|----|----|------|
| 芯片型号 | Ascend 910B3 | docs/environment.md |
| SocVersion | ASCEND910B | docs/environment.md |
| NpuArch | **DAV_2201** | npu-arch skill 产品映射表（A2 系列） |
| `__NPU_ARCH__` | **2201** | 同上 |
| archXX | arch22 | 同上 |
| `--npu-arch` 候选编译参数 | **`dav-2201`** | asc-devkit `CMakeASCInformation.cmake`（`ascend910b` → `dav-2201`；vec/cube 无独立变体） |
| CANN 版本 | 8.5.2 | docs/environment.md |
| UB 容量 | 192 KB（DAV_2201 典型值；运行时经 `GetCoreMemSize(UB)` 获取，禁止硬编码） | npu-arch / 骨架 tiling |
| AIV 核数 | 运行时 `GetCoreNumAiv()` 获取 | 骨架 tiling |

> **路线决策（Step 0.5）**：算子类型为 Elementwise+Broadcast（vector 类），目标架构 `DAV_2201`（非 DAV_3510）→ 走**通用 SIMD/MemBase 路线**，不适用 Blaze/RegBase。Tiling 方法论采纳 `/ascendc-tiling-design` 的 **Elementwise 场景路由（elewise/patterns.md + elewise/tiling.md）** 与 **Broadcast 场景路由（broadcast/patterns.md → OneDim/行运行模型）**，并按本算子 API 约束做适配。
>
> **ascend950 兼容性说明**：本环境仅 Ascend 910B3 可用于验证；op_def 保留 ascend950 注册。本设计选用 API（Div/Mul/Sub/Cast/Duplicate/DataCopyPad/Copy）在 DAV_3510 均受支持，且 950 上 Div/Mul/Sub 额外支持 bfloat16_t 直算；统一走 FP32 计算链在 950 同样正确，故方案跨 910B/950 兼容。

---

## 1. 算子设计

### 1.1 数学公式

```
输入: x1 - shape[S1], dtype ∈ {BF16, FP16, FP32}
      x2 - shape[S2], dtype ∈ {BF16, FP16, FP32}（与 x1 同 dtype）
输出: y  - shape[S] = broadcast(S1, S2)（numpy 右对齐广播规则）, dtype 与输入一致

数学公式（逐元素）:
  q   = x1 / x2            // 商
  qt  = trunc(q)           // 向 0 取整（直接丢弃小数部分）
  y   = x1 - qt * x2       // 截断取模
```

**数值语义**：
- `trunc(v)`：向 0 截断。`trunc(3.9)=3`，`trunc(-3.9)=-3`。
- `x2 == 0`：商为 ±Inf/NaN，`trunc(±Inf)=±Inf`，`trunc(NaN)=NaN`，`y` 为 Inf/NaN —— 与 CPU golden（IEEE 语义）一致，不特殊处理。
- 注意：**不可**用 `Cast<float→int32>` 做截断实现 —— `|q| ≥ 2^31` 时 int32 溢出；故采用 `Cast<float→float, CAST_TRUNC>`（见 §1.2 验证）。

### 1.2 API 映射

#### 1.2.1 计算链（按 dtype 分支）

| 数学操作 | 对应 API | 关键参数 | 数据布局 | 910b 支持 |
|---------|---------|---------|---------|-----------|
| `x1/x2` 除法 | `Div<float>(qLocal, x1FLocal, x2FLocal, count)` | count=本次元素数 | VECIN/VECCALC，连续 | ✅ half/float（**不支持 bf16**） |
| `trunc(q)` 截断取整 | `Cast<float,float>(qLocal, qLocal, RoundMode::CAST_TRUNC, count)` | 同 dtype 转换 + TRUNC 舍入模式 | VECCALC，连续 | ✅ float→float TRUNC（**官方 Trunc 高阶接口在 220 架构的实现即为本调用**，见 `trunc_v220_impl.h`） |
| `qt*x2` 乘法 | `Mul<float>(qLocal, qLocal, x2FLocal, count)` | dst 可复用 qLocal（100% 重叠） | VECCALC | ✅ |
| `x1-p` 减法 | `Sub<float>(x1FLocal, x1FLocal, qLocal, count)` | dst==src0（100% 重叠） | VECCALC | ✅ |
| 半精度→FP32（mode0/1） | `Cast<float,T>(xF, xLocal, RoundMode::CAST_NONE, count)` | T=half/bf16，无损 | 队列→VECCALC | ✅ bf16→float CAST_NONE、half→float CAST_NONE |
| FP32→半精度（mode0/1） | `Cast<T,float>(outLocal, x1FLocal, RoundMode::CAST_RINT, count)` | 结果回写到输出 dtype | VECCALC→VECOUT | ✅ float→half/bf16 均支持 RINT |
| 标量展开 | `Duplicate<float>(xF, scalar, count)` | 标量输入（inTotal==1）时；xF 按 mode 取 FP32 工作缓冲（mode0/1）或输入队列缓冲（mode2） | VECCALC | ✅ half/bf16/float |
| 标量读取 | `GetValue<T>(localTensor, 0)` | 1 元素拷贝后取标量；T→float 用 C++ 显式转换（half/bfloat16_t 的转换运算符），非 API | — | ✅ |
| GM→UB 搬运 | `DataCopyPad<T>(dst, srcGM, DataCopyExtParams{...}, DataCopyPadExtParams<T>{...})` | **910b GM→UB 仅 4 参原型**（含 padParams，无 3 参）；blockLen 单位 1B；非对齐块自动 dummy 填充 | GM→VECIN | ✅ |
| UB→GM 搬运 | `DataCopyPad<T>(dstGM, src, DataCopyExtParams{...})` | 910b UB→GM 为 **3 参原型**；非对齐 dummy 在写出时**丢弃**，不写入 GM（无越界风险） | VECIN/VECOUT→GM | ✅ |
| UB→UB 重复扩展 | `Copy<T>(dst, src, mask, repeatTime, CopyRepeatParams{dstStride, srcStride, dstRepeatSize, srcRepeatSize})` | mask 连续模式=每次迭代元素数；repeatTime 为 **uint8_t**∈[0,255]；stride/repeatSize 单位 32B；**910b 无 3 参连续搬运 `Copy(dst,src,count)`**（仅 950） | VECIN/VECCALC | ✅（mask 形式） |

> **禁止项（已文档验证）**：
> - `Divs`/`Subs`（灵活标量位置）：910b **不支持**（仅 950）→ 标量除法/减法改用 `Duplicate` + `Div`/`Sub`。
> - `Div`/`Mul`/`Sub`/`Trunc` 在 910b 均**不支持 bfloat16_t** → BF16 必须升 FP32 计算（BF16 是 910b 的**硬性 API 约束**，不只是精度考虑）。
> - 高阶 `Trunc` 需 tmp 缓冲（`GetTruncMaxMinTmpSize`）；本设计直接使用其底层等价的 `Cast<float,float,CAST_TRUNC>`，省去 tmp 缓冲与 UB 占用。

#### 1.2.2 API 语义验证表

| API | 数据布局 | 功能需求 | API 选择 | 限制条件 | 匹配 | 文档 |
|-----|---------|---------|---------|---------|-----|------|
| `Div` | 连续 1D tile，count 个元素 | 逐元素除法 | `template<typename T> void Div(LocalTensor<T>& dst, LocalTensor<T>& src0, LocalTensor<T>& src1, int32_t count)`（无 config 原型） | 910b：T∈{half,float}；count=0 视为 NOP；dst/src0/src1 同 dtype | ✅ | `docs/zh/api/SIMD-API/基础API/Memory矢量计算/基础算术/Div.md` |
| `Cast` | 连续 1D tile | 类型转换 + 舍入 | `template<typename T, typename U> void Cast(LocalTensor<T>& dst, LocalTensor<U>& src, const RoundMode& roundMode, uint32_t count)` | 910b：bf16→float CAST_NONE；half→float CAST_NONE；float→half/bf16 RINT/TRUNC/...；float→float TRUNC 等 | ✅ | `.../类型转换/Cast.md`（910b 表）+ `impl/adv_api/detail/math/trunc/trunc_v220_impl.h` |
| `Mul` | 连续 1D tile | 逐元素乘 | `Mul(dst, src0, src1, count)` | 910b：T∈{int16,half,int32,float} | ✅ | `.../基础算术/Mul.md` |
| `Sub` | 连续 1D tile | 逐元素减 | `Sub(dst, src0, src1, count)` | 910b：T∈{int16,half,int32,float} | ✅ | `.../基础算术/Sub.md` |
| `Duplicate` | 连续 1D tile | 标量填充 | `Duplicate<T>(dst, scalar, count)`（Memory 版） | 910b：T∈{int16,uint16,half,bf16,int32,uint32,float} | ✅ | `.../数据填充/Duplicate.md` |
| `DataCopyPad` | GM→UB / UB→GM，非对齐 | 搬运+填充 | GM→UB（910b 仅 **4 参**）：`DataCopyPad<T>(dst, srcGM, DataCopyExtParams{blockCount, blockLen, srcStride, dstStride, rsv}, DataCopyPadExtParams<T>{})`；UB→GM（910b 为 **3 参**）：`DataCopyPad<T>(dstGM, src, DataCopyExtParams)` | blockLen 单位 1B、须为 sizeof(T) 整数倍；910b srcStride∈[0,2^32-1]（**不支持负 stride 重复同块**，重复扩展用 UB `Copy`）；`DataCopyPadExtParams<T>{}` 默认 isPad=false、pad=0 → 每个 blockLen 右侧自动 dummy 填充至 32B；UB→GM 写出时 dummy **丢弃**（`DataCopyPad（UBToGM）.md` 图2），不向 GM 写填充字节 | ✅ | `.../数据搬运/DataCopyPad（GMToUB非对齐数据搬运）.md`（函数原型）+ `.../数据搬运/DataCopyPad（UBToGM非对齐数据搬运）.md`（函数原型、图2）+ `include/basic_api/kernel_struct_data_copy.h`（padParams 默认值） |
| `Copy` | UB→UB | 掩码式搬运/重复 | `Copy<T>(dst, src, mask, repeatTime, CopyRepeatParams{dstStride, srcStride, dstRepeatSize, srcRepeatSize})`：mask 连续模式=每次迭代处理元素数；repeatTime 为 **uint8_t**（∈[0,255]）；各 stride/repeatSize 单位 DataBlock(32B) | srcRepeatSize=0 → 相邻迭代读同一 DataBlock（"重复同块"）；repeatTime>255 需分片多次调用；**910b 不支持 3 参连续搬运 `Copy(dst,src,count)`**（`Copy（UBToUB连续数据搬运）.md`：仅 950） | ✅ | `.../数据搬运/Copy（UBToUB掩码式高维数据搬运）.md` + `Copy（UBToUB连续数据搬运）.md` + tiling-design broadcast/ub-broadcast（srcRepeatSize=0 重复读同一行同款模式） |
| `GetValue` | 标量读取 | 取 LocalTensor 单元素 | `T GetValue(int32_t idx)` | idx 常量 | ✅ | `.../数据结构/LocalTensor/GetValue.md` |

**验证清单**：
- [x] 1. 数据布局确认：所有计算均为连续 1D tile（广播在搬运层解决，见 §2.4.3），32B 对齐由 DataCopyPad 兜底
- [x] 2. 功能需求明确：除法/截断/乘/减/升精度/回写，逐元素
- [x] 3. 已查阅官方文档：上述链接（asc-devkit，CANN 8.5.2）
- [x] 4. 匹配验证：dtype 约束（bf16 不可直算 → 升精度）、标量约束（Divs/Subs 不可用 → Duplicate）
- [x] 5. 已记录验证过程（§1.2.2 + §1.2.1 禁止项）

### 1.3 数据流

```
GM x1 (T) ──DataCopyPad(4 参)──▶ VECIN queue x1Local (T, 双缓冲)
GM x2 (T) ──DataCopyPad(4 参)──▶ VECIN queue x2Local (T, 双缓冲)
                              │ EnQue/DeQue
                              ▼
              [mode0/1] Cast<T→float> → x1F / x2F (VECCALC)
              [mode2]   x1Local / x2Local 直接作为 FP32 输入
                              ▼
              q = Div(x1F/x1Local, x2F/x2Local)         // 商
              q = Cast<float,float,CAST_TRUNC>(q)       // 截断
              q = Mul(q, x2F/x2Local)                   // q * x2
              y = Sub(x1F/x1Local, q)                   // x1 - q*x2
                              ▼
              [mode0/1] Cast<float→T, RINT> → VECOUT queue yLocal (T, 双缓冲)
              [mode2]   Sub(yLocal, x1Local, qF) 直接写 VECOUT queue yLocal (T, 双缓冲)
                              ▼
VECOUT queue yLocal ──DataCopyPad(3 参)──▶ GM y (T)
```

> 说明：**全模式统一走 VECOUT outQueueY 标准输出路径**（含 mode2，不做 VECIN 就地输出），EnQue/DeQue 提供 PIPE_V→MTE3 流水同步与缓冲生命周期管理（决策依据见 §2.4.1）。

### 1.4 核心计算步骤

```
1. CopyIn   —— 每输入按广播加载策略（§2.4.3）从 GM 取 tile 数据到 UB 队列（双缓冲）
2. 升精度   —— [mode0/1] Cast<T,float> 到 FP32 工作缓冲 x1F/x2F（标量输入跳过，已由 Duplicate 直接写满）；[mode2] 直接使用队列缓冲
3. 商       —— Div(qF, x1F/x1Local, x2F/x2Local)
4. 截断     —— Cast<float,float,CAST_TRUNC>(qF, qF)   // 等价官方高阶 Trunc（220 实现）
5. 乘积     —— Mul(qF, qF, x2F/x2Local)
6. 取模     —— [mode0/1] Sub(x1F, x1F, qF)；[mode2] Sub(yLocal, x1Local, qF)
7. 回写     —— [mode0/1] Cast<float,T,RINT>(yLocal, x1F)；[mode2] 结果已在 yLocal（VECOUT）
8. CopyOut  —— EnQue(yLocal) → DeQue(yLocal) → DataCopyPad(3 参) 从 VECOUT 队列到 GM y
```

**关键设计要点**：
1. **全链 FP32 计算**：BF16 因 910b API 硬约束必须升精度；FP16 升精度为**精度达标所必需**（FP16 直算时商舍入误差约 2^-11，`trunc` 翻转概率约 2^-11×|q|/元素，误差幅值 |x2| 会击穿 FP16 `max_abs_error_limit=0.1`；升 FP32 后翻转概率降至约 **2^-23×|q|**/元素，**翻转概率与 |q| 线性相关**，精度验收数据须保证期望翻转数 λ = N×2^-23×E|q| ≪ 1，详见 §2.3 精度策略）。
2. **就地复用**：`Cast(qF,qF)`（官方 Trunc 同款用法）、`Mul(qF,qF,x2F)`、`Sub(x1F,x1F,qF)`（mode0/1）均为 100% 完全重叠（通用地址重叠约束允许）；x1 原值保存在 x1F 不被覆盖，直至最终 Sub 结果。mode2 的最终结果写入独立的 VECOUT yLocal（与输入缓冲分离）。
3. **Buffer 复用**：qF 依次承载 商→截断→乘积；x1F 承载 升精度输入→最终结果。
4. **参数使用规则**：DataCopyPad blockLen 与计算 API count 用**有效长度**；UB 缓冲偏移/分配用**对齐长度**（分配按 256B 对齐容量，DataCopyPad 自动填充尾部）。
5. **标量输入处理**：标量输入（inTotal==1）在 CopyIn 用策略 0（GetValue + Duplicate）；mode0/1 的 Duplicate 目标为 FP32 工作缓冲 xF（Compute 跳过对应 Cast），mode2 的目标为输入队列缓冲（x1Local/x2Local），见 §2.4.3。

### 1.5 内存管理（Buffer 规划）

| Buffer | 用途 | 大小（元素 = ubFactor） | TPosition | 双缓冲 |
|--------|------|------------------------|-----------|--------|
| inQueueX | x1 tile 数据（T） | `ubFactor × sizeof(T)` × 2 | VECIN | ✅（BUFFER_NUM=2） |
| inQueueX2 | x2 tile 数据（T） | `ubFactor × sizeof(T)` × 2 | VECIN | ✅ |
| outQueueY | y tile 数据（T），**全模式（含 mode2）标准输出队列** | `ubFactor × sizeof(T)` × 2 | VECOUT | ✅ |
| x1F | x1 的 FP32 工作副本（mode0/1；mode2 直接用队列） | `ubFactor × 4B` | VECCALC | ❌ 单份 |
| x2F | x2 的 FP32 工作副本（mode0/1；mode2 直接用队列） | `ubFactor × 4B` | VECCALC | ❌ 单份 |
| qF | 商/截断/乘积工作缓冲（全模式） | `ubFactor × 4B` | VECCALC | ❌ 单份 |
| scalarBuf | 标量输入读取（32B 对齐，1 元素） | 32B | VECCALC | ❌ |

**总 UB 使用量**（每元素字节除数 `bufferDivisor`，`ubFactor` 由此反推）：
- **mode0/mode1**（T=2B）：队列 `2×(2+2+2)=12` + FP32 工作 `3×4=12` → `bufferDivisor = 24` B/elem
- **mode2**（T=4B）：队列 `2×(4+4+4)=24` + qF `4` → `bufferDivisor = 28` B/elem
- `maxElemNum = (ubSize - extraSize) / bufferDivisor`，`ubFactor = floor_align(maxElemNum, 256 / sizeof(T))`（256B repeat 对齐）；extraSize 预留约 1KB（含 scalarBuf 等）。

数值示例（UB=192KB）：mode0/1 → ubFactor ≈ 8064；mode2 → ubFactor ≈ 6976。

> **设计决策（mode2 输出路径）**：全模式统一使用 outQueueY（VECOUT）标准输出路径，**否决"mode2 y 就地写入 x1Local（VECIN）"方案**。原因：① 就地输出无 `outQueueY.EnQue/DeQue` 提供 PIPE_V 完成信号，MTE3（CopyOut）读 VECIN 需额外 `SetFlag/WaitFlag(V_MTE3)` 手工同步，易引入半写竞态；② inQueueX 为 BUFFER_NUM=2 轮转，队列机制只跟踪 MTE2→V 的完成、不跟踪 MTE3 是否读完，tile t+2 的 CopyIn（MTE2）会与 tile t 的 CopyOut（MTE3）竞争同一缓冲。标准 outQueueY 路径以 ~8 B/elem 的 UB 代价换取正确的流水同步与缓冲生命周期管理（详见 §2.4.1）。

---

## 2. 架构设计

### 2.1 多核切分策略

| 项目 | 说明 |
|-----|------|
| 切分维度 | 输出展平为 1D（广播在搬运层解决，切分不感知广播结构） |
| 单核任务量 | `blockFactor = ceil_align(ceil(totalNum / coreNum), 512)`（512 元素对齐） |
| 使用核数 | `coreNum = min(ceil(totalNum × sizeof(T) / 4KB), GetCoreNumAiv())`（每核至少 4KB 数据） |
| 负载均衡 | 首/尾核区分：`blockNum = ceil(totalNum / blockFactor)`；核偏移 `start = blockIdx × blockFactor`，尾核 `currentBlockLen = min(blockFactor, totalNum - start)` |
| 广播边界 | 无需按广播结构切核 —— 每核任意 [start, start+len) 区间均由 Kernel 按加载策略正确取值 |

### 2.2 UB 切分策略

| 项目 | 说明 |
|-----|------|
| UB 容量 | 192KB（DAV_2201；运行时 `GetCoreMemSize(UB)` 获取） |
| 单次处理数据量 | `ubFactor`（见 §1.5 公式；mode0/1 ≈ 8064，mode2 ≈ 6976 元素） |
| 是否需要分 chunk | 是：每核任务按 `ubFactor` 循环切 tile，尾 tile 用实际剩余元素数 `currentNum` |
| chunk 大小公式 | `ubFactor = floor_align((ubSize - extraSize) / bufferDivisor, 256 / sizeof(T))` |
| 广播对齐优化 | 若输入 `inTotal`（重复周期，见 §2.4.3）≤ maxElemNum，则令 `ubFactor` 取 `period` 的整数倍（`floor_align(maxElemNum/period)×period`），使 tile **长度**为整周期倍数；两输入 period 不同时优先取较小者（两者成整数倍时自然同时对齐）。**注意：tile 长度对齐 ≠ tile 起点对齐**——策略 2 是否触发还取决于 `blockStart % period`（§2.4.3 运行时条件 `s % period == 0`）；`blockFactor`（512 元素对齐）无法同时对齐两个输入的 period，未命中的核走策略 3 兜底 |

### 2.3 分支场景覆盖

| 分支 | 决策条件 | 处理策略 |
|------|---------|---------|
| 数据类型 | FP16 → mode0；BF16 → mode1；FP32 → mode2 | tilingKey（schMode 模板参数）三路分派，Kernel 模板 T=half / bfloat16_t / float；mode0/1 升 FP32 计算，mode2 直算 |
| 广播形态 1 | x1.shape == x2.shape（ELEWISE） | 双输入均连续搬运（策略 1 恒成立：innerLen=totalNum, rowCount=1） |
| 广播形态 2 | 单输入 inTotal==1（标量） | 该输入 1 元素搬运 + `GetValue` + `Duplicate<float>` 展开（mode0/1 目标 xF 工作缓冲并跳过对应 Cast；mode2 目标输入队列缓冲），跳过周期模型 |
| 广播形态 3 | 一般广播（innerLen < totalNum） | row-run 模型 + 加载策略 1/2/3（§2.4.3） |
| 对齐 | tile 元素数非 32B 倍数 | `DataCopyPad` 自动 dummy 填充，计算按有效 count 进行 |
| 边界（尾块） | 最后一块 < blockFactor | `currentBlockLen = totalNum - start` 单独计算 |
| 边界（尾 tile） | tile 剩余 < ubFactor | 循环内 `currentNum = min(ubFactor, 剩余)` |
| 数值边界 | x2 = 0 | 商 ±Inf/NaN，经 trunc/mul/sub 传播，与 CPU golden IEEE 语义一致 |
| 大商值 | \|q\| ≥ 2^31 | 不经过 int32 Cast，无溢出风险（FP32 全链） |
| 精度 | FP16/BF16 | 升 FP32 计算（见 1.4 关键要点 1），比对标准见 PLAN §3 |

**精度策略（对照 ops-precision-standard 浮点类）**：

| dtype | rtol | atol | max_abs_error_limit | 达标途径 |
|-------|------|------|--------------------|----------|
| FP16 | 2^-9 (1.95e-3) | 2^-9 | 0.1（max(1e-1, 32×2^-10)） | FP32 全链计算；商误差 ~2^-24，trunc 翻转概率 ~2^-23×\|q\|/元素，matched_ratio≈1，max_abs 由边界翻转幅值 \|x2\| 控制在远小于 0.1 的概率下 → 达标 |
| BF16 | 2^-6 (1.56e-2) | 2^-6 | 1.0 | 同上，容差更宽 |
| FP32 | 2^-10 (9.77e-4) | 2^-16 (1.53e-5) | 1e-2 | FP32 直算（Div 1ulp + 各 1 次 round），满足 rtol=2^-10 |
| 达标判定 | matched_ratio ≥ 0.99 且 max_abs ≤ limit | | | 混合容差单标杆（CPU 高精度 golden，PLAN §3） |

> **trunc 翻转概率与验收数据分布（重要，v1.1 修正）**：`trunc` 在整数边界不连续。商经有限精度除法后恰落在整数两侧 ±|q|×2^-24（FP32 Div 相对误差 1ulp = |q|×2^-24，两侧区间合计 2|q|×2^-24）区间内时，截断结果与 float64 golden 相差 1，进而 y 误差 ≈ |x2|（FP16 下可 >0.1、FP32 下 >1e-2，直接击穿 max_abs_error_limit）。单元素翻转概率 **P ≈ 2^-23 × |q|**，与商幅值线性相关——这是 trunc 类算子的固有属性，任何有限精度实现相对 float64 golden 均无法根除。**因此：**
> 1. **精度验收数据必须把期望翻转数 λ = N × 2^-23 × E|q| 控制在 ≪ 1**：验收脚本/compare_data.py 应限定数据域（限制 |x1/x2| 比值上界使 |q| 有界、x2 远离 0，或使用整数/有限小数域数据），避免"随机大范围数据 + N=1M"导致 λ≈1~6 的偶发 FAIL（flaky）；
> 2. **K8 类大商值用例仅做功能性校验**（验证无 int32 溢出路径、Inf/NaN 传播正确），不纳入精度指标——其 |q| 巨大，翻转几乎必然发生，属预期行为；
> 3. PLAN §6.3 精度验收按上述数据域约束执行。

### 2.4 类别特有设计

#### 2.4.1 分支：mode2 —— FP32 直算

**适用场景**：输入 dtype = FP32（tilingKey = MODE_2，T=float）。

**Compute 核心流程伪代码**（`TruncateMod<float>::Compute(currentNum)`，每 tile）：

```cpp
// qF 为 FP32 工作缓冲（VECCALC，ubFactor×4B）
// x1Local/x2Local 为 VECIN 队列缓冲（float），yLocal 为 VECOUT 队列缓冲（float，双缓冲）
// 1. 商
Div(qF, x1Local, x2Local, currentNum);
// 2. 向 0 截断（等价官方高阶 Trunc<float> 的 220 实现）
Cast<float, float>(qF, qF, RoundMode::CAST_TRUNC, currentNum);
// 3. 乘 x2
Mul(qF, qF, x2Local, currentNum);
// 4. y = x1 - p，写入 VECOUT 输出队列（不做 VECIN 就地输出，理由见下方决策）
Sub(yLocal, x1Local, qF, currentNum);
// 5. CopyOut 阶段：EnQue(yLocal) → DeQue(yLocal) → DataCopyPad(3 参)(yGm, yLocal, {1, currentNum×sizeof(float), 0, 0, 0})
```

**Buffer 需求**：inQueueX(双缓冲)、inQueueX2(双缓冲)、outQueueY(双缓冲)、qF。`bufferDivisor = 28` B/elem。

> **设计决策（对应 §1.5）**：**否决"y 就地写入 x1Local（VECIN）"方案**，统一走 outQueueY 标准输出路径。理由：
> 1. **V→MTE3 无同步**：标准队列模型中，输出缓冲的 PIPE_V 完成信号由 `outQueueY.EnQue/DeQue` 提供；就地方案下 CopyOut（MTE3）直接读 x1Local（VECIN），没有 EnQue 信号保证 PIPE_V 已写完，MTE3 可能读到半写数据（需额外 `SetFlag/WaitFlag(V_MTE3)` 手工同步）。
> 2. **缓冲复用竞态**：inQueueX 为 BUFFER_NUM=2 轮转。tile t 的 CopyIn(MTE2)→DeQue→Compute(V)→CopyOut(MTE3) 均占用同一缓冲；tile t+2 的 AllocTensor 复用该缓冲时，队列机制只跟踪 MTE2→V 的完成、不跟踪 MTE3 是否读完，若无串行化，tile t+2 的 MTE2 写入会与 tile t 的 MTE3 读取竞争。
> 3. 标准路径仅以 ~8 B/elem（mode2 的 outQueueY 双缓冲）UB 代价换取正确性，收益/复杂度比显著更优。

#### 2.4.2 分支：mode0 / mode1 —— 半精度升精度

**适用场景**：输入 dtype = FP16（mode0，T=half）或 BF16（mode1，T=bfloat16_t），tilingKey = MODE_0 / MODE_1。两个分支计算结构完全一致，仅 T 与 Cast 模板参数不同。

**Compute 核心流程伪代码**（`TruncateMod<half>::Compute` / `TruncateMod<bfloat16_t>::Compute`）：

```cpp
// x1F/x2F/qF 为 FP32 工作缓冲（VECCALC，ubFactor×4B）
// 1. 升精度（无损，CAST_NONE）；标量输入已在 CopyIn 由 Duplicate<float> 直接写满对应 xF（策略 0），跳过 Cast
if (!x1Scalar) Cast<float, T>(x1F, x1Local, RoundMode::CAST_NONE, currentNum);
if (!x2Scalar) Cast<float, T>(x2F, x2Local, RoundMode::CAST_NONE, currentNum);
// 2. 商 / 3. 截断 / 4. 乘积 / 5. 取模 —— 同 mode2，但基于 x1F/x2F
Div(qF, x1F, x2F, currentNum);
Cast<float, float>(qF, qF, RoundMode::CAST_TRUNC, currentNum);
Mul(qF, qF, x2F, currentNum);
Sub(x1F, x1F, qF, currentNum);
// 6. 回写输出 dtype（RINT：四舍六入五成双，与硬件默认转换一致）
Cast<T, float>(yLocal, x1F, RoundMode::CAST_RINT, currentNum);
// 7. CopyOut：EnQue(yLocal) → DeQue(yLocal) → DataCopyPad(3 参)(yGm, yLocal, {1, currentNum×sizeof(T), 0, 0, 0})
```

**标量输入（策略 0）细节**：标量输入不 EnQue/DeQue 对应输入队列（CopyIn 直接 Duplicate 写满 xF，Compute 不 DeQue 该输入）；`GetValue<T>(scalarBuf, 0)` 返回 T（half/bfloat16_t/float），转 float 标量用 C++ 显式转换 `float s = static_cast<float>(...);`（half/bfloat16_t 提供到 float 的转换运算符，属标量 C++ 语义、非 API），再 `Duplicate<float>(xF, s, currentNum)`。

**Buffer 需求**：inQueueX(双缓冲)、inQueueX2(双缓冲)、outQueueY(双缓冲)、x1F、x2F、qF。`bufferDivisor = 24` B/elem。

> 说明：计算链中 Cast/Div/Mul/Sub 全部为 FP32 类型，规避了 910b 上 Div/Mul/Sub/Trunc 不支持 bfloat16_t 的 API 约束（§1.2.1 禁止项）。

#### 2.4.3 分支：广播 —— row-run 模型与加载策略

**适用场景**：x1、x2 为任意 numpy 合法广播组合（含标量）。

**Host 侧描述（Tiling 阶段计算，写入 tiling data）**：对每个输入按输出 rank 右对齐后，从最内维向左扫描：

```
innerLen = 1
for d = rank-1 → 0:
    if 输入 dim[d] == 输出 dim[d]:  innerLen *= dim[d]     // 匹配段，纳入连续运行
    else:                            break                   // 首个广播轴（输入 dim=1, 输出>1）
rowCount = inTotal / innerLen        // 源行数（输入自身的行重复次数）
```

- 输出索引 i → 输入偏移：`srcOffset(i) = ((i / innerLen) % rowCount) * innerLen + (i % innerLen)`。
- 重复周期 `period = innerLen × rowCount = inTotal`（输出空间中输入完整重复一次的长度）。
- 各形态退化为统一描述：ELEWISE → `(innerLen=totalNum, rowCount=1)`；标量 → `(innerLen=1, rowCount=1)`。

**Kernel 加载策略（对每个输入、每个 tile [s, s+T)）**，伪代码：

```cpp
// T: 当前 tile 元素数（tail tile 取实际值）
if (inTotal == 1) {
    // 策略 0（标量）: 1 元素 DataCopyPad(4 参) 到 scalarBuf → GetValue<T> → float 标量（C++ 转换）
    //   → Duplicate<float>(目标, scalar, T)
    //   目标：mode0/1 为 FP32 工作缓冲 x1F/x2F（Compute 跳过对应 Cast）；mode2 为输入队列缓冲 x1Local/x2Local
    //   DataCopyPad<T>(scalarBuf, inGm, {1, sizeof(T), 0, 0, 0}, DataCopyPadExtParams<T>{});
} else if (s % innerLen + T <= innerLen) {
    // 策略 1（tile 位于单个源行内）: 连续搬运
    //   srcOff = ((s / innerLen) % rowCount) * innerLen + (s % innerLen)
    //   DataCopyPad<T>(local, inGm[srcOff], {1, T * sizeof(T), 0, 0, 0}, DataCopyPadExtParams<T>{})
} else if (s % period == 0 && T % period == 0 && (period * sizeof(T)) % 32 == 0) {
    // 策略 2（整周期重复）: 搬入一个周期到 tile 缓冲头部 [0, period)，再用 mask 形式 Copy 扩展到 [period, T)
    //   首份 [0, period) 即真实输入且不需回写 → 全部 Copy 迭代的写区域 [(i+1)P, (i+2)P) 与读区域 [0, P) 不相交
    //   → 满足「通用地址重叠约束」（单次迭代 100% 重叠或完全不相交；前序 dst 不与后序 src 重叠）
    //   DataCopyPad<T>(local, inGm, {1, period * sizeof(T), 0, 0, 0}, DataCopyPadExtParams<T>{});
    //   uint32_t k = T / period;                       // 重复倍数
    //   for (uint32_t off = 0; off < k - 1; off += 255) {   // repeatTime 为 uint8_t，>255 时按 255 分片
    //       uint32_t n = min(255u, k - 1 - off);
    //       Copy(local + (off + 1) * period, local, /*mask=*/period,
    //            /*repeatTime=*/(uint8_t)n, /*CopyRepeatParams=*/{1, 1, period * sizeof(T) / 32, 0});
    //       // {dstStride, srcStride, dstRepeatSize, srcRepeatSize}：迭代内连续读/写；
    //       // dstRepeatSize = P*sizeof(T)/32（相邻迭代写首块步长 → 连续写出）；srcRepeatSize = 0（每迭代读 [0,P)）
    //   }
} else {
    // 策略 3（跨周期逐段兜底）: tile 内按 period 边界切成连续段 [a, b)（b ≤ 下一个边界）
    //   每段源地址连续：srcOff = a % period（= ((a/innerLen)%rowCount)*innerLen + a%innerLen），段长 b-a
    //   for (seg 遍历 [s, s+T) 的各连续段):
    //     DataCopyPad<T>(local + segOff, inGm[segSrcOff], {1, segLen * sizeof(T), 0, 0, 0}, DataCopyPadExtParams<T>{})
}
```

- **全部 GM→UB DataCopyPad 均为 4 参**（`DataCopyPad<T>(dst, src, DataCopyExtParams, DataCopyPadExtParams<T>{})`，padParams 默认 isPad=false → 非对齐块自动右侧 dummy 填充至 32B，与设计兜底预期一致）；**UB→GM CopyOut 为 3 参**。
- 策略 2 触发条件为 `s % period == 0 && T % period == 0 && (period*sizeof(T)) % 32 == 0`：`period×sizeof(T)` 非 32B 对齐或 T 非 period 整数倍时退化为逐段（策略 3）。**由于 `blockFactor`（512 元素对齐）不保证是 period 的整数倍，仅 `blockStart % period == 0` 的核上策略 2 生效**（整核所有 tile 命中）；其余核所有 tile 走策略 3——正确性不受影响（策略 3 为通用兜底），性能影响见 §2.2 说明。
- 策略 3 段数：tile 长度 T = k×period 时，每 tile 段数 = k（s%period==0）或 k+1（s%period≠0），上界 **`T/period + 1`**（非 v1.0 的"最多 2 段"——该表述仅在 T ≤ 2×period 时成立）。
- **DataCopyPad 在 910b 的 srcStride ∈ [0, 2^32-1]（不支持负 stride 重复同块），故"重复同块"统一用 UB→UB mask 形式 `Copy`（srcRepeatSize=0）实现**（§1.2.2 验证；与 tiling-design broadcast/ub-broadcast"srcStride=0 重复读同一行"同款模式，910b 下该模式在 CopyRepeatParams.srcRepeatSize=0 上等价落地）。

**Buffer 需求**：广播输入在 `inTotal==1`（标量）时使用 `scalarBuf + Duplicate`，无额外 tile 缓冲；其余场景搬运目标即对应输入队列缓冲。策略 2 的扩展在 tile 缓冲内部完成（[0,P) 作读源、[P,T) 作写区），不增加缓冲。

---

## 3. Host 侧设计（Tiling / InferShape 变更）

### 3.1 InferShape（`truncate_mod_infershape.cpp`）

- **输入 dtype 一致性校验（v1.1 新增）**：`x1.dtype != x2.dtype` → 返回 `ge::GRAPH_FAILED`。op_def 中 x1/x2 的 dtype 集合各自独立（`{DT_BF16, DT_FLOAT16, DT_FLOAT}`），混合 dtype（如 x1=FP16、x2=FP32）若不做校验，tilingKey 仅按 x1 dtype 分派会以错误步长解析输入，静默产出错误结果。
- 移除"输出拷贝输入 shape"的骨架逻辑，改为 **numpy 右对齐广播**：
  - 输入 shape 按输出 rank 右对齐（低 rank 左侧补 1；rank-0 标量按 `{1}` 处理，复用骨架 `EnsureNotScalar` 思想）。
  - 逐维：`outDim = (dimA == dimB) ? dimA : (dimA == 1 ? dimB : (dimB == 1 ? dimA : 非法))`。
  - 任一维非法 → 返回 `ge::GRAPH_FAILED`。
- 输出 y shape = 广播结果。
- **rank-0 标量约定（v1.1 显式记录，与 numpy 有偏差）**：numpy 中两个 0-d 标量广播结果为 0-d；本算子沿用骨架 `EnsureNotScalar` 语义，统一输出 shape `[1]`（aclnn 调用方视角为 1 元素张量）。该偏差是既有骨架行为的延续，测试用例（PLAN T8）按 `[1]` 预期，保持一致。

### 3.2 TilingData 结构（`truncate_mod_tiling_data.h`）

```cpp
struct TruncateModTilingData {
    int64_t totalNum = 0;    // 广播后输出总元素数
    int64_t blockFactor = 1; // 每核元素数（512 对齐）
    int64_t ubFactor = 0;    // 每次 UB 循环元素数（256B 对齐）
    int64_t x1InnerLen = 0;  // x1 连续运行长度（元素）
    int64_t x1RowCount = 0;  // x1 源行数（= x1Total / x1InnerLen）
    int64_t x2InnerLen = 0;  // x2 连续运行长度（元素）
    int64_t x2RowCount = 0;  // x2 源行数（= x2Total / x2InnerLen）
};
```

Kernel 由 `innerLen × rowCount` 得 `inTotal`（重复周期），由 `inTotal == 1` 判标量。**workspace 保持 0**（trunc 走 Cast，无需 tmp/workspace）。

### 3.3 Tiling（`truncate_mod_tiling.cpp`）

1. 读取 x1/x2 shape，与输出 shape 一并计算 row-run 描述（§2.4.3）。
2. 多核切分（§2.1）：`coreNum → blockFactor → blockNum`。
3. UB 切分（§2.2）：按 dtype 模式取 `bufferDivisor`（mode0/1=24，mode2=28）→ `ubFactor`；广播时按 period 对齐优化。
4. tilingKey 选择（**由 2 值扩为 3 值**，按 x1 dtype 分派；x1/x2 dtype 一致性已在 InferShape 校验，此处无需重复判断）：
   - `DT_FLOAT16` → `TRUNCATEMOD_TPL_SCH_MODE_0`
   - `DT_BF16`   → `TRUNCATEMOD_TPL_SCH_MODE_1`
   - `DT_FLOAT`  → `TRUNCATEMOD_TPL_SCH_MODE_2`
5. `context->SetBlockDim(coreNum)`；`GetWorkspaceSize` 保持 0。

### 3.4 TilingKey（`truncate_mod_tiling_key.h`）

`ASCENDC_TPL_UINT_DECL` / `ASCENDC_TPL_UINT_SEL` 的 UI 列表由 `{MODE_0, MODE_1}` 扩展为 `{MODE_0, MODE_1, MODE_2}`（新增 `#define TRUNCATEMOD_TPL_SCH_MODE_2 2`）。

### 3.5 Kernel 入口与类模板（`truncate_mod.cpp` / `truncate_mod.h`）

```cpp
// truncate_mod.cpp 入口：三路 if constexpr 分派
if constexpr (schMode == MODE_0)      NsTruncateMod::TruncateMod<half> op;          // FP16
if constexpr (schMode == MODE_1)      NsTruncateMod::TruncateMod<bfloat16_t> op;    // BF16
if constexpr (schMode == MODE_2)      NsTruncateMod::TruncateMod<float> op;         // FP32
// 每路 op.Init(x1, x2, y, &tilingData); op.Process();
```

`TruncateMod<T>` 类：
- 成员：`TPipe pipe; TQue<VECIN, BUFFER_NUM> inQueueX / inQueueX2; TQue<VECOUT, BUFFER_NUM> outQueueY;` GlobalTensor x1/x2/y；FP32 工作缓冲 `x1F/x2F/qF`；tiling 字段。
- `Init`：解析 tilingData（含 row-run 描述、标量标志、period），`pipe.InitBuffer` 分配队列与工作缓冲（按 §1.5 公式与 `ubFactor`），`GlobalTensor` 关联 GM。
- `Process`：入口先判 **`if (blockIdx >= blockNum) return;`**（`blockNum = ceil(totalNum / blockFactor)` 可能小于实际核数 `coreNum`，尾部核必须提前退出，否则 `start = blockIdx × blockFactor` 越界执行 CopyIn/CopyOut）；`start = blockIdx × blockFactor`；主循环按 tile 执行 `CopyIn → Compute → CopyOut`（EnQue/DeQue 双缓冲流水）。
- `CopyIn`：双输入各按 §2.4.3 加载策略（策略 0/1/2/3；标量输入走策略 0，不 EnQue 对应输入队列）。
- `Compute`：按 mode（模板 T 决定，内部恒为 FP32 链）执行 §2.4.1/§2.4.2。
- `CopyOut`：`EnQue(yLocal)` → `DeQue(yLocal)` → `DataCopyPad(3 参)(yGm + (start+tileOff), yLocal, {1, currentNum × sizeof(T), 0, 0, 0})`（全模式从 outQueueY 输出）。

---

## 4. 确认清单

- [x] 多核切分策略已确定（§2.1：展平 1D + blockFactor 512 对齐 + 首尾核区分）
- [x] UB 切分策略已确定（§2.2：bufferDivisor 公式 + 256B 对齐 + period 对齐优化）
- [x] Buffer 规划已完成（§1.5：三队列双缓冲 + FP32 工作缓冲，mode0/1=24、mode2=28 B/elem）
- [x] 分支场景已覆盖（§2.3：3 dtype × 4 广播形态 × 对齐/边界/数值边界）
- [x] 类别特有设计已完成（§2.4：三 Compute 分支 + row-run 广播模型与 4 种加载策略）
- [x] 全部 API 已通过 asc-devkit 官方文档验证（§1.2.2 链接）；910b 硬约束（bf16 不可直算、Divs/Subs 不可用、DataCopyPad 负 stride 不可用、**GM→UB DataCopyPad 仅 4 参**、**Copy 仅 mask 形式且 repeatTime≤255**）已纳入设计
- [x] NpuArch / `__NPU_ARCH__` / `--npu-arch` 已通过 `/npu-arch` skill 查得（§0.3）
- [x] 精度策略已对照 ops-precision-standard 评估（§2.3：FP16/BF16 升精度、FP32 直算均达标；翻转概率按数据分布表述并约束验收数据域）
- [x] 设计串讲问题已闭环（v1.1）：mode2 输出走标准 outQueueY（§2.4.1）；GM→UB DataCopyPad 4 参 / Copy mask 形式 + repeatTime 分片（§2.4.3）；翻转概率数据分布相关（§2.3）；x1/x2 dtype 一致性校验（§3.1）；策略 2 适用范围与策略 3 段数修正（§2.2/§2.4.3）；标量分支明确（§2.4.2/§2.4.3）；blockIdx 越界防护（§3.5）
