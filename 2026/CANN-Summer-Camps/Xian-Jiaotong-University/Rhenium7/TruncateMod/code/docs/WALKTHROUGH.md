# TruncateMod 设计串讲质疑清单（WALKTHROUGH.md）

> 审查者：Ascend C 算子设计独立审查者
> 审查对象：`docs/DESIGN.md` v1.0（2026-07-31）
> 审查日期：2026-07-31
> 审查范围：§1（API 映射、数据流）、§2（架构设计、伪代码）为主；§0/§4/§5 仅作参考
> 验证源：`asc-devkit`（CANN 8.5.2）本地 API 文档 + `impl/` 实现 + 接口头文件

---

## 设计串讲

### 审查结论

- [ ] 设计可直接开发（无阻塞问题）
- [x] 设计需要修改后开发（有阻塞/讨论问题）
- [ ] 设计存在严重问题，无法开发

**一句话结论**：设计方向（FP32 全链计算、row-run 广播模型、4 种加载策略、3 路 tilingKey 分派）正确，绝大部分 API 选择经查证属实；但存在 **1 个阻塞问题（mode2 输出缓冲方案自相矛盾且就地输出缺失流水同步）** 和 **4 个需讨论问题（DataCopyPad 签名、Copy 签名、精度翻转风险、dtype 一致性校验）**，需 Architect 回应修正后进入开发。

---

### 已核验项（证据记录，供 Architect 参考）

以下 API 可行性主张经本地 asc-devkit 文档/头文件核实为**正确**：

| 设计主张 | 验证结果 | 依据 |
|---------|---------|------|
| `Div<T>(dst,src0,src1,count)` 910b 支持 half/float，**不支持 bf16** | ✅ 属实 | `SIMD-API/基础API/Memory矢量计算/基础算术/Div.md`「数据类型」：Atlas A2 支持 half、float |
| `Mul`/`Sub` 910b 支持 int16/half/int32/float，**不支持 bf16** | ✅ 属实 | 同上目录 `Mul.md` / `Sub.md`：Atlas A2 数据类型表 |
| `Cast<float,T>` 各转换组合（bf16→float CAST_NONE、half→float CAST_NONE、float→half/bf16 CAST_RINT、float→float CAST_TRUNC） | ✅ 属实 | `.../类型转换/Cast.md`「Atlas A2 数据类型组合表」 |
| 高阶 `Trunc` 的 220 实现等价 `Cast<float,float,CAST_TRUNC>` | ✅ 属实 | `impl/adv_api/detail/math/trunc/trunc_v220_impl.h`：`TruncCastForTrunc` 即 `Cast<float,float>(dst,src,CAST_TRUNC,...)`；`Trunc.md` 支持 src/dst 重叠；`GetTruncMaxMinTmpSize` 需 tmp 的说明属实 |
| 高阶 `Trunc` 910b 不支持 bfloat16_t（T 仅 half/float） | ✅ 属实 | `高阶API/数学计算/Trunc接口/Trunc.md`：`T 支持的数据类型为：half、float` |
| `Divs`（灵活标量位置）910b 不支持 | ✅ 属实 | `.../基础算术/Divs（灵活标量位置）.md`：Atlas A2 不支持 |
| `Duplicate<float>` 910b 支持 | ✅ 属实 | `.../数据填充/Duplicate.md`：Atlas A2 支持 float |
| `GetValue` 910b 支持（VECIN/VECCALC/VECOUT） | ✅ 属实 | `.../LocalTensor/GetValue.md` |
| `DataCopyPad<T>` 910b 支持 bf16 等全部目标 dtype | ✅ 属实 | `DataCopyPad（GMToUB非对齐数据搬运）.md`：Atlas A2 数据类型含 bfloat16_t |
| DataCopyPad 910b **不支持负 stride 重复同块**（负 stride 仅 950） | ✅ 属实 | `DataCopyPad（GMToUB...）.md` 示例3标注「仅 Ascend 950PR/Ascend 950DT 支持」 |
| DataCopyPad 非 32B 对齐 blockLen 自动 dummy 填充 | ✅ 属实 | `DataCopyPad（GMToUB...）.md` 功能说明 + 示例2 |
| UB→GM DataCopyPad 支持 VECIN→GM 数据通路、3 参原型存在 | ✅ 属实 | `DataCopyPad（UBToGM非对齐数据搬运）.md`：通路 VECIN->GM / VECOUT->GM，3 参原型 |
| 目标架构 `dav-2201` 编译参数 | ✅ 属实 | `asc-devkit/cmake/asc/asc_modules/CMakeASCInformation.cmake`：`ascend910b → dav-2201` |
| Buffer 数值示例（mode0/1≈8064、mode2≈6976） | ✅ 自洽 | 按 §1.5 公式验算：UB=192KB、extra≈1KB 时 mode0/1=8064、mode2=6976 |

---

### 质疑清单

#### 问题 1：mode2 输出缓冲方案自相矛盾；"就地输出" 缺失流水同步与缓冲复用约束（阻塞）
- **类别**：内存规划 / 伪代码可实现性
- **严重程度**：🔴 阻塞
- **设计文档位置**：DESIGN.md §1.5（Buffer 规划）与 §2.4.1（mode2 分支）互相冲突；§1.3（数据流）亦受影响
- **问题描述**：
  1. §1.5 的 mode2 公式 `bufferDivisor = 2×(4+4+4) + 4 = 28` 明确**计入** `outQueueY`（三队列 inQueueX/inQueueX2/outQueueY 各双缓冲），即 28 B/elem 的前提是 mode2 也分配 VECOUT 输出队列；
  2. §2.4.1 却称"y 就地写入 x1Local（VECIN 位置亦可作为 DataCopyPad 源）"，其 Buffer 需求只列 `inQueueX / inQueueX2 / qF`，未列 outQueueY。
  - 若按 **§2.4.1 就地方案**实现，存在两个正确性隐患（伪代码中均未处理）：
    - **V→MTE3 无同步**：标准队列模型中，输出缓冲的 V 完成信号由 `outQueueY.EnQue/DeQue` 提供；就地方案下 CopyOut（MTE3）直接读 `x1Local`（VECIN），没有 EnQue 信号保证 PIPE_V 已写完，MTE3 可能读到半写数据。
    - **缓冲复用竞态**：inQueueX 为 BUFFER_NUM=2 轮转。tile t 的 CopyIn(MTE2)→DeQue→Compute(V)→CopyOut(MTE3) 均占用同一缓冲；tile t+2 的 `AllocTensor` 会复用该缓冲，而队列机制只跟踪 MTE2→V 的完成，**不跟踪 MTE3 是否读完**。若不串行化，tile t+2 的 MTE2 写入会与 tile t 的 MTE3 读取竞争。
  - 若按 **§1.5 公式（含 outQueueY）**实现，则 §2.4.1 的伪代码应改为写入 yLocal，就地方案应删除——但当前文档没有说明。
- **审查者视角**：这是两条无法并存的实现路径，直接决定了 Compute/CopyOut/流水调度的写法；按任一伪代码原样开发都会出错或与内存规划不符，必须先裁决。
- **建议方案**：**推荐采用 outQueueY 标准输出路径**（与 §1.5 divisor=28 一致，改动最小）：mode2 Compute 改为
  `Sub(yLocal, x1Local, qF, currentNum);`（yLocal 为 VECOUT 队列缓冲），CopyOut 从 yLocal 走标准 `DeQue → DataCopyPad(yGm, yLocal, ...)`。就地方案需额外引入 `SetFlag/WaitFlag`（V_MTE3）并手工管理缓冲生命周期，复杂度高、收益仅 ~8 B/elem，不建议。请 Architect 明确裁决并同步修正 §1.3/§1.5/§2.4.1。

#### 问题 2：GM→UB DataCopyPad 在 2201 上无 3 参原型，伪代码缺 padParams 参数（需讨论）
- **类别**：API 可行性
- **严重程度**：🟡 需讨论（按伪代码原样不可编译，修正机械）
- **设计文档位置**：DESIGN.md §1.2.1、§1.2.2、§2.4.3（策略 1/2/3 的全部 GM→UB 调用）
- **问题描述**：设计在 §1.2.2 记录原型为 `DataCopyPad(dst, src, DataCopyExtParams{blockCount, blockLen, srcStride, dstStride, rsv})`（3 参），§2.4.3 所有 CopyIn 调用均写作 3 参形式 `DataCopyPad(local, inGm[srcOff], {1, T*sizeof(T), 0, 0, 0})`。经查证，**2201 上 GM→UB 方向只有 4 参原型**：
  ```cpp
  // include/basic_api/kernel_operator_data_copy_intf.h L403-406（__NPU_ARCH__ != 3510/5102 分支）
  template <typename T>
  __aicore__ inline __inout_pipe__(MTE2) void DataCopyPad(
      const LocalTensor<T>& dst, const GlobalTensor<T>& src, const DataCopyExtParams& dataCopyParams,
      const DataCopyPadExtParams<T>& padParams);
  ```
  3 参形式仅存在于 **UB→GM 方向**（`DataCopyPad(dst_GM, src_UB, dataCopyParams)`，同文件 L424-425，与 `DataCopyPad（UBToGM非对齐数据搬运）.md` 一致）。
- **影响分析**：CopyIn 全部 3 个加载策略（含标量策略 0）的伪代码在 2201 目标下无法通过编译，开发阶段必然返工。
- **建议方案**：所有 GM→UB 调用补第 4 参：`DataCopyPad<T>(local, inGm[srcOff], {1, T*sizeof(T), 0, 0, 0}, DataCopyPadExtParams<T>{})`（isPad=false、leftPadding/rightPadding=0，即默认"每个 blockLen 右侧 dummy 填充至 32B"行为，符合设计对非对齐 tile 的兜底预期）；UB→GM 的 CopyOut 保持 3 参即可。同步修正 §1.2.2 验证表与 §1.2.1 映射表。

#### 问题 3：策略 2 的 UB→UB Copy 伪签名与 910b 实际 API 不符（需讨论）
- **类别**：API 可行性
- **严重程度**：🟡 需讨论
- **设计文档位置**：DESIGN.md §1.2.2、§2.4.3 策略 2
- **问题描述**：设计将"整周期重复扩展"写作 `Copy(local, local, {blockCount=T/period, blockLen=period*sizeof(T)/32, srcStride=0, dstStride=0})`。经查证，910b（Atlas A2）上：
  - `Copy（UBToUB掩码式高维数据搬运）.md`：**支持**，原型为 `Copy(dst, src, mask, repeatTime, CopyRepeatParams{dstStride, srcStride, dstRepeatSize, srcRepeatSize})`（mask 单位元素数、各 stride/repeatSize 单位 DataBlock(32B)、repeatTime 为 **uint8_t**）；
  - `Copy（UBToUB连续数据搬运）.md`：`Copy(dst, src, count)` **仅 Ascend 950 支持**，910b 不支持。
  设计伪代码中的 `{blockCount, blockLen, ...}` 参数集合**不匹配任何 910b 原型**。
- **影响分析**：策略 2 无法按伪代码实现；且 `repeatTime` 为 uint8_t（最大 255），广播扩展倍数 `T/period > 255` 时单次 Copy 无法完成。
- **建议方案**：改用 mask 形式实现"读同块、写连续"：`Copy(local, local, /*mask=*/period(元素数), /*repeatTime=*/T/period, {dstStride=1, srcStride=1, dstRepeatSize=period*sizeof(T)/32, srcRepeatSize=0})`；当 `T/period > 255` 时对扩展分片多次调用 Copy，或直接退回策略 3。地址重叠方面，src==dst 下"第 r 次迭代写 [r·P,(r+1)·P)、读 [0,P)"满足通用地址重叠约束（单次迭代 100% 重叠或完全不相交、前序 dst 不与后序 src 重叠），可在文档中补充说明以打消实现疑虑。

#### 问题 4：trunc 翻转概率量化不严谨，精度验收存在偶发失败风险（需讨论）
- **类别**：精度风险
- **严重程度**：🟡 需讨论
- **设计文档位置**：DESIGN.md §1.4 关键要点 1、§2.3 精度策略
- **问题描述**：设计称"升 FP32 后 trunc 翻转概率约 6e-8/元素，满足 99% 通过率 + 绝对误差上限"。该数值隐含假设商 `q≈0.5`。实际翻转概率与商幅值成正比：`P(flip) ≈ 2^-23 × |q|`（FP32 Div 1ulp 误差 = q×2^-24，q 落在任一整数两侧 2×q×2^-24 区间内即翻转）。对随机数据（如 x1、x2 均匀采样），`E[q] = E[x1]×E[1/x2]` 通常达 5~50（取决于最小非零 x2 与数据范围），翻转概率升至 **~1e-6/元素**。按 PLAN §6.3 的大规模验收：
  - N=1M 元素时期望翻转次数 λ≈1~6，**大概率出现至少 1 次翻转**；
  - 一旦翻转，误差 ≈ |x2|，FP16 下可轻易 >0.1、FP32 下 >1e-2，**直接击穿 max_abs_error_limit**（FP32 数据动态范围大，风险最高）。
  - 翻转是 trunc 在整数边界不连续的固有属性，任何有限精度实现相对 float64 golden 都无法根除；设计对此的"达标"论证未覆盖数据分布与测试规模的影响。
- **影响分析**：Step 6 精度验收可能偶发 FAIL（flaky），导致不必要的修复循环与验收反复。
- **建议方案**：a) 在 gen_data.py 中明确定义测试数据分布，并用 `λ = N × 2^-23 × E[q]` 预估期望翻转数，保证 λ≪1（如限定 |q| 范围、x2 下界，或数据取整数/有限小数域）；b) 对翻转元素在 compare_data.py 中做可解释豁免或按 golden 分界对齐策略处理；c) 将 §2.3 的定量描述改为"翻转概率依赖数据分布，验收数据需保证 λ≪1"的表述，避免给人"任何随机数据都必然达标"的错误预期。

#### 问题 5：未校验 x1/x2 的 dtype 一致性，混合 dtype 输入将静默出错（需讨论）
- **类别**：伪代码可实现性 / 分支覆盖
- **严重程度**：🟡 需讨论
- **设计文档位置**：DESIGN.md §1.1、§3.1、§3.3
- **问题描述**：op_def（`op_host/truncate_mod_def.cpp`）中 x1/x2 的 DataType 为各自独立的 `{DT_BF16, DT_FLOAT16, DT_FLOAT}`，未声明两者必须相同；设计 §3.3 的 tilingKey 仅由 `inputDesc(0)`（x1）的 dtype 决定。若调用方传入 x1=FP16、x2=FP32，tiling 会按 half 模式运行，以 2B 步长解析 x2 的 4B 数据，输出静默错误结果。
- **影响分析**：aclnn 通路（PLAN §3.4）无混合 dtype 用例，问题在 UT 中不易暴露，但真实调用方传入混合 dtype 时结果不可预期。
- **建议方案**：在 InferShape 或 Tiling 入口校验 `x1.dtype == x2.dtype`，不一致返回 `GRAPH_FAILED`；并在 PLAN 测试计划中补充 1 个负例用例。

#### 问题 6：多核 blockFactor 与广播 period 不对齐，"策略 2 覆盖主导"被高估（建议）
- **类别**：多核策略 / 性能
- **严重程度**：🟢 建议
- **设计文档位置**：DESIGN.md §2.1、§2.2、§2.4.3
- **问题描述**：§2.2 只把 `ubFactor` 对齐到 period 整数倍，但策略 2 的触发条件 `s % period == 0` 中的 `s = blockIdx × blockFactor + k×ubFactor`；`blockFactor` 为 512 元素对齐，**不保证是 period 的整数倍**。对 `blockStart % period != 0` 的核，其所有 tile 的 `s % period` 均不为 0 → 整核退化为策略 3。§2.4.3"每 tile 至多跨 1 个 period 边界、最多 2 段"的前提是 tile 已 period 对齐，实际未对齐时（T=ubFactor 为数倍 period）段数可达 `T/period + 1`。
- **影响分析**：策略 3 多段 DataCopyPad 增加搬运指令数，影响广播大 shape 性能；不影响正确性。
- **建议方案**：文档明示策略 2 仅在 `blockStart % period == 0` 的核上生效；如需提升覆盖率，可将 blockFactor 对齐到 period 整数倍（代价：轻微负载不均），或接受策略 3 兜底并修正"最多 2 段"的表述。

#### 问题 7：mode2 标量输入的 Duplicate 目标缓冲未指定；Compute 伪代码缺标量分支（建议）
- **类别**：伪代码可实现性
- **严重程度**：🟢 建议
- **设计文档位置**：DESIGN.md §2.4.3 策略 0、§2.4.1/§2.4.2
- **问题描述**：策略 0 写"`Duplicate<float>(xF, scalar, count)`"，但 **mode2 没有 x1F/x2F 工作缓冲**，Duplicate 目标不明；mode0/1 的 Compute 伪代码无条件执行 `Cast<float,T>(x1F, x1Local, ...)`，对标量输入（数据已由 Duplicate 直接写入 xF）会重复/错误处理，伪代码未体现标量分支。另外 GetValue 返回 T（half/bf16），转 float 标量的转换步骤也未说明。
- **建议方案**：明确 mode2 标量展开目标为输入队列缓冲（如 x2 标量 → `Duplicate<float>(x2Local, scalar, currentNum)` 后 `Div(qF, x1Local, x2Local)`）；Compute 中按标量标志跳过对应输入的 Cast 步骤；补充 T→float 标量转换说明。

#### 问题 8：blockIdx ≥ blockNum 的核退出分支未声明（建议）
- **类别**：多核策略
- **严重程度**：🟢 建议
- **设计文档位置**：DESIGN.md §3.5 Process
- **问题描述**：`blockNum = ceil(totalNum / blockFactor)` 可能小于 `coreNum`（小 shape 或对齐舍入后），尾部核若未提前返回，会按 `start = blockIdx × blockFactor` 越界执行 CopyIn/CopyOut。
- **建议方案**：Process 开头增加 `if (blockIdx >= blockNum) return;`，并在文档中注明。

#### 问题 9：尾 tile CopyOut 的 DataCopyPad 会向 GM 写出 32B 对齐填充字节（建议）
- **类别**：伪代码可实现性
- **严重程度**：🟢 建议
- **设计文档位置**：DESIGN.md §2.4.1/§2.4.2 CopyOut
- **问题描述**：`totalNum` 非 32B 倍数时（如 PLAN K9 的 1003 元素），最后一次 UB→GM DataCopyPad 的 `blockLen = currentNum×sizeof(T)` 非 32B 对齐，硬件会向 GM 写出最多 31B 的填充数据，位于 y 逻辑末尾之后，依赖 GM 分配器的对齐余量。
- **影响分析**：若运行时对 y 的分配恰好无余量，存在越界写相邻内存的隐患；通常 Ascend 运行时按 32B/512B 对齐分配，风险较低，但需确认。
- **建议方案**：确认运行时 GM 分配对齐行为（a）满足；或在文档中记录该依赖；UT 侧 compare_data 只读有效长度不受影响。

#### 问题 10：rank-0 标量广播输出语义与 numpy 有偏差（建议）
- **类别**：分支覆盖
- **严重程度**：🟢 建议
- **设计文档位置**：DESIGN.md §3.1
- **问题描述**：numpy 中两个 0-d 标量广播结果为 0-d，设计统一按 `{1}` 输出 shape [1]（复用骨架 `EnsureNotScalar`）。对纯 aclnn 调用方，返回 shape 与 numpy 语义不一致。
- **建议方案**：明确接受该约定（与骨架行为延续一致）并在文档/用例中记录；PLAN T8 用例已按 [1] 预期，保持一致即可。

---

## 汇总

| # | 问题 | 严重程度 | 类别 |
|---|------|---------|------|
| 1 | mode2 输出缓冲方案矛盾 + 就地输出缺流水同步/缓冲复用约束 | 🔴 阻塞 | 内存规划 / 伪代码可实现性 |
| 2 | GM→UB DataCopyPad 2201 无 3 参原型，缺 padParams | 🟡 需讨论 | API 可行性 |
| 3 | 策略 2 Copy 伪签名与 910b mask 形式不符；repeatTime≤255 | 🟡 需讨论 | API 可行性 |
| 4 | trunc 翻转概率量化不严谨，大规模验收有偶发失败风险 | 🟡 需讨论 | 精度风险 |
| 5 | x1/x2 dtype 一致性未校验 | 🟡 需讨论 | 伪代码可实现性 |
| 6 | blockFactor 与 period 不对齐，策略 2 覆盖被高估 | 🟢 建议 | 多核策略 |
| 7 | mode2 标量 Duplicate 目标未指定；Compute 缺标量分支 | 🟢 建议 | 伪代码可实现性 |
| 8 | blockIdx ≥ blockNum 核退出分支未声明 | 🟢 建议 | 多核策略 |
| 9 | 尾 tile CopyOut 向 GM 写 32B 填充字节 | 🟢 建议 | 伪代码可实现性 |
| 10 | rank-0 标量输出语义与 numpy 偏差 | 🟢 建议 | 分支覆盖 |

**阻塞/讨论问题合计 5 项**，按流程应进入 Step 2.5c 由 Architect 串讲回应并更新本清单。

---

## 设计串讲仲裁

### Architect 回应

> 回应者：Ascend C 算子架构设计
> 回应对象：上表 10 项质疑 ｜ 设计版本：v1.0 → **v1.1**（已同步更新 `docs/DESIGN.md`）
> 验证源：asc-devkit（CANN 8.5.2）本地 API 文档 + 头文件 + tiling-design skill

#### 问题 1：mode2 输出缓冲方案自相矛盾；"就地输出" 缺失流水同步（阻塞）
- **回应**：接受
- **理由**：两条实现路径确实无法并存，§1.5（divisor=28 计入 outQueueY）与 §2.4.1（y 就地写 x1Local）冲突，必须裁决。采纳 Reviewer 推荐的标准 outQueueY 路径：mode2 的 Compute 改为 `Sub(yLocal, x1Local, qF, currentNum)`，CopyOut 走 `EnQue(yLocal) → DeQue(yLocal) → DataCopyPad(3 参)`。就地方案的两个隐患（V→MTE3 无同步、BUFFER_NUM=2 轮转下 MTE3 读完时序不被队列跟踪）确认成立，需额外 `SetFlag/WaitFlag(V_MTE3)` 手工同步且收益仅 ~8 B/elem，否决。
- **文档依据**：`DataCopyPad（UBToGM非对齐数据搬运）.md` 数据通路含 VECIN→GM / VECOUT→GM（就地方案物理可行但同步缺陷属实）；流水同步由 `TQue.EnQue/DeQue` 队列机制承担（标准队列模型）。
- **DESIGN.md 变更**：§1.3 数据流（mode2 走 VECOUT 输出，附全模式统一输出路径说明）；§1.4 步骤 7/8 与要点 1；§1.5 设计决策注记；§2.4.1 伪代码与 Buffer 需求整体重写（Sub 写入 yLocal + 决策理由 3 条）；§3.5 CopyOut。

#### 问题 2：GM→UB DataCopyPad 在 2201 上无 3 参原型（需讨论）
- **回应**：接受
- **理由**：属实。`DataCopyPad（GMToUB非对齐数据搬运）.md`「函数原型」章节确认 910b（Atlas A2）**只有 4 参原型** `DataCopyPad(dst, src, dataCopyParams, padParams)`；3 参形式仅存在于 UB→GM 方向（`DataCopyPad（UBToGM）.md`）。v1.0 的 §1.2.2 记录 3 参原型是文档误读，CopyIn 全部 4 个加载策略按 3 参写必然编译失败。
- **文档依据**：`.../数据搬运/DataCopyPad（GMToUB非对齐数据搬运）.md` 函数原型（910b 分支）；`include/basic_api/kernel_struct_data_copy.h` `DataCopyPadExtParams<T>{}` 默认 `isPad=false, leftPadding=0, rightPadding=0` → 每个 blockLen 右侧自动 dummy 填充至 32B，与设计对非对齐 tile 的兜底预期一致，无需显式传 padParams 值。
- **DESIGN.md 变更**：§1.2.1 映射表（GM→UB 4 参 / UB→GM 3 参两行分离）；§1.2.2 验证表；§2.4.3 策略 0/1/2/3 全部 GM→UB 调用补第 4 参 `DataCopyPadExtParams<T>{}`，并加注"全部 GM→UB 为 4 参、UB→GM 为 3 参"。

#### 问题 3：策略 2 的 UB→UB Copy 伪签名与 910b 实际 API 不符（需讨论）
- **回应**：接受（并在 Reviewer 建议基础上做了更严格的实现修正）
- **理由**：属实。`Copy（UBToUB掩码式高维数据搬运）.md` 确认 910b 原型为 `Copy<T>(dst, src, mask, repeatTime, CopyRepeatParams{dstStride, srcStride, dstRepeatSize, srcRepeatSize})`（mask 连续模式=每次迭代元素数，repeatTime 为 uint8_t，stride/repeatSize 单位 32B）；`Copy（UBToUB连续数据搬运）.md` 确认 3 参连续搬运 `Copy(dst,src,count)` **仅 950 支持**。v1.0 的 `{blockCount, blockLen, ...}` 参数集合不匹配任何 910b 原型。
  实现修正（比 Reviewer 建议更严格）：为避免"迭代 0 的 dst=[0,P) 与后续迭代 src=[0,P) 重叠"这一对「通用地址重叠约束」（多次迭代间前序 dst 不得与后序 src 重叠）的模糊地带，**首份 [0,P) 不参与 Copy 回写**（DataCopyPad 已搬入真实输入），Copy 只负责把 [0,P) 复制扩展到 [P,T)：`Copy(local + (off+1)·P, local, mask=P, repeatTime=n, {1, 1, P·sizeof(T)/32, 0})`。此时所有 Copy 迭代的写区域 `[(i+1)P, (i+2)P)` 与读区域 `[0,P)` 完全不相交，单次迭代 100% 重叠或完全不相交、前序 dst 不与后序 src 重叠均严格满足。
  repeatTime≤255 约束：`k = T/period > 256` 时按 255 分片循环调用（`off += 255`），与策略 2 触发条件一并写入伪代码。
- **文档依据**：`Copy（UBToUB掩码式高维数据搬运）.md`（函数原型/参数/示例2）+ `Copy（UBToUB连续数据搬运）.md`（产品支持：910b 不支持）+ `SIMD-API/通用说明和约束.md`「通用地址重叠约束」+ tiling-design `references/broadcast/ub-broadcast.md`（srcStride=0/srcRepeatSize=0 重复读同一行同款扩展模式）。
- **DESIGN.md 变更**：§1.2.1/§1.2.2 的 Copy 行改为 mask 形式签名并注明 repeatTime 上限与 910b 无 3 参连续搬运；§2.4.3 策略 2 整体重写（首份就位 + 分片 Copy + 重叠约束论证 + 参数注释）。

#### 问题 4：trunc 翻转概率量化不严谨（需讨论）
- **回应**：接受
- **理由**：属实。v1.0 的"翻转概率约 6e-8/元素"隐含假设 |q|≈0.5，未覆盖数据分布与测试规模。正确公式为单元素翻转概率 `P ≈ 2^-23 × |q|`（FP32 Div 相对误差 1ulp=|q|×2^-24，q 落在任一整数两侧 2|q|×2^-24 区间即翻转），与商幅值线性相关；随机大范围数据下 E|q| 可达 5~50，N=1M 时 λ=N×2^-23×E|q|≈1~6，偶发击穿 max_abs_error_limit 风险真实存在。已按 Reviewer 建议 a/c 落地：§2.3 改为数据分布相关表述，并规定验收数据须保证 λ≪1（限定 |q| 上界 / x2 下界 / 整数域），K8 类大商值用例仅做功能性校验不纳入精度指标（建议 b 的"翻转元素豁免"不采纳——翻转误差幅值 |x2| 可达 0.1 量级，豁免会掩盖真实实现缺陷，且数据域受控后无需豁免）。
- **文档依据**：`ops-precision-standard` 浮点类混合容差（max_abs_error_limit 判定）；翻转为 trunc 整数边界不连续的固有属性（数学语义，PLAN §3.1 golden 用 float64 `np.trunc` 亦同）。
- **DESIGN.md 变更**：§1.4 关键要点 1（翻转概率改 `2^-23×|q|` 并强调数据相关）；§2.3 精度策略表（FP16 行改用新公式）+ 新增「trunc 翻转概率与验收数据分布（重要，v1.1 修正）」说明块（3 条落地约束）。

#### 问题 5：未校验 x1/x2 的 dtype 一致性（需讨论）
- **回应**：接受
- **理由**：属实。op_def 中 x1/x2 的 dtype 集合各自独立，混合 dtype（x1=FP16、x2=FP32）可被框架接受，而 tilingKey 仅按 x1 dtype 分派，会以错误步长解析 x2 → 静默错误。在 InferShape 入口增加 `x1.dtype != x2.dtype → GRAPH_FAILED`（校验前置到最早上游，aclnn/UT 均生效）。
- **文档依据**：`op_host/truncate_mod_def.cpp`（x1/x2 DataType 独立集合）；设计 §3.3 tilingKey 仅由 inputDesc(0) 决定。
- **DESIGN.md 变更**：§3.1 新增「输入 dtype 一致性校验」条目；§3.3 第 4 步加注"一致性已在 InferShape 校验"。PLAN 侧建议补充负例用例（T 系列）由 Developer 落地。

#### 问题 6：多核 blockFactor 与 period 不对齐，"策略 2 覆盖主导"被高估（建议）
- **回应**：部分修改
- **理由**：诊断成立，但"将 blockFactor 对齐到 period 整数倍"的建议**不采纳**：本算子有两个输入、两个不同 period，blockFactor 无法同时对齐二者（取 lcm 可能远超数据量或破坏 512 对齐/负载均衡）。采纳文档修正方向：① 明确策略 2 仅 `blockStart % period == 0` 的核上生效，其余核整核退化为策略 3（正确性不受影响）；② 修正策略 3 段数表述：tile 长度 T=k×period 时段数 = k 或 k+1，上界 `T/period + 1`，v1.0 的"最多 2 段"仅在 T≤2×period 时成立。
- **文档依据**：§2.4.3 策略 2 触发条件本身含 `s % period == 0`（v1.0 已有，但 §2.2 的"覆盖主导"表述与之矛盾）；`s = blockIdx×blockFactor + k×ubFactor`，blockFactor 为 512 元素对齐（§2.1）。
- **DESIGN.md 变更**：§2.2 广播对齐优化行（tile 长度对齐 ≠ 起点对齐、双 period 说明）；§2.4.3 策略 2 适用性注记 + 策略 3 段数上界修正。

#### 问题 7：mode2 标量输入的 Duplicate 目标缓冲未指定（建议）
- **回应**：接受
- **理由**：属实。v1.0 策略 0 写 `Duplicate<float>(xF, ...)` 但 mode2 无 xF 工作缓冲，且 Compute 伪代码无条件执行 Cast。已明确：mode0/1 的 Duplicate 目标为 FP32 工作缓冲 x1F/x2F（Compute 跳过对应 Cast）；mode2 目标为输入队列缓冲（x1Local/x2Local）；标量输入不 EnQue/DeQue 对应输入队列；T→float 标量转换用 C++ 显式转换（half/bfloat16_t 转换运算符，属标量语义非 API）。
- **文档依据**：`GetValue.md`（返回 T）；half/bfloat16_t 为 C++ 类类型提供到 float 的转换运算符（类型语义，非 AscendC API）。
- **DESIGN.md 变更**：§1.4 要点 5（新增标量处理说明）；§2.4.2 伪代码第 1 步改条件 Cast + 新增「标量输入（策略 0）细节」段；§2.4.3 策略 0 注释明确 Duplicate 目标按 mode。

#### 问题 8：blockIdx ≥ blockNum 的核退出分支未声明（建议）
- **回应**：接受
- **理由**：属实。`blockNum = ceil(totalNum / blockFactor)` 在小 shape 下可能小于 `coreNum`，尾部核不提前退出会按 `start = blockIdx × blockFactor` 越界执行。Process 入口增加 `if (blockIdx >= blockNum) return;`。
- **文档依据**：§2.1 多核切分公式（blockNum 由 totalNum/blockFactor 推出，与 coreNum 解耦）。
- **DESIGN.md 变更**：§3.5 Process 入口防护说明。

#### 问题 9：尾 tile CopyOut 会向 GM 写出 32B 对齐填充字节（建议）
- **回应**：保留原设计
- **理由**：该质疑的前提不成立。`DataCopyPad（UBToGM非对齐数据搬运）.md`「功能说明」与图2 明确：UB→GM 方向非 32B 对齐时，框架在**读 UB 时**补 dummy 对齐到 32B，**搬入 GM 时自动将 dummy 空数据丢弃**，且 GM 目的地址只需 1 字节对齐。因此尾 tile `blockLen = currentNum×sizeof(T)` 非 32B 对齐时**不会向 GM 写出填充字节**，不存在越界写相邻内存的依赖，也无需依赖 GM 分配器对齐余量。v1.1 已在设计中记录该语义以打消实现疑虑。
- **文档依据**：`.../数据搬运/DataCopyPad（UBToGM非对齐数据搬运）.md` 功能说明（"搬入Global Memory时会将dummy空数据丢弃"）+ 图2（blockLen=47 非对齐场景）。
- **DESIGN.md 变更**：§1.2.1/§1.2.2 的 UB→GM DataCopyPad 行补充"写出时 dummy 丢弃，不污染 GM"说明（无功能性变更）。

#### 问题 10：rank-0 标量广播输出语义与 numpy 有偏差（建议）
- **回应**：保留原设计
- **理由**：该偏差是既有骨架 `EnsureNotScalar` 行为的延续，统一输出 shape `[1]` 可避免 Kernel/Tiling 对 0-d 的额外分支；numpy 0-d→0-d 语义对 aclnn 调用方影响极小且与骨架保持一致。采纳"在文档/用例中显式记录"的建议，已在 §3.1 记录约定；PLAN T8 继续按 `[1]` 预期，无需变更。
- **文档依据**：骨架 `EnsureNotScalar` 语义（既有工程行为）；PLAN T8 用例预期。
- **DESIGN.md 变更**：§3.1 新增「rank-0 标量约定（v1.1 显式记录，与 numpy 有偏差）」条目（无功能性变更）。

### 回应统计
- 接受 7 项（问题 1、2、3、4、5、7、8），保留 2 项（问题 9、10），部分修改 1 项（问题 6）
- DESIGN.md **已更新**：v1.0 → v1.1（§1.2/§1.3/§1.4/§1.5/§2.2/§2.3/§2.4.1/§2.4.2/§2.4.3/§3.1/§3.3/§3.5/§4 均含变更，头部附变更摘要）
- 遗留分歧：无。阻塞问题 1 已裁决（标准 outQueueY 路径），进入 Step 3 开发。
