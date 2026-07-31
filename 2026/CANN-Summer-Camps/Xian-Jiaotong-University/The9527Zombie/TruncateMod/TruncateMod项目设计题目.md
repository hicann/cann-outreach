# 项目设计题目：基于 Ascend C 的 TruncateMod 算子开发

**CANN SUMMER CAMPS 2026 · XJTU · 第3组**

## 一、项目题目

基于 CANN/Ascend C 完成 TruncateMod 算子的分析、设计、实现与验证。

## 二、项目目标

1. 理解截断取模（truncated modulo / fmod）的数学定义及其符号规则。
2. 完成 TruncateMod 的算子原型、Shape/Dtype 推导和 Tiling 设计。
3. 使用 Ascend C 实现支持 FLOAT16、FLOAT、BFLOAT16、INT32、INT8、UINT8 的 AI Core Kernel。
4. 支持 NumPy 风格广播、动态 shape、标量、空 Tensor 和非 32 Byte 对齐尾块。
5. 正确实现除零语义，与内置算子 golden 对齐。
6. 提供可复现的 CPU 参考测试和 ACLNN 端到端样例。
7. 完成算子设计文档、项目说明、测试报告等任务交付件。

## 三、输入输出

| 名称 | 类型 | 说明 |
|---|---|---|
| x1 | Tensor | 被除数 |
| x2 | Tensor | 除数 |
| y | Tensor | 截断取模结果 |

计算公式：`y = x1 - trunc(x1 / x2) * x2`，其中 `trunc()` 向零取整，余数符号跟随被除数 `x1`（等价 numpy/torch `fmod`）。

输出与 `x1`、`x2` 的 broadcast 结果同 shape，dtype 与 `x1` 一致（x1、x2 须同 dtype）。核心 AI Core 算子接收 ACLNN 层预处理后的同 shape、同 dtype 输入。

除零语义（`x2 == 0`）须与内置算子 golden 对齐：

| 输入类型 | 输出 |
|---|---|
| 浮点（FLOAT16 / FLOAT / BFLOAT16） | `NaN` |
| 有符号整型（INT32 / INT8） | `-1` |
| 无符号整型（UINT8） | `255` |

## 四、项目内容

### 4.1 Host 侧

- 定义算子输入、输出和平台配置（6 种 dtype、ND format，开启动态 shape / 动态 rank）。
- 校验 x1、x2 数据类型一致，校验 dtype 是否受支持。
- 完成输出 Shape（右对齐广播）与数据类型（取 x1）推导。
- 依据输入 dtype 生成 tiling key，区分 kernel 侧类型分支。
- 构建广播输出 shape 与各输入的广播 stride（被广播维 stride 置 0）。
- 按 32 Byte block 粒度、满核优先原则，根据元素总数选择 1～40 个 AI Vector 核；核间不能均分时余出的 block 分配到前几个核，非对齐尾块并入最后一个核。
- 依据 UB 容量与双缓冲计算每核长度和 Tile 长度。

### 4.2 Kernel 侧

- 采用 GM→UB→Vector→GM 的流水线。
- 使用双缓冲隐藏搬运时延。
- 依据 tiling key 在编译期分发到对应 dtype 的模板实现，避免运行期分支。
- 各 dtype 统一提升为 FLOAT 计算：`trunc(t)` 以无分支写法 `floor(max(t,0)) + ceil(min(t,0))` 实现，规避整型溢出；再按 `y = x1 - trunc(x1/x2) * x2` 求余。
- 通过 `CompareScalar(x2 == 0) + Duplicate + Select` 完成除零掩码替换（浮点 → NaN，有符号 → -1，无符号 → 255）。
- FLOAT16/BFLOAT16/INT 类型转为 FLOAT 进行计算，输出时再转换回原 dtype（注意各类型舍入模式；INT8/UINT8 经 half 中转）。
- 最后一块使用 `DataCopyPad` 按有效字节搬运，避免越界写入。

### 4.3 测试

- CPU 参考测试：边界、广播、标量、空 Tensor、除零、含负数、近整数商、非法参数。
- 精度对拍：以 numpy `fmod`（或内置算子）为 golden，覆盖 6 种 dtype × 多组 shape 逐元素比对。
- ACLNN 样例：在 Atlas A2/A3 环境编译安装后执行。
- 精度标准：逐元素结果与 CPU golden 一致，满足 CANN Judge 平台默认阈值。
- 性能标准：全核参与计算场景下性能不低于内置算子的 95%；小 shape（10μs 以下、相差 3μs 以内）以性能仿真图与分析结论证明与内置一致或更优。

## 五、预期成果

- 完整自定义算子源码。
- 算子设计文档。
- 项目设计题目说明。
- CPU 和 NPU 测试程序。
- 编译、安装、运行脚本。
- 验收测试报告。

## 六、团队提交位置

```text
2026/CANN-Summer-Camps/Xian-Jiaotong-University/The9527Zombie/TruncateMod
```
