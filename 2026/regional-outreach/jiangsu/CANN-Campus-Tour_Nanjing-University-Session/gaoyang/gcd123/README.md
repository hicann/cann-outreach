# Gcd123 自定义算子（AscendC）

基于 AscendC 框架开发的自定义算子 **Gcd123**：对两个 4 维 ND 张量按 broadcast 逐元素求最大公约数（GCD）。

## 算子规格

| 项 | 说明 |
|---|---|
| 算子名称 | `Gcd123`（核函数名 `gcd123`） |
| 输入 | `self`（REQUIRED）、`other`（REQUIRED） |
| 输出 | `out`（REQUIRED） |
| Shape | 均为 4 维 `[N4, N3, N2, N1]`，`self` 与 `other` 需满足标准 broadcast 关系；`out` 的 shape 即 broadcast 结果 |
| 数据类型 | `float16`（DT_FLOAT16） |
| Format | `ND` |

## 计算语义

```
out[i] = gcd(round(|self[i]|), round(|other[i]|))
```

- GCD 是整数运算：float16 值先取绝对值、四舍五入（0.5 远离零方向）到整数，再做欧几里得辗转相除，结果以 float16 写回。
- 零值自然处理：`gcd(0, a) == gcd(a, 0) == |a|`，`gcd(0, 0) == 0`。
- NaN 输入按 0 处理；±Inf 截断到 float16 最大值 65504。

## 目录结构

```
gcd123_op/
├── build.sh                  # 编译入口（需在装有 CANN 的 Linux 环境执行）
├── CMakeLists.txt            # 工程顶层 CMake
├── op_host/
│   ├── gcd123.cpp            # 原型注册 OpDef + InferShape(broadcast) + TilingFunc
│   ├── gcd123_tiling.h       # TilingData 定义（host/kernel 共用）
│   └── CMakeLists.txt
├── op_kernel/
│   ├── gcd123.cpp            # AscendC 核函数：broadcast 索引映射 + 欧几里得 GCD
│   └── CMakeLists.txt
└── tests/
    └── test_gcd123.py        # 基于 aclnn 接口的端到端验证脚本
```

## 关键设计

### broadcast 处理
- Host 侧 `InferShape` 逐维校验两个输入满足 broadcast 规则（相等、或其一为 1），否则返回 `GRAPH_FAILED`。
- Tiling 把三个 4D shape（self/other/out）全部下发；Kernel 侧将输出线性下标拆成 4 维下标 `(i4, i3, i2, i1)`，输入维度为 1 时该维下标折叠为 0，直接得到 self/other 的线性偏移。

### 多核切分与 64B 对齐
- 输出按线性下标均匀切到各 AI Core。
- GM 标量写（`GlobalTensor::SetValue`）经 64 字节 DataCache 行回写，不同核写同一 CacheLine 会互相覆盖脏行，因此每个核的输出区间按 32 个 float16（64B）对齐。
- `BLOCK_DIM`（op_host/gcd123.cpp）默认 20，可按目标平台核数（`GetCoreNumAiv`）调整。

### 性能说明
当前 Kernel 采用逐元素标量 GM 访问，正确性对所有 broadcast 形状成立、实现最简单。若 tensor 很大，可在不改变 tiling 约定的前提下改为 UB 流水线（DataCopy 搬块到 UB，UB 内标量/向量计算，再搬回），吞吐可显著提升。

## 编译与部署（CANN 环境）

```bash
export ASCEND_TOOLKIT_HOME=/usr/local/Ascend/ascend-toolkit/latest
source ${ASCEND_TOOLKIT_HOME}/bin/setenv.bash

cd gcd123_op
./build.sh ascend910b4          # 按实际 SoC 型号替换
./build_out/custom_opp_*.run    # 安装算子包
```

## 验证

```bash
python3 tests/test_gcd123.py
```

用例覆盖：
1. 同 shape：`[4, 3, 2, 2]` vs `[4, 3, 2, 2]`
2. 标量广播：`[4, 3, 2, 2]` vs `[1, 1, 1, 1]`
3. 逐维广播：`[4, 1, 2, 1]` vs `[1, 3, 1, 2]`

输出逐元素与 numpy 参考实现的 GCD 比对（全部一致则打印 PASS）。

> 注意：本工程需在装有 CANN Toolkit 的 Atlas 系列环境（如 Atlas A2）编译运行；本仓库交付时仅在代码层完成，尚未在真实 NPU 上跑过验证。
