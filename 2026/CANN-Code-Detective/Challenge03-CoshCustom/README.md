# Challenge 03 · CoshCustom —— 双曲余弦算子

## 题目背景

双曲余弦（Hyperbolic Cosine，cosh）是常见的数学函数，在机器学习、信号处理等领域有广泛应用。其数学定义为：

$$
\cosh(x) = \frac{e^x + e^{-x}}{2}
$$

本题需要完成cosh算子开发。参与者需要继续巩固 Ascend C 算子开发的核心流程和 TBuf 的使用：**Tiling 切分 → 数据搬运（CopyIn）→ 计算（Compute）→ 结果搬出（CopyOut）**，完成 host 侧的 kernel 直调测试代码，体验从 kernel 侧到 host 侧的完整开发过程。


## 算子描述

| 项目 | 说明 |
| :--- | :--- |
| 输入 | `x`：float32 张量，长度 `totalLength` |
| 输出 | `y`：float32 张量，长度 `totalLength` |
| 计算 | `y[i] = (Exp(x[i]) + Exp(-x[i])) / 2` |
| 数据量 | `totalLength = 8 * 2048 = 16384` |

## 代码骨架

代码骨架位于 [Template/CoshCustom/](./Template/CoshCustom/) 目录，请复制一份并以你的 gitcode 用户名命名，例如：

```bash
cp -r Template/CoshCustom <你的用户名>/CoshCustom
```

然后在你自己的目录中完成实现：

- [cosh_custom.asc](./Template/CoshCustom/cosh_custom.asc) —— 算子实现骨架，按照代码中的 `// 请补充……` 注释完成实现，注释中的提示仅为参考，可自行实现相关逻辑
- [CMakeLists.txt](./Template/CoshCustom/CMakeLists.txt) —— 构建脚本，无需修改
- [run.sh](./Template/CoshCustom/run.sh) —— 编译运行脚本，可不用修改，如有修改CANN路径等特殊需求可根据实际情况修改

## 你需要完成的内容

打开 `cosh_custom.asc`，按照代码中的 `// 请补充……` 注释提示，补充以下部分：

1. **`KernelCosh::Init`** —— 初始化需要的对象，例如 `TBuf`用于存放类似 `Exp(x)` 的中间结果。
2. **`KernelCosh::Process`** —— 主处理循环，按 tile 迭代执行 `CopyIn → Compute → CopyOut`。
3. **`KernelCosh::CopyIn`** —— 将 GM 数据搬入 Unified Buffer（UB），并入队。
4. **`KernelCosh::Compute`** —— 出队，**使用 `Exp` 接口**按照 `cosh(x) = (Exp(x) + Exp(-x)) / 2` 计算结果，入队。中间结果存放在 `TBuf` 申请的临时 tensor 上。
5. **`KernelCosh::CopyOut`** —— 出队，将结果从 UB 搬回 GM。
6. **私有成员变量** —— 定义 `TPipe`、`TQue`等变量。
7. **`cosh_custom` kernel 函数** —— 实例化 `KernelCosh` 并调用 `Init` + `Process`。
8. **`kernel_cosh` host 函数** —— 申请 GM 内存、搬运数据、调用kernel函数、回拷结果。
9. **`main` 函数** —— 调用 `kernel_cosh` 并验证结果。

## 编程范式提示

```
GM (x) ──CopyIn──► UB (xLocal) ──Compute──► UB (yLocal) ──CopyOut──► GM (y)
              EnQue                  EnQue                   DeQue
```

- 使用 `GlobalTensor<T>` 管理 GM 数据
- 使用 `LocalTensor<T>` 管理 UB 数据
- 使用 `TQue<T, BUFFER_NUM>` 管理 UB 队列（Double Buffer）
- 使用 `DataCopy` 完成数据搬运
- **必须使用 `Exp` 指令**按照 `cosh(x) = (Exp(x) + Exp(-x)) / 2` 实现计算，**不得直接调用 `Cosh`**
- 使用 `TBuf` 管理计算过程中的中间变量（如`Exp(x)`、`Exp(-x)`）

### Compute 阶段详细流程

计算过程可以有多种实现方式，如下流程可作为参考：
```
         ┌── Exp ──► tmpTensor0 (TBuf tmpBuf0) ───────────────┐
xLocal ──┤                                                    ├── Add ──► Muls(0.5) ──► yLocal
         └── Muls(-1) ──► Exp ──► tmpTensor1 (TBuf tmpBuf1) ──┘
```

> 提示：可通过类似 `Muls(xLocal, xLocal, -1.0f)` 实现与标量的计算。

## 如何运行

```bash
# 在根目录下编译运行
# 注意：在线环境一般不用修改脚本，可根据实际情况修改脚本中激活cann环境的路径，source $ASCEND_TOOLKIT_HOME/set_env.sh

bash run.sh
```

预期输出：
```
Output: 122.348 122.348 122.348 122.348 122.348 122.348 122.348 122.348 122.348 122.348 122.348 122.348 122.348 122.348 122.348 122.348 122.348 122.348 122.348 122.348 ...
Golden: 122.348 122.348 122.348 122.348 122.348 122.348 122.348 122.348 122.348 122.348 122.348 122.348 122.348 122.348 122.348 122.348 122.348 122.348 122.348 122.348 ...
[Success] Case accuracy is verification passed.
```

## 评分标准

- ✅ 通过精度验证（输出 `[Success]`），在PR描述中提供通过截图
- ✅ 代码风格规范，注释清晰

## 参考学习资料

如果对 Ascend C算子开发还不熟悉，建议先学习以下课程：

- [AscendC 算子开发基础概览](https://gitcode.com/cann/cann-learning-hub/blob/master/tutorials/ascendc_operator_development/01_basic_overview/01.01_chapter_intro.ipynb)
- [AscendC 算子基础编程](https://gitcode.com/cann/cann-learning-hub/blob/master/tutorials/ascendc_operator_development/02_AscendC_basic/02.01_chapter_intro.ipynb)
- [章节实践：Sigmoid 算子](https://gitcode.com/cann/cann-learning-hub/blob/master/tutorials/ascendc_operator_development/02_AscendC_basic/02.05_chapter_test.ipynb) —— 参考样例
- [TBuf 的使用](https://gitcode.com/cann/asc-devkit/blob/9.1.0/docs/guide/%E7%AE%97%E5%AD%90%E5%AE%9E%E8%B7%B5%E5%8F%82%E8%80%83/SIMD%E7%AE%97%E5%AD%90%E5%AE%9E%E7%8E%B0/%E7%9F%A2%E9%87%8F%E7%BC%96%E7%A8%8B/TBuf%E7%9A%84%E4%BD%BF%E7%94%A8.md) —— 本题中间变量管理参考文档

完成学习后，再回到本题目尝试独立实现。
