# Challenge 01 · DivCustom —— 逐元素除法算子

## 题目背景

逐元素除法（Element-wise Division）是算子开发中最基础的二元操作之一。给定两个相同长度的输入张量 `x` 和 `y`，计算输出 `z = x / y`。

虽然逻辑简单，但它完整覆盖了 AscendC 算子开发的核心流程：**Tiling 切分 → 数据搬运（CopyIn）→ 计算（Compute）→ 结果搬出（CopyOut）**。

## 算子描述

| 项目 | 说明 |
| :--- | :--- |
| 输入 | `x`：float32 张量，长度 `totalLength` |
| 输入 | `y`：float32 张量，长度 `totalLength` |
| 输出 | `z`：float32 张量，长度 `totalLength` |
| 计算 | `z[i] = x[i] / y[i]` |
| 数据量 | `totalLength = 8 * 2048 = 16384` |

## 代码骨架

代码骨架位于 [Template/DivCustom/](./Template/DivCustom/) 目录，请复制一份并以你的 gitcode 用户名命名，例如：

```bash
cp -r Template/DivCustom <你的用户名>/DivCustom
```

然后在你自己的目录中完成实现：

- [div_custom.asc](./Template/DivCustom/div_custom.asc) —— 算子实现骨架，按照代码中的 `// 请补充……` 注释提示完成实现
- [CMakeLists.txt](./Template/DivCustom/CMakeLists.txt) —— 构建脚本，无需修改
- [run.sh](./Template/DivCustom/run.sh) —— 编译运行脚本，可不用修改，如有修改CANN路径等特殊需求可根据实际情况修改

## 你需要完成的内容

打开 `div_custom.asc`，按照代码中的 `// 请补充……` 注释提示，补充以下部分：

1. **`KernelDiv::Init`** —— 初始化需要的对象。
2. **`KernelDiv::Process`** —— 主处理循环，按 tile 迭代执行 `CopyIn → Compute → CopyOut`。
3. **`KernelDiv::CopyIn`** —— 将 GM 数据搬入 Unified Buffer（UB），并入队。
4. **`KernelDiv::Compute`** —— 出队，执行逐元素除法，入队。
5. **`KernelDiv::CopyOut`** —— 出队，将结果从 UB 搬回 GM。
6. **私有成员变量** —— 定义 `TPipe`、`TQue`、`TBuf` 等。
7. **`div_custom` kernel 函数** —— 实例化 `KernelDiv` 并调用 `Init` + `Process`。
8. **`kernel_div` host 函数** —— 申请 GM 内存、搬运数据、调用kernel函数、回拷结果。
9. **`main` 函数** —— 调用 `kernel_div` 并验证结果。

## 编程范式提示

```
GM (x, y) ──CopyIn──► UB (xLocal, yLocal) ──Compute──► UB (zLocal) ──CopyOut──► GM (z)
                  EnQue                       EnQue                    DeQue
```

- 使用 `GlobalTensor<T>` 管理 GM 数据
- 使用 `LocalTensor<T>` 管理 UB 数据
- 使用 `TQue<T, BUFFER_NUM>` 管理 UB 队列（Double Buffer）
- 使用 `DataCopy` 完成数据搬运
- 使用 `Div` 指令完成计算


## 如何运行

```bash
# 在根目录下编译运行
# 注意：在线环境一般不用修改脚本，可根据实际情况修改脚本中激活cann环境的路径，source $ASCEND_TOOLKIT_HOME/set_env.sh

bash run.sh
```

预期输出：
```
Output: 0.521739 0.521739 0.521739 0.521739 0.521739 0.521739 0.521739 0.521739 0.521739 0.521739 0.521739 0.521739 0.521739 0.521739 0.521739 0.521739 0.521739 0.521739 0.521739 0.521739 ...
Golden: 0.521739 0.521739 0.521739 0.521739 0.521739 0.521739 0.521739 0.521739 0.521739 0.521739 0.521739 0.521739 0.521739 0.521739 0.521739 0.521739 0.521739 0.521739 0.521739 0.521739 ...
[Success] Case accuracy is verification passed.
```

## 评分标准

- ✅ 通过精度验证（输出 `[Success]`），在PR描述中提供通过截图
- ✅ 代码风格规范，注释清晰

## 参考学习资料

如果对 Ascend C算子开发还不熟悉，建议先学习以下课程：

- [AscendC 算子开发基础概览](https://gitcode.com/cann/cann-learning-hub/blob/master/tutorials/ascendc_operator_development/01_basic_overview/01.01_chapter_intro.ipynb)
- [AscendC 算子基础编程](https://gitcode.com/cann/cann-learning-hub/blob/master/tutorials/ascendc_operator_development/02_AscendC_basic/02.01_chapter_intro.ipynb)

完成学习后，再回到本题目尝试独立实现。
