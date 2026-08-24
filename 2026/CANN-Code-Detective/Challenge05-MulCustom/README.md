# Challenge 05 · MulCustom —— 逐元素乘法算子

## 题目背景

逐元素乘法（Element-wise Multiplication）是算子开发中最基础的二元操作之一。给定两个相同长度的输入张量 `x` 和 `y`，计算输出 `z = x * y`。

通过前两章课程，学习掌握了 Ascend C 的核函数开发和 Vector 编程范式（**CopyIn → Compute → CopyOut**），以及多核数据切分、Double Buffer 等核心概念，本题基于这些知识实现乘法算子开发，依然适合适合初学者入门学习。

## 算子描述

| 项目 | 说明 |
| :--- | :--- |
| 输入 | `x`：float32 张量，长度 `totalLength` |
| 输入 | `y`：float32 张量，长度 `totalLength` |
| 输出 | `z`：float32 张量，长度 `totalLength` |
| 计算 | `z[i] = x[i] * y[i]` |
| 数据量 | `totalLength = 8 * 2048 = 16384` |

## 涉及知识点

本题覆盖了 Ascend C 算子开发前两章的核心知识点：

- **第 1 章 - 基础概念**
  - AI Core 抽象架构：计算单元（Scalar / Vector / Cube）、存储单元（Local Memory）、搬运单元（DMA）
  - 计算数据流：DMA 搬运数据 → Vector 计算 → DMA 搬运结果
  - Global Memory 与 Local Memory 的分工

- **第 2 章 - 核函数与编程范式**
  - 核函数声明：`__global__ __aicore__` 和 `KERNEL_TASK_TYPE_DEFAULT`
  - Vector 编程范式：`CopyIn → Compute → CopyOut` 三段式流水线
  - Double Buffer 机制：`TQue` + `BUFFER_NUM = 2` 实现数据搬运与计算并行
  - 多核数据切分：`GetBlockNum()` / `GetBlockIdx()` / `SetGlobalBuffer()`
  - 基础 API：`DataCopy`、`Mul`、`AllocTensor`、`FreeTensor`、`EnQue`、`DeQue`
  - Host 侧调用：`aclrtMalloc`、`aclrtMemcpy`、kernel 调用语法 `<<<>>`

## 代码骨架

代码骨架位于 [Template/MulCustom/](./Template/MulCustom/) 目录，请复制一份并以你的 gitcode 用户名命名，例如：

```bash
cp -r Template/MulCustom <你的用户名>/MulCustom
```

然后在你自己的目录中完成实现如下3个文件，分别对应算子的核函数实现、构建脚本和编译运行脚本：

- [mul_custom.asc](./Template/MulCustom/mul_custom.asc) —— 算子实现骨架，按照代码中的 `// 请补充……` 注释提示完成实现
- [CMakeLists.txt](./Template/MulCustom/CMakeLists.txt) —— 构建脚本，无需修改
- [run.sh](./Template/MulCustom/run.sh) —— 编译运行脚本，可不用修改，如有修改 CANN 路径等特殊需求可根据实际情况修改

## 你需要完成的内容

打开 `mul_custom.asc`，按照代码中的 `// 请补充……` 注释提示，补充以下部分：

### Kernel 侧（`KernelMul` 类）

1. **`KernelMul::Init`** —— 初始化需要的对象：
   - 计算单核处理数据长度 `blockLength = totalLength / GetBlockNum()`
   - 计算 `tileLength`，考虑 `tileNum` 和 `BUFFER_NUM`（Double Buffer）
   - 使用 `SetGlobalBuffer` 设置每个核的 Global Memory 起始地址
   - 使用 `pipe.InitBuffer` 为队列分配内存

2. **`KernelMul::Process`** —— 主处理循环，按 tile 迭代执行 `CopyIn → Compute → CopyOut`

3. **`KernelMul::CopyIn`** —— 将 GM 数据搬入 Local Memory，并入队：
   - `AllocTensor` → `DataCopy` → `EnQue`

4. **`KernelMul::Compute`** —— 出队，执行逐元素乘法，入队：
   - `DeQue` → `Mul(zLocal, xLocal, yLocal, tileLength)` → `EnQue` → `FreeTensor`

5. **`KernelMul::CopyOut`** —— 出队，将结果从 Local Memory 搬回 GM：
   - `DeQue` → `DataCopy` → `FreeTensor`

6. **私有成员变量** —— 定义 `TPipe`、`TQue`、`GlobalTensor` 等

### Kernel 入口函数

7. **`mul_custom` kernel 函数** —— 实例化 `KernelMul` 并调用 `Init` + `Process`，添加 `KERNEL_TASK_TYPE_DEFAULT`

### Host 侧

8. **`kernel_mul` host 函数** —— 申请 GM 内存、搬运数据、调用 kernel 函数、回拷结果

### main 函数

9. **`main` 函数** —— 调用 `kernel_mul` 并验证结果

## 编程范式提示

```
GM (x, y) ──CopyIn──► UB (xLocal, yLocal) ──Compute──► UB (zLocal) ──CopyOut──► GM (z)
                  EnQue                       EnQue                    DeQue
```

- 使用 `GlobalTensor<float>` 管理 GM 数据
- 使用 `LocalTensor<float>` 管理 UB 数据
- 使用 `TQue<TPosition::VECIN, BUFFER_NUM>` 管理输入队列
- 使用 `TQue<TPosition::VECOUT, BUFFER_NUM>` 管理输出队列
- 使用 `DataCopy` 完成数据搬运
- 使用 `Mul` 指令完成逐元素乘法计算

### 参考代码流程（基于课程 Add 算子示例）

```cpp
// Init 中核心计算
this->blockLength = totalLength / AscendC::GetBlockNum();
this->tileNum = tileNum;
this->tileLength = this->blockLength / tileNum / BUFFER_NUM;
xGm.SetGlobalBuffer((__gm__ float *)x + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
yGm.SetGlobalBuffer((__gm__ float *)y + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
zGm.SetGlobalBuffer((__gm__ float *)z + this->blockLength * AscendC::GetBlockIdx(), this->blockLength);
pipe.InitBuffer(inQueueX, BUFFER_NUM, this->tileLength * sizeof(float));
pipe.InitBuffer(inQueueY, BUFFER_NUM, this->tileLength * sizeof(float));
pipe.InitBuffer(outQueueZ, BUFFER_NUM, this->tileLength * sizeof(float));

// Compute 中核心计算
AscendC::Mul(zLocal, xLocal, yLocal, this->tileLength);
```

> **注意**：`Mul` 接口的用法与课程中 `Add` 接口一致，区别在于计算语义从加法变为乘法。

## 如何运行

```bash
# 在 Challenge05-MulCustom 目录下执行
# 注意：在线环境一般不用修改脚本，可根据实际情况修改脚本中激活 can 环境的路径，source $ASCEND_TOOLKIT_HOME/set_env.sh

bash run.sh
```

> 如果 run.sh 在 Challenge05-MulCustom 根目录下执行有问题，也可进入你自己的目录直接运行：
> ```bash
> cd <你的用户名>/MulCustom && bash run.sh
> ```

预期输出：
```
Output: 2.76 2.76 2.76 2.76 2.76 2.76 2.76 2.76 2.76 2.76 2.76 2.76 2.76 2.76 2.76 2.76 2.76 2.76 2.76 2.76 ...
Golden: 2.76 2.76 2.76 2.76 2.76 2.76 2.76 2.76 2.76 2.76 2.76 2.76 2.76 2.76 2.76 2.76 2.76 2.76 2.76 2.76 ...
[Success] Case accuracy is verification passed.
```

## 评分标准

- ✅ 通过精度验证（输出 `[Success]`），在 PR 描述中提供通过截图
- ✅ 代码提交只需要包含 `mul_custom.asc`、`CMakeLists.txt`、`run.sh` 三个文件，不要提交build相关文件
- ✅ 代码风格规范，注释清晰

## 推荐学习路径

1. 先学习 [AscendC 算子开发基础概览（第 1 章）](https://gitcode.com/cann/cann-learning-hub/blob/master/tutorials/ascendc_operator_development/01_basic_overview/01.01_chapter_intro.ipynb)，了解 AI Core 架构和计算数据流
2. 再学习 [AscendC 算子基础编程（第 2 章）](https://gitcode.com/cann/cann-learning-hub/blob/master/tutorials/ascendc_operator_development/02_AscendC_basic/02.01_chapter_intro.ipynb)，掌握 Vector 编程范式
3. 参考第 2 章中 Add 算子的完整实现，完成 Mul 算子的开发
4. 如果遇到困难，可以回顾第 2 章课后练习（SubCustom / SigmoidCustom）的答案代码，或参考课程提供的模板
