# Challenge 02 · DivBf16Custom —— bfloat16 除法算子

## 题目背景

在 Challenge 01 中，我们实现了 float32 类型的逐元素除法。但在实际算子开发中，经常会遇到 **bfloat16**（BF16）数据类型——它是 AI 训练和推理中广泛使用的低精度浮点格式，只用 16 位存储，但保留了与 float32 相同的指数范围。

**Div 接口不支持直接对 bfloat16_t 操作**，因此需要先将 bfloat16_t 转为 float 进行计算，再将结果转回 bfloat16_t。这个转换过程中的中间结果需要临时内存存储，而 **TBuf** 就是 Ascend C 中管理临时内存的数据结构。

本题将数据类型改为 bfloat16_t，要求使用 TBuf 管理类型转换的临时内存，完成 bfloat16 除法算子。为了简化实现，本次不做tiling切分，一次性处理2048个数据。

## 算子描述

| 项目 | 说明 |
| :--- | :--- |
| 输入 | `x`：bfloat16_t 张量，长度 `totalLength` |
| 输入 | `y`：bfloat16_t 张量，长度 `totalLength` |
| 输出 | `z`：bfloat16_t 张量，长度 `totalLength` |
| 计算 | `z[i] = x[i] / y[i]`（需经 float 中间转换） |
| 数据量 | `totalLength = 2048`（无 Tiling，单次处理） |

## 计算流程

```
GM (x, y)          CopyIn           Compute                              CopyOut        GM (z)
 bfloat16_t  ──────────►  UB (xLocal, yLocal)  ──────────►  UB (zLocal)  ──────────►  bfloat16_t
                    DataCopy                   Cast/Div/Cast              DataCopy
                    EnQue                      TBuf(tmpBuf0, tmpBuf1)     DeQue
```

Compute 阶段详细流程：

```
bfloat16_t xLocal ──Cast(CAST_NONE)──► float tmpTensor0 ──┐
                                                          ├── Div ──► float tmpTensor0 ──Cast(CAST_RINT)──► bfloat16_t zLocal
bfloat16_t yLocal ──Cast(CAST_NONE)──► float tmpTensor1 ──┘
```

## 代码骨架

代码骨架位于 [Template/DivBf16Custom/](./Template/DivBf16Custom/) 目录，请复制一份并以你的 gitcode 用户名命名，例如：

```bash
cp -r Template/DivBf16Custom <你的用户名>/DivBf16Custom
```

然后在你自己的目录中完成实现：

- [div_bf16_custom.asc](./Template/DivBf16Custom/div_bf16_custom.asc) —— 算子实现骨架
- [CMakeLists.txt](./Template/DivBf16Custom/CMakeLists.txt) —— 构建脚本，无需修改
- [run.sh](./Template/DivBf16Custom/run.sh) —— 编译运行脚本，可不用修改，如有修改CANN路径等特殊需求可根据实际情况修改

## 你需要完成的内容

本题**已预填** host 侧函数和 kernel 入口函数 `div_bf16_custom`，参与者只需在 `KernelDivBf16` 类中补充以下 4 处（均按 `// 请补充……` 注释提示完成）：

### 1. 成员变量声明（TBuf 部分）

在 `KernelDivBf16` 类的 `private` 成员末尾声明两个 TBuf：
```cpp
AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf0;
AscendC::TBuf<AscendC::TPosition::VECCALC> tmpBuf1;
```

### 2. `Init` 中的 TBuf 初始化

使用 `pipe.InitBuffer` 初始化 `tmpBuf0`、`tmpBuf1`，大小为 `totalLength * sizeof(float)`。

### 3. `CopyIn` / `CopyOut`

按注释提示实现数据搬运。**本题无需 Tiling，直接使用整个 `totalLength` 一次性 `DataCopy`**：
- `CopyIn`：`AllocTensor` → `DataCopy` 从 GM 到 UB → `EnQue`
- `CopyOut`：`DeQue` → `DataCopy` 从 UB 到 GM → `FreeTensor`

### 4. `Compute`（核心）

完成 TBuf 获取和类型转换流程：

1. 从 `inQueueX`、`inQueueY` 中 `DeQue` 获取 `xLocal`、`yLocal`（bfloat16_t）
2. 从 `outQueueZ` 中 `AllocTensor` 获取 `zLocal`（bfloat16_t）
3. 从 `tmpBuf0`、`tmpBuf1` 中 `Get<float>` 获取临时 tensor
4. 使用 `Cast` 将 `xLocal`、`yLocal` 从 bfloat16_t 转为 float，存入 tmpTensor（`RoundMode::CAST_NONE`）
5. 使用 `Div` 执行 float 除法：`tmpTensor0 = tmpTensor0 / tmpTensor1`
6. 使用 `Cast` 将结果从 float 转回 bfloat16_t，存入 `zLocal`（`RoundMode::CAST_RINT`）
7. `EnQue(zLocal)`，`FreeTensor(xLocal)`、`FreeTensor(yLocal)`

## 关键知识点

### TBuf 与 TQue 的区别

TBuf主要是用于Vector计算申请的临时空间，不需要有释放动作。TBuf和TQue 相同和不同如下：  

**相同：** TBuf和TQue都通过InitBuffer来初始化内存  

**不同：**

* 获取内存，TBuf通过Get()，TQue通过AllocTensor()  

* TBuf分配的内存空间只参与计算，无法执行入队出队操作。TQue的出入队列EnQue和DeQue必须成对出现。  

* TBuf申请的内存无需释放，TQue申请内存AllocTensor和释放内存FreeTensor必须成对出现，

### Cast 接口

```cpp
// bfloat16_t → float
AscendC::Cast(dstFloat, srcBf16, AscendC::RoundMode::CAST_NONE, length);

// float → bfloat16_t
AscendC::Cast(dstBf16, srcFloat, AscendC::RoundMode::CAST_RINT, length);
```

- `CAST_NONE`：bfloat16 → float 时使用，表示舍入模式在转换有精度损失时使用CAST_RINT模式，在不涉及精度损失时不进行舍入
- `CAST_RINT`：float → bfloat16 时使用，表示舍入模式为向最近的偶数舍入

## 如何运行

```bash
# 在根目录下编译运行
# 注意：在线环境一般不用修改脚本，可根据实际情况修改脚本中激活cann环境的路径，source $ASCEND_TOOLKIT_HOME/set_env.sh

bash run.sh
```

预期输出：
```
Output: 0.519531 0.519531 0.519531 0.519531 0.519531 0.519531 0.519531 0.519531 0.519531 0.519531 0.519531 0.519531 0.519531 0.519531 0.519531 0.519531 0.519531 0.519531 0.519531 0.519531 ...
Golden: 0.519531 0.519531 0.519531 0.519531 0.519531 0.519531 0.519531 0.519531 0.519531 0.519531 0.519531 0.519531 0.519531 0.519531 0.519531 0.519531 0.519531 0.519531 0.519531 0.519531 ...
[Success] Case accuracy is verification passed.
```

> 注意：由于 bfloat16 精度限制，输出值与 Challenge 01 的 float 结果略有差异，这是正常的。

## 评分标准

- ✅ 通过精度验证（输出 `[Success]`），在PR描述中提供通过截图
- ✅ 正确使用 TBuf 管理临时内存
- ✅ 正确使用 Cast 接口进行类型转换

## 参考学习资料

- [TBuf 的使用](https://gitcode.com/cann/asc-devkit/blob/9.1.0/docs/guide/%E7%AE%97%E5%AD%90%E5%AE%9E%E8%B7%B5%E5%8F%82%E8%80%83/SIMD%E7%AE%97%E5%AD%90%E5%AE%9E%E7%8E%B0/%E7%9F%A2%E9%87%8F%E7%BC%96%E7%A8%8B/TBuf%E7%9A%84%E4%BD%BF%E7%94%A8.md) —— 本题核心参考文档
- [Ascend C 算子开发基础概览](https://gitcode.com/cann/cann-learning-hub/blob/master/tutorials/ascendc_operator_development/01_basic_overview/01.01_chapter_intro.ipynb)
- [Ascend C 算子基础编程](https://gitcode.com/cann/cann-learning-hub/blob/master/tutorials/ascendc_operator_development/02_AscendC_basic/02.01_chapter_intro.ipynb)
